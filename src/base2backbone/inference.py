import os
import os.path as osp
import tempfile
import warnings
from argparse import ArgumentParser
from typing import Any, Tuple, cast

import MDAnalysis as mda
import numpy as np
import torch
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser
from MDAnalysis.coordinates.memory import MemoryReader
from torch_geometric.data import Batch

from .data import (
    BACKBONE_ATOMS,
    CHAIN_END_CLASS_3_PRIME,
    CHAIN_END_CLASS_5_PRIME,
    FIVE_PRIME_PHOSPHATE_ATOMS,
    parse_dna,
    parse_dna_universe,
)
from .geometry import build_batch_window_backbone_from_torsions
from .io import default_atoms_provider, inference_atoms_provider
from .onnx_inference import OnnxSampler
from .runtime import MODEL_DIR, PROGRESS_BAR_COLOR
from .torsion_constants import N_TORSIONS, TAU_M_MAX, TAU_M_MIN, TOR_ALPHA, TOR_EPS, TOR_ZETA
from tqdm import tqdm

WINDOW_SIZE = 3


def _load_model(model_path, device):
    return OnnxSampler(model_path, device=device)


def _chain_list_direction(chain):
    """Same convention as ``parse_dna``: +1 if resid increases toward 3' in list order."""
    first_resid = chain[0].e_residue.resids[0]
    last_resid = chain[-1].e_residue.resids[0]
    return 1 if last_resid >= first_resid else -1


def _merge_window_pred_for_residue(
    predictions: dict,
    window_pred: dict[Any, Any],
    nt,
) -> None:
    """Write backbone entries from ``window_pred`` only for residue ``nt`` (one chain step)."""
    sk = (nt.segid, int(nt.resid))
    for key, val in window_pred.items():
        if key[0] == sk[0] and key[1] == sk[1]:
            predictions[key] = val


@torch.no_grad()
def _prepare_target_view(sample_data, target_idx: int):
    dc = sample_data.clone()
    dc.target_nt_idx = torch.tensor(target_idx, dtype=torch.long)
    is_target = torch.zeros(WINDOW_SIZE, dtype=torch.float32)
    is_target[target_idx] = 1.0
    dc.is_target_nt = is_target
    origin_t = dc.nt_origins_world[target_idx]
    frame_t = dc.nt_frames_world[target_idx]
    dc.rel_origins = ((dc.nt_origins_world - origin_t) @ frame_t).float()
    dc.rel_frames = torch.einsum('ji,njk->nik', frame_t, dc.nt_frames_world).float()
    dc.pair_rel_origins = ((dc.pair_origins_world - origin_t) @ frame_t).float()
    dc.pair_rel_frames = torch.einsum('ji,njk->nik', frame_t, dc.pair_frames_world).float()
    return dc


def _window_predictions_from_coords(window, coords: np.ndarray) -> dict[Any, Any]:
    preds: dict[Any, Any] = {}
    for local_i, nt in enumerate(window):
        row_w = coords[local_i]
        for j_atom, atom_name in enumerate(BACKBONE_ATOMS):
            pos = row_w[j_atom]
            if np.isfinite(pos).all():
                preds[(nt.segid, int(nt.resid), atom_name)] = pos
    return preds


@torch.no_grad()
def _predict_full_window_predictions_dicts(model, window_jobs, device) -> dict[tuple[int, str, int], dict[Any, Any]]:
    """Infer all targets for all windows in one batch and decode the windows together."""
    if not window_jobs:
        return {}

    target_samples = []
    sample_keys: list[tuple[int, int]] = []
    for job_idx, (_, _, _, _window, data) in enumerate(window_jobs):
        for target_idx in range(WINDOW_SIZE):
            target_samples.append(_prepare_target_view(data, target_idx))
            sample_keys.append((job_idx, target_idx))

    batch = cast(Any, Batch.from_data_list(target_samples)).to(device)
    pred_theta, pred_tau_m = model.sample(batch)
    pred_theta = pred_theta.detach().cpu()
    pred_tau_m = pred_tau_m.detach().cpu().reshape(-1)

    theta_acc = torch.stack([data.torsions.clone() for _, _, _, _, data in window_jobs], dim=0)
    tau_acc = torch.stack([data.tau_m.clone() for _, _, _, _, data in window_jobs], dim=0)
    for sample_idx, (job_idx, target_idx) in enumerate(sample_keys):
        theta_acc[job_idx, target_idx] = pred_theta[sample_idx]
        tau_acc[job_idx, target_idx] = pred_tau_m[sample_idx]

    theta_w = theta_acc.float().to(device)
    tau_w = tau_acc.float().to(device).clamp(min=TAU_M_MIN, max=TAU_M_MAX)
    mask = torch.stack(
        [_inference_chain_end_mask_tensor(data) for _, _, _, _, data in window_jobs],
        dim=0,
    ).to(device)
    ri = torch.stack(
        [data.base_types.argmax(dim=-1).long() for _, _, _, _, data in window_jobs],
        dim=0,
    ).to(device)
    origins_w = torch.stack(
        [data.nt_origins_world.float() for _, _, _, _, data in window_jobs],
        dim=0,
    ).to(device)
    frames_w = torch.stack(
        [data.nt_frames_world.float() for _, _, _, _, data in window_jobs],
        dim=0,
    ).to(device)

    bb_t = build_batch_window_backbone_from_torsions(
        theta_w, tau_w, ri, origins_w, frames_w, mask,
    )
    coords_all = bb_t.detach().cpu().numpy()

    cached: dict[tuple[int, str, int], dict[Any, Any]] = {}
    for job_idx, (structure_idx, chain_key, window_idx, window, _data) in enumerate(window_jobs):
        cached[(structure_idx, chain_key, window_idx)] = _window_predictions_from_coords(
            window,
            coords_all[job_idx],
        )
    return cached


@torch.no_grad()
def _predict_full_window_predictions_dict(model, sample_data, device) -> dict[Any, Any]:
    """Backwards-compatible single-window wrapper over the batched inference path."""
    window = getattr(sample_data, '_window_ref', None)
    if window is None:
        return {}
    # Keep the legacy helper name while routing through build_batch_window_backbone_from_torsions.
    cached = _predict_full_window_predictions_dicts(
        model,
        [(0, '', 0, window, sample_data)],
        device,
    )
    return cached[(0, '', 0)]


def _inference_chain_end_mask_tensor(sample_data):
    """Boolean mask over window torsions: chain terminals disable α (5′) / ε ζ (3′).

    Used with window-level inference so phosphate bridges consume predicted ε ζ / α β on
    both sides rather than pretending base-only placeholders are trustworthy.
    """
    pos_mask = torch.ones(WINDOW_SIZE, N_TORSIONS, dtype=torch.bool)
    for i in range(WINDOW_SIZE):
        ce = sample_data.chain_end_class[i]
        if ce[CHAIN_END_CLASS_5_PRIME].item():
            pos_mask[i, TOR_ALPHA] = False
        if ce[CHAIN_END_CLASS_3_PRIME].item():
            pos_mask[i, TOR_EPS] = False
            pos_mask[i, TOR_ZETA] = False
    return pos_mask


def _chain_indices_5prime_to_3prime(chain):
    idxs = np.arange(len(chain), dtype=np.int64)
    if _chain_list_direction(chain) >= 0:
        return idxs.tolist()
    return idxs[::-1].tolist()


def _window_tidx_for_chain_index(chain_len: int, j: int):
    """Return (window_start_index, target_index_in_window)."""
    c = WINDOW_SIZE // 2
    w_cent = j - c
    if 0 <= w_cent <= chain_len - WINDOW_SIZE:
        return int(w_cent), int(c)
    if j <= c:
        return 0, int(j)
    if j >= chain_len - c - 1 + (WINDOW_SIZE % 2 == 1):
        return chain_len - WINDOW_SIZE, int(j - (chain_len - WINDOW_SIZE))
    return int(w_cent), int(c)


def _batched_window_predictions(model, chain_records_by_structure, device, show_progress, window_batch_size):
    jobs = []
    for structure_idx, chain_records in enumerate(chain_records_by_structure):
        for chain_key, _chain, windows in chain_records:
            for window, window_idx, data in windows:
                jobs.append((structure_idx, chain_key, int(window_idx), window, data.clone()))

    if not jobs:
        return {}

    if window_batch_size is None or window_batch_size <= 0:
        window_batch_size = len(jobs)

    cached: dict[tuple[int, str, int], dict[Any, Any]] = {}
    chunk_starts = range(0, len(jobs), window_batch_size)
    for start in tqdm(
        chunk_starts,
        total=(len(jobs) + window_batch_size - 1) // window_batch_size,
        desc='Backbone inference',
        leave=False,
        disable=not show_progress,
        colour=PROGRESS_BAR_COLOR,
    ):
        chunk = jobs[start:start + window_batch_size]
        cached.update(_predict_full_window_predictions_dicts(model, chunk, device))
    return cached


def _assemble_predictions(chain_records_by_structure, cached_window_predictions):
    predictions_by_structure: list[dict[Any, Any]] = []
    for structure_idx, chain_records in enumerate(chain_records_by_structure):
        predictions: dict[Any, Any] = {}
        for chain_key, chain, windows in chain_records:
            if len(chain) < WINDOW_SIZE:
                continue
            available_window_indices = {int(widx) for _window, widx, _data in windows}
            chain_len = len(chain)
            for j in _chain_indices_5prime_to_3prime(chain):
                window_idx, _target_idx = _window_tidx_for_chain_index(chain_len, j)
                if window_idx not in available_window_indices:
                    continue
                window_pred = cached_window_predictions.get((structure_idx, chain_key, window_idx))
                if window_pred is None:
                    continue
                _merge_window_pred_for_residue(predictions, window_pred, chain[j])
        predictions_by_structure.append(predictions)
    return predictions_by_structure


def _predict_backbone_from_chain_records(
    chain_records_by_structure,
    model,
    device,
    show_progress: bool = False,
    window_batch_size: int | None = None,
):
    cached_window_predictions = _batched_window_predictions(
        model,
        chain_records_by_structure,
        device,
        show_progress,
        window_batch_size,
    )
    return _assemble_predictions(chain_records_by_structure, cached_window_predictions)


def _build_output_trajectory(universes):
    if not universes:
        raise RuntimeError('No structures to write; trajectory prediction produced no frames.')

    first = universes[0]
    assert first.atoms is not None
    ref_names = first.atoms.names.tolist()
    ref_resids = first.atoms.resids.tolist()
    ref_chainids = first.atoms.chainIDs.tolist()
    positions = [first.atoms.positions.copy()]

    for universe in universes[1:]:
        assert universe.atoms is not None
        if (
            universe.atoms.n_atoms != first.atoms.n_atoms
            or universe.atoms.names.tolist() != ref_names
            or universe.atoms.resids.tolist() != ref_resids
            or universe.atoms.chainIDs.tolist() != ref_chainids
        ):
            raise RuntimeError('Trajectory frames do not share a stable atom topology.')
        positions.append(universe.atoms.positions.copy())

    trajectory = mda.Merge(first.atoms)
    trajectory.load_new(np.stack(positions, axis=0).astype(np.float32), format=MemoryReader)
    return trajectory


def _predict_backbone_outputs(
    input_path,
    model_path=MODEL_DIR,
    device='cuda',
    show_progress: bool = False,
) -> Tuple[dict, Any]:
    _, chain_records = parse_dna(
        input_path,
        use_full_nucleotide=False,
        window_size=WINDOW_SIZE,
    )
    model = _load_model(model_path, device)
    predictions = _predict_backbone_from_chain_records(
        [chain_records],
        model,
        device,
        show_progress=show_progress,
    )[0]
    return predictions, chain_records


def predict_backbone(
    input_path,
    model_path=MODEL_DIR,
    device='cuda',
    show_progress: bool = False,
    generate_5prime_phosphate: bool = False,
):
    predictions, chain_records = _predict_backbone_outputs(
        input_path,
        model_path=model_path,
        device=device,
        show_progress=show_progress,
    )
    return _build_output_universe(
        chain_records,
        predictions,
        generate_5prime_phosphate=generate_5prime_phosphate,
    )


def _predict_backbone_trajectory_outputs(
    universe,
    model_path=MODEL_DIR,
    device='cuda',
    show_progress: bool = False,
    window_batch_size: int | None = None,
) -> Tuple[list[dict[Any, Any]], list[Any]]:
    """Predict backbone atoms for every frame of an MDAnalysis trajectory."""
    chain_records_by_structure = []
    for _frame in tqdm(
        universe.trajectory,
        desc='Trajectory parsing',
        leave=False,
        disable=not show_progress,
        colour=PROGRESS_BAR_COLOR,
    ):
        _, chain_records = parse_dna_universe(
            universe,
            use_full_nucleotide=False,
            window_size=WINDOW_SIZE,
        )
        chain_records_by_structure.append(chain_records)

    model = _load_model(model_path, device)
    predictions_by_structure = _predict_backbone_from_chain_records(
        chain_records_by_structure,
        model,
        device,
        show_progress=show_progress,
        window_batch_size=window_batch_size,
    )
    return predictions_by_structure, chain_records_by_structure


def predict_backbone_trajectory(
    universe,
    model_path=MODEL_DIR,
    device='cuda',
    show_progress: bool = False,
    window_batch_size: int | None = None,
    generate_5prime_phosphate: bool = False,
):
    predictions_by_structure, chain_records_by_structure = _predict_backbone_trajectory_outputs(
        universe,
        model_path=model_path,
        device=device,
        show_progress=show_progress,
        window_batch_size=window_batch_size,
    )
    frame_universes = [
        _build_output_universe(
            chain_records,
            predictions,
            generate_5prime_phosphate=generate_5prime_phosphate,
        )
        for predictions, chain_records in zip(predictions_by_structure, chain_records_by_structure)
    ]
    return _build_output_trajectory(frame_universes)


def _element_of(atom_name):
    first = atom_name.lstrip('0123456789')[:1]
    return first.upper() if first else ' '


def _five_prime_keys(chain_records):
    keys = set()
    for _, chain, _ in chain_records:
        if not chain:
            continue
        if len(chain) == 1:
            nt = chain[0]
        else:
            first_resid = chain[0].e_residue.resids[0]
            last_resid = chain[-1].e_residue.resids[0]
            nt = chain[0] if last_resid >= first_resid else chain[-1]
        keys.add((nt.segid, int(nt.resid)))
    return keys


def _build_output_universe(chain_records, predictions, generate_5prime_phosphate=False):
    atom_names = []
    atom_elements = []
    atom_positions = []
    atom_chainids = []
    atom_resindex = []
    residue_resnames = []
    residue_resids = []
    residue_segindex = []
    segment_segids = []
    seg_idx_of_key = {}

    suppress_phosphate_keys = (
        set() if generate_5prime_phosphate else _five_prime_keys(chain_records)
    )

    for chain_key, chain, _ in chain_records:
        if chain_key not in seg_idx_of_key:
            seg_idx_of_key[chain_key] = len(seg_idx_of_key)
            segment_segids.append(chain_key)
        seg_idx = seg_idx_of_key[chain_key]

        for nucleotide in chain:
            segid = nucleotide.segid
            resid = int(nucleotide.resid)
            exp_positions = dict(default_atoms_provider(nucleotide))
            residue_had_atoms = False
            for atom_name, _ in inference_atoms_provider(nucleotide):
                if atom_name in BACKBONE_ATOMS:
                    if (
                        atom_name in FIVE_PRIME_PHOSPHATE_ATOMS
                        and (segid, resid) in suppress_phosphate_keys
                    ):
                        continue
                    xyz = predictions.get((segid, resid, atom_name))
                else:
                    xyz = exp_positions.get(atom_name)
                if xyz is None:
                    continue
                atom_names.append(atom_name)
                atom_elements.append(_element_of(atom_name))
                atom_positions.append(np.asarray(xyz, dtype=np.float32))
                atom_chainids.append(chain_key)
                atom_resindex.append(len(residue_resnames))
                residue_had_atoms = True
            if residue_had_atoms:
                residue_resnames.append(nucleotide.restype)
                residue_resids.append(resid)
                residue_segindex.append(seg_idx)

    n_atoms = len(atom_names)
    n_residues = len(residue_resnames)
    n_segments = len(segment_segids)
    if n_atoms == 0:
        raise RuntimeError('No atoms to write; predictions dict is empty.')

    u = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_residues,
        n_segments=n_segments,
        atom_resindex=atom_resindex,
        residue_segindex=residue_segindex,
        trajectory=True,
    )
    u.add_TopologyAttr('names', atom_names)
    u.add_TopologyAttr('elements', atom_elements)
    u.add_TopologyAttr('chainIDs', atom_chainids)
    u.add_TopologyAttr('resnames', residue_resnames)
    u.add_TopologyAttr('resids', residue_resids)
    u.add_TopologyAttr('segids', segment_segids)
    assert u.atoms is not None
    u.atoms.positions = np.stack(atom_positions).astype(np.float32)
    return u


def _normalize_output_path(output_path, output_format: str | None = None):
    output_path = str(output_path)
    ext = output_format if output_format is not None else osp.splitext(output_path)[1].lower()
    if ext == '.mmcif':
        ext = '.cif'
    if ext not in ('.pdb', '.cif'):
        raise ValueError(f'Unsupported output format: {ext!r} (expected .pdb/.cif)')
    root, current_ext = osp.splitext(output_path)
    if output_format is not None and current_ext.lower() != ext:
        output_path = root + ext if current_ext else output_path + ext
    return output_path, ext


def write_structure(universe, output_path, output_format: str | None = None):
    output_path, ext = _normalize_output_path(output_path, output_format)
    assert universe.atoms is not None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        if ext == '.pdb':
            universe.atoms.write(output_path, frames='all')
            return output_path
        fd, tmp_pdb = tempfile.mkstemp(suffix='.pdb')
        os.close(fd)
        try:
            universe.atoms.write(tmp_pdb, frames='all')
            parser = PDBParser(QUIET=True)
            bio_structure = parser.get_structure('regen', tmp_pdb)
            io = MMCIFIO()
            io.set_structure(bio_structure)
            io.save(output_path)
        finally:
            if osp.exists(tmp_pdb):
                os.remove(tmp_pdb)
    return output_path


def _parse_args():
    p = ArgumentParser(description='Regenerate DNA backbone atoms from a base-only PDB/mmCIF.')
    p.add_argument('--input', required=True, help='Path to input topology .pdb/.cif/.mmcif (DNA without backbone).')
    p.add_argument('--trajectory', help='Optional trajectory path (for example .xtc/.dcd); writes a multi-model output.')
    p.add_argument('--output', required=True, help='Output path; format from extension or --output-format.')
    p.add_argument(
        '--output-format',
        choices=['.pdb', '.cif'],
        help='Override output format. .cif writes mmCIF.',
    )
    p.add_argument(
        '--generate-5-prime-phosphate',
        action='store_true',
        help="Include 5'-terminal P, OP1, OP2 atoms in the output (omitted by default).",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.trajectory:
        input_universe = mda.Universe(args.input, args.trajectory)
        output_universe = predict_backbone_trajectory(
            input_universe,
            device=device,
            generate_5prime_phosphate=args.generate_5_prime_phosphate,
        )
    else:
        output_universe = predict_backbone(
            args.input,
            device=device,
            generate_5prime_phosphate=args.generate_5_prime_phosphate,
        )
    output_path = write_structure(
        output_universe,
        args.output,
        output_format=args.output_format,
    )
    assert output_universe.atoms is not None
    n_frames = len(output_universe.trajectory)
    print(f'Wrote {output_path} ({output_universe.atoms.n_atoms} atoms, {n_frames} frame(s)).')


if __name__ == '__main__':
    main()
