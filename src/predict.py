import os
import os.path as osp
import tempfile
import warnings
from argparse import ArgumentParser
from typing import Any, Optional, Tuple

import MDAnalysis as mda
import numpy as np
import torch
from tqdm import tqdm
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser
from torch_geometric.data import Batch

import utils
from model import PytorchLightningModule
from torsion_geometry import (N_TORSIONS, TOR_ALPHA, TOR_EPS, TOR_ZETA,
                              build_backbone_from_torsions)
from utils import (CHAIN_END_CLASS_3_PRIME, CHAIN_END_CLASS_5_PRIME,
                   resolve_run_dir)

WINDOW_SIZE = 3
_FIVE_PRIME_PHOSPHATE_ATOMS = frozenset({'P', 'OP1', 'OP2'})


def _load_model(ckpt_path, device):
    return (
        PytorchLightningModule
        .load_from_checkpoint(ckpt_path, map_location=device)
        .float()
        .to(device)
        .eval()
    )


def _chain_list_direction(chain):
    """Same convention as utils.parse_dna: +1 if resid increases toward 3' in list order."""
    first_resid = chain[0].e_residue.resids[0]
    last_resid = chain[-1].e_residue.resids[0]
    return 1 if last_resid >= first_resid else -1


@torch.no_grad()
def _predict_window(
    model,
    data,
    device,
    tidx: int,
    o3_prev_local: Optional[np.ndarray] = None,
) -> Tuple[dict, np.ndarray]:
    """Sample from the model; build local backbone → world coords for target residue `tidx`."""
    window = getattr(data, '_window_ref', None)
    if window is None:
        return {}, np.zeros(N_TORSIONS, dtype=np.float64)
    data.target_nt_idx = torch.tensor(tidx, dtype=torch.long)
    is_target = torch.zeros(WINDOW_SIZE, dtype=torch.float)
    is_target[tidx] = 1.0
    data.is_target_nt = is_target
    o_t = data.nt_origins_world[tidx]
    r_t = data.nt_frames_world[tidx]
    data.rel_origins = ((data.nt_origins_world - o_t) @ r_t).float()
    data.rel_frames = torch.einsum('ji,njk->nik', r_t, data.nt_frames_world).float()

    # Partner nucleotide positions/frames in the target nucleotide's local frame
    data.pair_rel_origins = ((data.pair_origins_world - o_t) @ r_t).float()
    data.pair_rel_frames = torch.einsum('ji,njk->nik', r_t, data.pair_frames_world).float()

    batch = Batch.from_data_list([data]).to(device)  # type: ignore[union-attr]
    pred_theta, pred_tau_m = model.sample(batch)

    nt = window[tidx]
    o_tt = data.nt_origins_world[tidx]
    r_tt = data.nt_frames_world[tidx]

    torsions_np = pred_theta[0].cpu().numpy()
    tau_m = float(pred_tau_m[0].clamp(min=1e-3, max=7.4).item())
    o3pl = (
        np.asarray(o3_prev_local, dtype=np.float64).reshape(3)
        if o3_prev_local is not None
        else None
    )

    local_bb = build_backbone_from_torsions(
        torsions_np,
        nt.restype,
        o3_prev_local=o3pl,
        tau_m=tau_m,
    )
    o = o_tt.cpu().numpy().reshape(3)
    r = r_tt.cpu().numpy().reshape(3, 3)
    world_bb = {nm: xyz @ r.T + o for nm, xyz in local_bb.items()}
    segid, resid = nt.segid, int(nt.resid)
    preds = {(segid, resid, nm): pos for nm, pos in world_bb.items()}
    return preds, torsions_np


def _apply_inference_pos_mask(sample_data):
    """Replace data-derived masks with positional chain-end masking (distribution shift fix)."""
    pos_mask = torch.ones(WINDOW_SIZE, N_TORSIONS, dtype=torch.bool)
    for i in range(WINDOW_SIZE):
        ce = sample_data.chain_end_class[i]
        if ce[CHAIN_END_CLASS_5_PRIME].item():
            pos_mask[i, TOR_ALPHA] = False
        if ce[CHAIN_END_CLASS_3_PRIME].item():
            pos_mask[i, TOR_EPS] = False
            pos_mask[i, TOR_ZETA] = False
    sample_data.torsion_mask = pos_mask


def _run_target(
    model,
    sample_data,
    device,
    predictions,
    window,
    tidx: int,
    o3_prev_local: Optional[np.ndarray],
):
    sample_data._window_ref = window
    _apply_inference_pos_mask(sample_data)
    bb, tort = _predict_window(
        model,
        sample_data,
        device,
        tidx,
        o3_prev_local=o3_prev_local,
    )
    predictions.update(bb)


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


def predict_backbone(input_path, ckpt_path, device='cuda', show_progress: bool = False):
    _, chain_records = utils.parse_dna(
        input_path,
        use_full_nucleotide=False,
        window_size=WINDOW_SIZE,
    )
    model = _load_model(ckpt_path, device)
    predictions: dict = {}

    for _chain_key, chain, windows in chain_records:
        if len(chain) < WINDOW_SIZE:
            continue
        w_by_start = {widx: (w, d.clone()) for w, widx, d in windows}
        L = len(chain)
        o3_prev_world: Optional[np.ndarray] = None
        ordered_j = _chain_indices_5prime_to_3prime(chain)

        for j in tqdm(
                ordered_j,
                desc='Backbone inference',
                leave=False,
                disable=not show_progress,
                colour=utils.PBAR_COLOR,
        ):
            widx, tidx = _window_tidx_for_chain_index(L, j)
            if widx not in w_by_start:
                continue
            window, data = w_by_start[widx]
            nt = chain[j]
            o_t = data.nt_origins_world[tidx]
            R_t = data.nt_frames_world[tidx]
            onp = o_t.numpy().reshape(3)
            Rnp = R_t.numpy().reshape(3, 3)

            o3_prev_local: Optional[np.ndarray] = None
            if o3_prev_world is not None:
                o3_prev_local = (o3_prev_world - onp) @ Rnp

            dc = data.clone()
            dc._window_ref = window
            _run_target(
                model,
                dc,
                device,
                predictions,
                window,
                tidx,
                o3_prev_local,
            )

            segid_, resid_ = nt.segid, int(nt.resid)
            nk = (segid_, resid_, "O3'")
            if nk in predictions:
                o3_prev_world = np.asarray(predictions[nk], dtype=np.float64)

    return predictions, chain_records


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
            exp_positions = dict(utils.default_atoms_provider(nucleotide))
            residue_had_atoms = False
            for atom_name, _ in utils.inference_atoms_provider(nucleotide):
                if atom_name in utils.backbone_atoms:
                    if (
                        atom_name in _FIVE_PRIME_PHOSPHATE_ATOMS
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


def write_structure(chain_records, predictions, output_path, generate_5prime_phosphate=False):
    ext = osp.splitext(output_path)[1].lower()
    if ext not in ('.pdb', '.cif', '.mmcif'):
        raise ValueError(f'Unsupported output format: {ext!r} (expected .pdb/.cif/.mmcif)')
    universe = _build_output_universe(chain_records, predictions, generate_5prime_phosphate)
    assert universe.atoms is not None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        if ext == '.pdb':
            universe.atoms.write(output_path)
            return
        fd, tmp_pdb = tempfile.mkstemp(suffix='.pdb')
        os.close(fd)
        try:
            universe.atoms.write(tmp_pdb)
            parser = PDBParser(QUIET=True)
            bio_structure = parser.get_structure('regen', tmp_pdb)
            io = MMCIFIO()
            io.set_structure(bio_structure)
            io.save(output_path)
        finally:
            if osp.exists(tmp_pdb):
                os.remove(tmp_pdb)


def _parse_args():
    p = ArgumentParser(description='Regenerate DNA backbone atoms from a base-only PDB/mmCIF.')
    p.add_argument('--input', required=True, help='Path to input .pdb/.cif/.mmcif (DNA without backbone).')
    p.add_argument('--output', required=True, help='Output path; format from extension (.pdb or .cif/.mmcif).')
    p.add_argument('--run-dir', required=True, help='Experiment id under logs/ (e.g. "fixed_swa/baseline").')
    p.add_argument(
        '--generate-5-prime-phosphate',
        action='store_true',
        help="Include 5'-terminal P, OP1, OP2 atoms in the output (omitted by default).",
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    ckpt_path = utils.find_best_checkpoint(resolve_run_dir(args.run_dir))
    print(f'checkpoint: {ckpt_path}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions, chain_records = predict_backbone(args.input, ckpt_path, device=device)
    write_structure(
        chain_records,
        predictions,
        args.output,
        generate_5prime_phosphate=args.generate_5_prime_phosphate,
    )
    print(f'Wrote {args.output} ({len(predictions)} predicted backbone atoms).')
