import os
import os.path as osp
import tempfile
import warnings
from argparse import ArgumentParser

import MDAnalysis as mda
import numpy as np
import torch
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser
from torch_geometric.data import Batch

import utils
from export_to_onnx import resolve_run_dir
from model import PytorchLightningModule

WINDOW_SIZE = 3


def _window_atom_meta(window):
    """Per-atom (segid, resid, atom_name) triples in the order used by the inference provider."""
    return [
        (nt.segid, int(nt.resid), name)
        for nt in window
        for name, _ in utils.inference_atoms_provider(nt)
    ]


def _load_model(ckpt_path, device):
    return (
        PytorchLightningModule
        .load_from_checkpoint(ckpt_path, map_location=device)
        .float()
        .to(device)
        .eval()
    )


@torch.no_grad()
def _predict_window(model, data, frame_atom_idx, device):
    """Reverse-diffuse a single window; returns world-frame positions of the atoms the
    model was asked to predict (selected via `data.is_target`), in the order they
    appear in the window. `frame_atom_idx` names the atom whose nucleotide frame
    `data.pos` is currently expressed in (matches the training-time frame).
    """
    batch = Batch.from_data_list([data]).to(device)  # type: ignore
    gen_pos_local = model.sample(batch)
    origin = data.origins[frame_atom_idx].to(device)
    ref_frame = data.ref_frames[frame_atom_idx].to(device)
    return (gen_pos_local @ ref_frame.T + origin).cpu().numpy()


def _run_target(model, data, target_mask_attr, device, meta, predictions):
    """Set `is_target` to the requested mask, run the diffusion sampler, and store
    the world-frame predicted coordinates into the shared `predictions` dict.
    """
    target = getattr(data, target_mask_attr) & data.backbone_mask
    if not target.any():
        return
    sample_data = data.clone()
    sample_data.is_target = getattr(sample_data, target_mask_attr).clone()
    # Mirror training-time frame choice: edge targets use the edge nucleotide's
    # frame, central targets keep the central nucleotide's frame.
    if target_mask_attr == 'is_chain_edge':
        frame_atom_idx = int(sample_data.is_chain_edge.nonzero(as_tuple=True)[0][0])
        utils.reframe_positions_to_atom(sample_data, frame_atom_idx)
    else:
        frame_atom_idx = int(sample_data.central_mask.nonzero(as_tuple=True)[0][0])
    pred = _predict_window(model, sample_data, frame_atom_idx, device)
    target_idx = target.nonzero(as_tuple=True)[0].tolist()
    for idx, xyz in zip(target_idx, pred):
        predictions[meta[idx]] = xyz


def predict_backbone(input_path, ckpt_path, device='cuda'):
    """Load input, run reverse diffusion per window, return predicted backbone atoms.

    Returns:
        predictions:   dict keyed by (segid, resid, atom_name) -> (x, y, z) ndarray
        chain_records: list of (chain_key, chain, windows) from `parse_dna`, reused
                       downstream for output writing.
    """
    # use_full_nucleotide=False: graph-match against the base heterocycle only,
    # so inputs without backbone atoms still pass nucleotide recognition.
    _, chain_records = utils.parse_dna(
        input_path,
        use_full_nucleotide=False,
        window_size=WINDOW_SIZE,
        atoms_provider=utils.inference_atoms_provider,
    )

    model = _load_model(ckpt_path, device)

    predictions = {}
    for _, _, windows in chain_records:
        for window, _, data in windows:
            meta = _window_atom_meta(window)
            # The unified model produces one target per forward pass. Always ask for
            # the central nucleotide; additionally, for chain-edge windows, ask for
            # the edge nucleotide so positions 0 and L-1 of each chain are covered.
            _run_target(model, data, 'central_mask', device, meta, predictions)
            if data.is_chain_edge.any():
                _run_target(model, data, 'is_chain_edge', device, meta, predictions)

    return predictions, chain_records


def _element_of(atom_name):
    """Infer element symbol from the canonical atom name (references contain only C/N/O/P)."""
    first = atom_name.lstrip("0123456789")[:1]
    return first.upper() if first else ' '


def _build_output_universe(chain_records, predictions):
    """Assemble a new MDAnalysis Universe with all DNA atoms in canonical reference order,
    using experimental positions for base atoms and predicted ones for backbone atoms.
    """
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
            # Iterate canonical heavy atoms in reference order; positions are ignored here
            # and sourced below from predictions (backbone) or exp_positions (other).
            for atom_name, _ in utils.inference_atoms_provider(nucleotide):
                if atom_name in utils.backbone_atoms:
                    # Skip backbone atoms we could not predict (e.g. chain too short).
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
    u.add_TopologyAttr('names', atom_names)  # type: ignore
    u.add_TopologyAttr('elements', atom_elements)  # type: ignore
    u.add_TopologyAttr('chainIDs', atom_chainids)  # type: ignore
    u.add_TopologyAttr('resnames', residue_resnames)  # type: ignore
    u.add_TopologyAttr('resids', residue_resids)  # type: ignore
    u.add_TopologyAttr('segids', segment_segids)  # type: ignore
    u.atoms.positions = np.stack(atom_positions).astype(np.float32)  # type: ignore
    return u


def write_structure(chain_records, predictions, output_path):
    """Write predicted DNA with backbone to PDB or mmCIF, chosen by file extension."""
    ext = osp.splitext(output_path)[1].lower()
    if ext not in ('.pdb', '.cif', '.mmcif'):
        raise ValueError(f'Unsupported output format: {ext!r} (expected .pdb/.cif/.mmcif)')

    universe = _build_output_universe(chain_records, predictions)

    # MDA emits harmless warnings about missing attrs (altLocs, icodes, ...) when writing
    # a synthesized universe; suppress them so CLI output stays readable.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        if ext == '.pdb':
            universe.atoms.write(output_path)  # type: ignore
            return

        # MDA 2.x has no native mmCIF writer; round-trip via a temp PDB and BioPython's MMCIFIO.
        fd, tmp_pdb = tempfile.mkstemp(suffix='.pdb')
        os.close(fd)
        try:
            universe.atoms.write(tmp_pdb)  # type: ignore
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
    p.add_argument(
        '--output',
        required=True,
        help='Path to output file; format is chosen from the extension (.pdb or .cif/.mmcif).',
    )
    p.add_argument(
        '--run-dir',
        required=True,
        help='Experiment id relative to logs/ (e.g. "fixed_swa/baseline").',
    )
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    ckpt_path = utils.find_best_checkpoint(resolve_run_dir(args.run_dir))
    print(f'checkpoint: {ckpt_path}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions, chain_records = predict_backbone(args.input, ckpt_path, device=device)
    write_structure(chain_records, predictions, args.output)
    print(f'Wrote {args.output} ({len(predictions)} predicted backbone atoms).')
