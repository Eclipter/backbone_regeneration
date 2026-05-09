import os
import os.path as osp
import tempfile
import warnings
from argparse import ArgumentParser
from typing import Any, Dict, Optional, Tuple

import MDAnalysis as mda
import numpy as np
import torch
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.PDBParser import PDBParser
from torch_geometric.data import Batch

import utils
from model import PytorchLightningModule
from torsion_geometry import (N_TORSIONS, TOR_ALPHA, TOR_EPS, TOR_ZETA,
                              _get_template, build_backbone_from_torsions,
                              nerf_place)
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


def _blen_tpl(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _bangle_tpl(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def _base_atoms_local(nt, o_t: torch.Tensor, r_t: torch.Tensor) -> Dict[str, np.ndarray]:
    exp = dict(utils.default_atoms_provider(nt))
    o = o_t.cpu().numpy().reshape(3)
    r = r_t.cpu().numpy().reshape(3, 3)
    out: Dict[str, np.ndarray] = {}
    for nm in ('N9', 'N1', 'C4', 'C2'):
        if nm not in exp:
            continue
        w = np.asarray(exp[nm], dtype=np.float64).reshape(3)
        out[nm] = (w - o) @ r
    return out


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

    batch = Batch.from_data_list([data]).to(device)  # type: ignore[union-attr]
    pred_theta, pred_tau_m = model.sample(batch)

    nt = window[tidx]
    o_tt = data.nt_origins_world[tidx]
    r_tt = data.nt_frames_world[tidx]

    torsions_np = pred_theta[0].cpu().numpy()
    tau_m = float(pred_tau_m[0].clamp(min=1e-3, max=7.4).item())
    base_loc = _base_atoms_local(nt, o_tt, r_tt)
    o3pl = (
        np.asarray(o3_prev_local, dtype=np.float64).reshape(3)
        if o3_prev_local is not None
        else None
    )

    local_bb = build_backbone_from_torsions(
        torsions_np,
        nt.restype,
        o3_prev_local=o3pl,
        base_atoms_local=base_loc if base_loc else None,
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
    torsions_cache: Dict[Tuple[Any, int], np.ndarray],
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
    nt = window[tidx]
    torsions_cache[(nt.segid, int(nt.resid))] = tort


def _refine_eps_zeta(
    chain_ordered_5prime: list,
    predictions: dict,
    torsions_cache: Dict[Tuple[Any, int], np.ndarray],
):
    HP = np.pi / 2.0
    L = len(chain_ordered_5prime)
    for i in range(L - 1):
        nt_curr = chain_ordered_5prime[i]
        nt_next = chain_ordered_5prime[i + 1]
        sid_c, rid_c = nt_curr.segid, int(nt_curr.resid)
        sid_n, rid_n = nt_next.segid, int(nt_next.resid)

        kv_c = (sid_c, rid_c)
        tort_c = torsions_cache.get(kv_c)
        if tort_c is None:
            continue

        c4 = predictions.get((sid_c, rid_c, "C4'"))
        c3 = predictions.get((sid_c, rid_c, "C3'"))
        o3_orig = predictions.get((sid_c, rid_c, "O3'"))
        p_next = predictions.get((sid_n, rid_n, 'P'))
        if c4 is None or c3 is None or o3_orig is None or p_next is None:
            continue

        tpl_c = _get_template(nt_curr.restype)
        tpl_n = _get_template(nt_next.restype)
        eps = float(tort_c[TOR_EPS])
        r_o3 = _blen_tpl(tpl_c["O3'"], tpl_c["C3'"])
        th_o3 = _bangle_tpl(tpl_c["C4'"], tpl_c["C3'"], tpl_c["O3'"])
        c4f = np.asarray(c4, dtype=np.float64).reshape(3)
        c3f = np.asarray(c3, dtype=np.float64).reshape(3)
        pnf = np.asarray(p_next, dtype=np.float64).reshape(3)
        o3_eps = nerf_place(c4f, c3f, pnf, r_o3, th_o3, eps - HP)

        oo = np.asarray(o3_orig, dtype=np.float64).reshape(3)
        predictions[(sid_c, rid_c, "O3'")] = 0.5 * (oo + o3_eps)

        zeta = float(tort_c[TOR_ZETA])
        o5_k = (sid_n, rid_n, "O5'")
        o5_orig = predictions.get(o5_k)
        if o5_orig is None:
            continue
        r_o5p = _blen_tpl(tpl_n["O5'"], tpl_n['P'])
        th_z = _bangle_tpl(tpl_n["O3'"], tpl_n['P'], tpl_n["O5'"])
        c3_use = predictions.get((sid_c, rid_c, "C3'"))
        o3_use = predictions[(sid_c, rid_c, "O3'")]
        c3u = np.asarray(c3_use, dtype=np.float64).reshape(3)
        o3u = np.asarray(o3_use, dtype=np.float64).reshape(3)
        pn2 = predictions[(sid_n, rid_n, 'P')]
        pnf2 = np.asarray(pn2, dtype=np.float64).reshape(3)
        o5_z = nerf_place(c3u, o3u, pnf2, r_o5p, th_z, zeta - HP)
        oof = np.asarray(o5_orig, dtype=np.float64).reshape(3)
        predictions[o5_k] = 0.5 * (oof + o5_z)


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


def predict_backbone(input_path, ckpt_path, device='cuda'):
    _, chain_records = utils.parse_dna(
        input_path,
        use_full_nucleotide=False,
        window_size=WINDOW_SIZE,
    )
    model = _load_model(ckpt_path, device)
    predictions: dict = {}
    torsions_cache: Dict[Tuple[Any, int], np.ndarray] = {}

    for _chain_key, chain, windows in chain_records:
        if len(chain) < WINDOW_SIZE:
            continue
        w_by_start = {widx: (w, d.clone()) for w, widx, d in windows}
        L = len(chain)
        o3_prev_world: Optional[np.ndarray] = None
        ordered_j = _chain_indices_5prime_to_3prime(chain)

        for j in ordered_j:
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
                torsions_cache,
                window,
                tidx,
                o3_prev_local,
            )

            segid_, resid_ = nt.segid, int(nt.resid)
            nk = (segid_, resid_, "O3'")
            if nk in predictions:
                o3_prev_world = np.asarray(predictions[nk], dtype=np.float64)

        ordered_53 = [chain[j] for j in ordered_j]
        _refine_eps_zeta(ordered_53, predictions, torsions_cache)

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
