"""Reusable analysis helpers shared across scripts and baselines."""

import os.path as osp
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from pynamod.atomic_analysis.nucleotides_parser import nucleotide_graphs
from torch_geometric.data import Data

from ..data import BACKBONE_ATOMS, BASE_TO_INDEX, parse_dna
from ..dataset import PyGDataset
from ..io import default_atoms_provider, rename_atom


def world_to_local_np(
    points_world,
    target_origin,
    target_frame,
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    target_origin = np.asarray(target_origin, dtype=np.float64)
    target_frame = np.asarray(target_frame, dtype=np.float64)
    return (points_world - target_origin) @ target_frame


def local_to_world_np(
    points_local,
    origin_world,
    frame_world,
) -> np.ndarray:
    points_local = np.asarray(points_local, dtype=np.float64)
    origin_world = np.asarray(origin_world, dtype=np.float64)
    frame_world = np.asarray(frame_world, dtype=np.float64)
    return points_local @ frame_world.T + origin_world


def backbone_predictions_from_matched_local(
    ref_local: np.ndarray,
    origin_world: np.ndarray,
    frame_world: np.ndarray,
    segid: str,
    resid: int,
) -> dict[tuple[str, int, str], np.ndarray]:
    predictions: dict[tuple[str, int, str], np.ndarray] = {}
    for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
        local_xyz = ref_local[atom_idx]
        if not np.isfinite(local_xyz).all():
            continue
        world_xyz = local_to_world_np(local_xyz, origin_world, frame_world)
        predictions[(segid, resid, atom_name)] = world_xyz.astype(np.float32)
    return predictions


@lru_cache(maxsize=1)
def _backbone_bonds() -> tuple[tuple[str, str], ...]:
    graph = nucleotide_graphs['A']
    backbone_atoms = set(BACKBONE_ATOMS)
    bonds: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src_idx, dst_idx in graph.edges():
        src_name = rename_atom(graph.nodes[src_idx]['atom'].name)
        dst_name = rename_atom(graph.nodes[dst_idx]['atom'].name)
        if src_name not in backbone_atoms or dst_name not in backbone_atoms or src_name == dst_name:
            continue
        key = (src_name, dst_name) if src_name < dst_name else (dst_name, src_name)
        if key in seen:
            continue
        seen.add(key)
        bonds.append((src_name, dst_name))
    return tuple(bonds)


def ordered_backbone_segments(
    coords_by_name: dict[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (coords_by_name[name_a], coords_by_name[name_b])
        for name_a, name_b in _backbone_bonds()
        if name_a in coords_by_name and name_b in coords_by_name
    ]


def backbone_segments_from_local_coords(
    bb_local: np.ndarray,
    valid_mask: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    atom_names = [
        BACKBONE_ATOMS[idx]
        for idx in range(len(BACKBONE_ATOMS))
        if valid_mask[idx]
    ]
    coords_by_name = {
        atom_name: xyz
        for atom_name, xyz in zip(atom_names, bb_local)
    }
    return ordered_backbone_segments(coords_by_name)


def backbone_local_in_target_frame(
    sample_data: Data,
    nucleotide_idx: int,
    origin_world: np.ndarray,
    frame_world: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    bb_world = sample_data.bb_xyz_world[nucleotide_idx].numpy()
    valid = ~np.any(np.isnan(bb_world), axis=1)
    if not valid.any():
        return [], np.empty((0, 3), dtype=np.float64)
    local = world_to_local_np(bb_world[valid], origin_world, frame_world)
    row_valid = ~np.any(np.isnan(local), axis=1)
    names = [
        BACKBONE_ATOMS[j]
        for k, j in enumerate(range(len(BACKBONE_ATOMS)))
        if valid[j] and row_valid[k]
    ]
    return names, local[row_valid]


def find_window_matching_sample(
    dataset: PyGDataset,
    pdb_id: str,
    data: Data,
) -> Any | None:
    """Reload structure and locate the sliding window whose backbone tensor matches `data`."""
    raw_path = osp.join(dataset.raw_dir, f'{pdb_id}.cif')
    if not osp.exists(raw_path):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        _, chain_records = parse_dna(
            raw_path,
            use_full_nucleotide=True,
            window_size=dataset.window_size,
        )
    ref_bb = data.bb_xyz_world.detach().cpu()
    for _chain_key, _chain, windows in chain_records:
        for window, _widx, wdata in windows:
            cand = wdata.bb_xyz_world.detach().cpu()
            if not torch.equal(torch.isnan(cand), torch.isnan(ref_bb)):
                continue
            if torch.allclose(
                torch.nan_to_num(cand),
                torch.nan_to_num(ref_bb),
                rtol=5e-4,
                atol=5e-4,
            ):
                return window
    return None


def coords_local_per_nt(
    window,
    origin_world: np.ndarray,
    frame_world: np.ndarray,
) -> list[dict[str, np.ndarray]]:
    """Per-residue atom name → coordinates in the target nucleotide frame."""
    return [
        {
            name: world_to_local_np(pos, origin_world, frame_world).reshape(3)
            for name, pos in default_atoms_provider(nucleotide)
        }
        for nucleotide in window
    ]


def bond_segments_from_nt_graph(
    coords_by_name: dict[str, np.ndarray],
    restype_letter: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Intra-nucleotide bonds using the pynamod template graph."""
    if restype_letter not in nucleotide_graphs:
        return []
    graph = nucleotide_graphs[restype_letter]
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[str, str]] = set()
    for src_idx, dst_idx in graph.edges():
        src_name = rename_atom(graph.nodes[src_idx]['atom'].name)
        dst_name = rename_atom(graph.nodes[dst_idx]['atom'].name)
        if src_name not in coords_by_name or dst_name not in coords_by_name or src_name == dst_name:
            continue
        key = (src_name, dst_name) if src_name < dst_name else (dst_name, src_name)
        if key in seen:
            continue
        seen.add(key)
        segments.append((coords_by_name[src_name], coords_by_name[dst_name]))
    return segments


def phosphodiester_segments_local(
    data: Data,
    origin_world: np.ndarray,
    frame_world: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Inter-residue O3'(i) — P(i+1) in the target frame when backbone atoms exist."""
    window_size = data.bb_xyz_world.shape[0]
    jo3 = BACKBONE_ATOMS.index("O3'")
    jp = BACKBONE_ATOMS.index('P')
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for idx in range(window_size - 1):
        o3 = data.bb_xyz_world[idx, jo3].numpy()
        p_next = data.bb_xyz_world[idx + 1, jp].numpy()
        if np.isnan(o3).any() or np.isnan(p_next).any():
            continue
        segments.append((
            world_to_local_np(o3, origin_world, frame_world).reshape(3),
            world_to_local_np(p_next, origin_world, frame_world).reshape(3),
        ))
    return segments


def local_backbone_rmsd(
    pred_local: np.ndarray,
    gt_local: np.ndarray,
) -> float:
    """RMSD in the local frame with permutation-invariant OP1/OP2 matching."""
    sq: list[float] = []
    j_op1 = BACKBONE_ATOMS.index('OP1')
    j_op2 = BACKBONE_ATOMS.index('OP2')
    for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
        if atom_name in ('OP1', 'OP2'):
            continue
        pred_atom = pred_local[atom_idx]
        gt_atom = gt_local[atom_idx]
        if np.isnan(pred_atom).any() or np.isnan(gt_atom).any():
            continue
        sq.append(float(np.sum((pred_atom - gt_atom) ** 2)))
    pred_op1 = pred_local[j_op1]
    pred_op2 = pred_local[j_op2]
    gt_op1 = gt_local[j_op1]
    gt_op2 = gt_local[j_op2]
    if not (
        np.isnan(pred_op1).any()
        or np.isnan(pred_op2).any()
        or np.isnan(gt_op1).any()
        or np.isnan(gt_op2).any()
    ):
        direct = np.sum((pred_op1 - gt_op1) ** 2) + np.sum((pred_op2 - gt_op2) ** 2)
        swapped = np.sum((pred_op1 - gt_op2) ** 2) + np.sum((pred_op2 - gt_op1) ** 2)
        sq.append(float(min(direct, swapped)) / 2.0)
    return float(np.sqrt(np.mean(sq))) if sq else np.nan
