"""Full-structure backbone RMSD vs a reference coordinate file."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..data import BACKBONE_ATOMS, parse_dna
from ..io import default_atoms_provider
from .local_geometry import local_backbone_rmsd, world_to_local_np


def residue_backbone_positions(chain_records):
    residue_positions = {}
    for chain_key, chain, _ in chain_records:
        for nucleotide in chain:
            residue_key = (chain_key, int(nucleotide.resid))
            residue_positions[residue_key] = {
                atom_name: np.asarray(atom_pos, dtype=np.float64)
                for atom_name, atom_pos in default_atoms_provider(nucleotide)
                if atom_name in BACKBONE_ATOMS
            }
    return residue_positions


def backbone_local_array(atom_positions, origin_world, frame_world):
    local_coords = np.full((len(BACKBONE_ATOMS), 3), np.nan, dtype=np.float64)
    for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
        atom_pos = atom_positions.get(atom_name)
        if atom_pos is None or not np.isfinite(atom_pos).all():
            continue
        local_coords[atom_idx] = world_to_local_np(atom_pos, origin_world, frame_world).reshape(3)
    return local_coords


@lru_cache(maxsize=None)
def parse_chain_records_for_backbone_rmsd(path: str, window_size: int = 3):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'1 A\^3 CRYST1 record, this is usually a placeholder\..*',
            category=UserWarning,
            module=r'MDAnalysis\.coordinates\.PDB',
        )
        _, chain_records = parse_dna(
            path,
            use_full_nucleotide=True,
            window_size=window_size,
        )
    return chain_records


def collect_backbone_local_arrays(ref_chain_records, *position_maps):
    local_arrays = []
    seen_residues = set()
    for chain_key, _, windows in ref_chain_records:
        for window, _, data in windows:
            for nt_idx, nucleotide in enumerate(window):
                residue_key = (chain_key, int(nucleotide.resid))
                if residue_key in seen_residues:
                    continue
                seen_residues.add(residue_key)

                atoms_per_map = [positions.get(residue_key) for positions in position_maps]
                if any(atoms is None for atoms in atoms_per_map):
                    continue

                origin_world = data.nt_origins_world[nt_idx].numpy()
                frame_world = data.nt_frames_world[nt_idx].numpy()
                if np.isnan(origin_world).any() or np.isnan(frame_world).any():
                    continue

                local_arrays.append((
                    tuple(
                        backbone_local_array(atoms, origin_world, frame_world)
                        for atoms in atoms_per_map
                    ),
                    bool(data.is_chain_edge_nt[nt_idx].item()),
                ))
    return local_arrays


def compute_structure_vs_ref_backbone_rmsd(
    reference_path: str | Path,
    output_path: str | Path,
    window_size: int = 3,
) -> dict[str, Any]:
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    ref_chain_records = parse_chain_records_for_backbone_rmsd(
        str(reference_path.resolve()),
        window_size=window_size,
    )
    output_chain_records = parse_chain_records_for_backbone_rmsd(
        str(output_path.resolve()),
        window_size=window_size,
    )

    ref_positions = residue_backbone_positions(ref_chain_records)
    output_positions = residue_backbone_positions(output_chain_records)

    local_arrays = collect_backbone_local_arrays(
        ref_chain_records,
        ref_positions,
        output_positions,
    )
    rmsds = []
    rmsd_is_edge = []
    for (ref_local, output_local), is_edge in local_arrays:
        rmsd = local_backbone_rmsd(output_local, ref_local)
        if np.isfinite(rmsd):
            rmsds.append(float(rmsd))
            rmsd_is_edge.append(is_edge)

    if not rmsds:
        return {
            'success': False,
            'n_residues': 0,
            'mean_rmsd': None,
            'median_rmsd': None,
            'p90_rmsd': None,
        }

    rmsd_arr = np.asarray(rmsds, dtype=np.float64)
    return {
        'success': True,
        'n_residues': int(len(rmsd_arr)),
        'mean_rmsd': float(np.mean(rmsd_arr)),
        'median_rmsd': float(np.median(rmsd_arr)),
        'p90_rmsd': float(np.percentile(rmsd_arr, 90)),
    }


def median_backbone_rmsd_vs_reference(
    reference_path: str | Path,
    output_path: str | Path,
    window_size: int = 3,
) -> float | None:
    """Median per-residue backbone RMSD; lower is better for best-of-k selection."""
    stats = compute_structure_vs_ref_backbone_rmsd(
        reference_path,
        output_path,
        window_size=window_size,
    )
    if not stats['success']:
        return None
    return stats['median_rmsd']
