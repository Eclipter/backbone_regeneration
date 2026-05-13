"""Torsion extraction and pseudorotation helpers."""

import functools
import math

import numpy as np
import torch

from ..torsion_constants import (
    N_TORSIONS,
    TOR_ALPHA,
    TOR_BETA,
    TOR_CHI,
    TOR_EPS,
    TOR_GAMMA,
    TOR_PUCKER_P,
    TOR_ZETA,
)
from .primitives import dihedral_rad, dihedral_rad_torch

RING_TORSION_DEFS = (
    ("C1'", "C2'", "C3'", "C4'"),
    ("C2'", "C3'", "C4'", "O4'"),
    ("C3'", "C4'", "O4'", "C1'"),
    ("C4'", "O4'", "C1'", "C2'"),
    ("O4'", "C1'", "C2'", "C3'"),
)
_RING_ANGLES = 2.0 * 2.0 * math.pi * np.arange(5, dtype=np.float64) / 5.0
_PSEUDOROTATION_OFFSETS = (
    np.array([0.0, 4.0, 8.0, 2.0, 6.0], dtype=np.float64) * (math.pi / 5.0)
)


def pseudorotation_nus_numpy(P_rad: float, tau_m: float) -> np.ndarray:
    """Return ν₂…ν₁ cycle from pseudorotation phase and amplitude."""
    return float(tau_m) * np.cos(float(P_rad) + _PSEUDOROTATION_OFFSETS)


def pseudorotation_P_rad_from_nus(nu_deg):
    """Return MDAnalysis `phase_as` phase angle in radians from five endocyclic torsions."""
    nu = np.asarray(nu_deg, dtype=np.float64)
    b = np.dot(nu, np.sin(_RING_ANGLES)) * (-2.0 / 5.0)
    a = np.dot(nu, np.cos(_RING_ANGLES)) * (2.0 / 5.0)
    return float(np.arctan2(b, a))


def pucker_amplitude_rad(nu_deg, P_rad):
    """Return τ_m in radians with pseudorotation phase held fixed."""
    nu = np.deg2rad(np.asarray(nu_deg, dtype=np.float64))
    idx = np.arange(5, dtype=np.float64)
    phases = P_rad + (2.0 * math.pi * idx / 5.0)
    cos_phase = np.cos(phases)
    denom = float(np.dot(cos_phase, cos_phase)) + 1e-12
    tau = float(np.dot(nu, cos_phase) / np.sqrt(denom))
    return abs(tau)


def _chi_quads(base_one_letter):
    if base_one_letter in ('A', 'G'):
        return ("O4'", "C1'", 'N9', 'C4')
    return ("O4'", "C1'", 'N1', 'C2')


def nucleotide_torsions_numpy(
    xyz_by_name_cur,
    xyz_by_name_prev,
    xyz_by_name_next,
    base_one_letter,
):
    """Return torsions, mask, τ_m and τ_m validity for a nucleotide."""

    def get_xyz(atom_dict, atom_name):
        if atom_dict is None or atom_name not in atom_dict:
            return None
        return np.asarray(atom_dict[atom_name], dtype=np.float64).reshape(3)

    torsions = np.zeros(N_TORSIONS, dtype=np.float64)
    mask = np.zeros(N_TORSIONS, dtype=bool)

    o3_prev = get_xyz(xyz_by_name_prev, "O3'")
    p_cur = get_xyz(xyz_by_name_cur, 'P')
    o5 = get_xyz(xyz_by_name_cur, "O5'")
    c5 = get_xyz(xyz_by_name_cur, "C5'")
    c4 = get_xyz(xyz_by_name_cur, "C4'")
    c3 = get_xyz(xyz_by_name_cur, "C3'")
    o3_cur = get_xyz(xyz_by_name_cur, "O3'")
    p_next = get_xyz(xyz_by_name_next, 'P')
    o5_next = get_xyz(xyz_by_name_next, "O5'")

    if all(x is not None for x in (o3_prev, p_cur, o5, c5)):
        torsions[TOR_ALPHA] = dihedral_rad(o3_prev, p_cur, o5, c5)
        mask[TOR_ALPHA] = True
    if all(x is not None for x in (p_cur, o5, c5, c4)):
        torsions[TOR_BETA] = dihedral_rad(p_cur, o5, c5, c4)
        mask[TOR_BETA] = True
    if all(x is not None for x in (o5, c5, c4, c3)):
        torsions[TOR_GAMMA] = dihedral_rad(o5, c5, c4, c3)
        mask[TOR_GAMMA] = True
    if all(x is not None for x in (c4, c3, o3_cur, p_next)):
        torsions[TOR_EPS] = dihedral_rad(c4, c3, o3_cur, p_next)
        mask[TOR_EPS] = True
    if all(x is not None for x in (c3, o3_cur, p_next, o5_next)):
        torsions[TOR_ZETA] = dihedral_rad(c3, o3_cur, p_next, o5_next)
        mask[TOR_ZETA] = True

    o4 = get_xyz(xyz_by_name_cur, "O4'")
    c1 = get_xyz(xyz_by_name_cur, "C1'")
    a0, a1, a2, a3 = _chi_quads(base_one_letter)
    atom2 = get_xyz(xyz_by_name_cur, a2)
    atom3 = get_xyz(xyz_by_name_cur, a3)
    if all(x is not None for x in (o4, c1, atom2, atom3)):
        torsions[TOR_CHI] = dihedral_rad(o4, c1, atom2, atom3)
        mask[TOR_CHI] = True

    nu_deg = []
    for atom0, atom1, atom2, atom3 in RING_TORSION_DEFS:
        pts = [
            get_xyz(xyz_by_name_cur, atom0),
            get_xyz(xyz_by_name_cur, atom1),
            get_xyz(xyz_by_name_cur, atom2),
            get_xyz(xyz_by_name_cur, atom3),
        ]
        if not all(x is not None for x in pts):
            nu_deg = None
            break
        nu_deg.append(float(np.degrees(dihedral_rad(pts[0], pts[1], pts[2], pts[3]))))

    tau_m_val = 0.0
    if nu_deg is not None:
        nu_arr = np.asarray(nu_deg, dtype=np.float64)
        p_rad = pseudorotation_P_rad_from_nus(nu_arr)
        torsions[TOR_PUCKER_P] = p_rad
        mask[TOR_PUCKER_P] = True
        tau_m_val = float(pucker_amplitude_rad(nu_arr, p_rad))
    tau_m_valid = nu_deg is not None
    return torsions, mask, tau_m_val, tau_m_valid


@functools.lru_cache(maxsize=None)
def _pseudorotation_offsets_torch(
    device_str: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(
        _PSEUDOROTATION_OFFSETS,
        device=torch.device(device_str),
        dtype=dtype,
    )


def nus_rad_from_P_tau_torch(
    P_rad: torch.Tensor,
    tau_m: torch.Tensor,
) -> torch.Tensor:
    """Return five endocyclic ν in the same order as `RING_TORSION_DEFS`."""
    p = P_rad.reshape(-1, 1)
    t = tau_m.reshape(-1, 1)
    offsets = _pseudorotation_offsets_torch(str(P_rad.device), P_rad.dtype)
    return t * torch.cos(p + offsets.unsqueeze(0))


pseudorotation_to_nus_torch = nus_rad_from_P_tau_torch


def _ring_dihedrals_from_coords_torch(ring_atoms: dict[str, torch.Tensor]) -> torch.Tensor:
    out = []
    for atom0, atom1, atom2, atom3 in RING_TORSION_DEFS:
        out.append(
            dihedral_rad_torch(
                ring_atoms[atom0],
                ring_atoms[atom1],
                ring_atoms[atom2],
                ring_atoms[atom3],
            ).unsqueeze(-1),
        )
    return torch.cat(out, dim=-1)


def sugar_ring_torsions_torch(atoms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return endocyclic ν torsions in `RING_TORSION_DEFS` order."""
    return _ring_dihedrals_from_coords_torch(atoms)
