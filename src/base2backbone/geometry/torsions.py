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
    TOR_PSEUDOROTATION_PHASE,
    TOR_ZETA,
)
from .primitives import _dihedral_rad, dihedral_rad

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


def pseudorotation_phase_rad_from_nus(nu_deg):
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


def nucleotide_torsions(xyz_by_name_cur, xyz_by_name_prev, xyz_by_name_next, base_one_letter):
    """Return torsions, mask, τ_m and τ_m validity for a nucleotide."""

    def g(d, name):
        if d is None or name not in d:
            return None
        return np.asarray(d[name], dtype=np.float64).reshape(3)

    torsions = np.zeros(N_TORSIONS, dtype=np.float64)
    mask = np.zeros(N_TORSIONS, dtype=bool)

    o3_prev = g(xyz_by_name_prev, "O3'")
    p_cur = g(xyz_by_name_cur, 'P')
    o5 = g(xyz_by_name_cur, "O5'")
    c5 = g(xyz_by_name_cur, "C5'")
    c4 = g(xyz_by_name_cur, "C4'")
    c3 = g(xyz_by_name_cur, "C3'")
    o3_cur = g(xyz_by_name_cur, "O3'")
    p_next = g(xyz_by_name_next, 'P')
    o5_next = g(xyz_by_name_next, "O5'")

    def _set(idx, *pts):
        if all(x is not None for x in pts):
            torsions[idx] = _dihedral_rad(*pts)
            mask[idx] = True

    _set(TOR_ALPHA, o3_prev, p_cur, o5, c5)
    _set(TOR_BETA,  p_cur, o5, c5, c4)
    _set(TOR_GAMMA, o5, c5, c4, c3)
    _set(TOR_EPS,   c4, c3, o3_cur, p_next)
    _set(TOR_ZETA,  c3, o3_cur, p_next, o5_next)

    o4 = g(xyz_by_name_cur, "O4'")
    c1 = g(xyz_by_name_cur, "C1'")
    a0, a1, a2, a3 = _chi_quads(base_one_letter)
    _set(TOR_CHI, o4, c1, g(xyz_by_name_cur, a2), g(xyz_by_name_cur, a3))

    nu_deg = []
    for atom0, atom1, atom2, atom3 in RING_TORSION_DEFS:
        pts = [g(xyz_by_name_cur, n) for n in (atom0, atom1, atom2, atom3)]
        if not all(x is not None for x in pts):
            nu_deg = None
            break
        nu_deg.append(float(np.degrees(_dihedral_rad(*pts))))

    tau_m_val = 0.0
    if nu_deg is not None:
        nu_arr = np.asarray(nu_deg, dtype=np.float64)
        p_rad = pseudorotation_phase_rad_from_nus(nu_arr)
        torsions[TOR_PSEUDOROTATION_PHASE] = p_rad
        mask[TOR_PSEUDOROTATION_PHASE] = True
        tau_m_val = float(pucker_amplitude_rad(nu_arr, p_rad))
    tau_m_valid = nu_deg is not None
    return torsions, mask, tau_m_val, tau_m_valid


@functools.lru_cache(maxsize=None)
def _pseudorotation_offsets(
    device_str: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(
        _PSEUDOROTATION_OFFSETS,
        device=torch.device(device_str),
        dtype=dtype,
    )


def nus_rad_from_phase_and_amplitude(
    P_rad: torch.Tensor,
    tau_m: torch.Tensor,
) -> torch.Tensor:
    """Return five endocyclic ν in the same order as `RING_TORSION_DEFS`."""
    p = P_rad.reshape(-1, 1)
    t = tau_m.reshape(-1, 1)
    offsets = _pseudorotation_offsets(str(P_rad.device), P_rad.dtype)
    return t * torch.cos(p + offsets.unsqueeze(0))


phase_and_amplitude_to_nus = nus_rad_from_phase_and_amplitude


def _ring_dihedrals_from_coords(ring_atoms: dict[str, torch.Tensor]) -> torch.Tensor:
    out = []
    for atom0, atom1, atom2, atom3 in RING_TORSION_DEFS:
        out.append(
            dihedral_rad(
                ring_atoms[atom0],
                ring_atoms[atom1],
                ring_atoms[atom2],
                ring_atoms[atom3],
            ).unsqueeze(-1),
        )
    return torch.cat(out, dim=-1)


def sugar_ring_torsions(atoms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return endocyclic ν torsions in `RING_TORSION_DEFS` order."""
    return _ring_dihedrals_from_coords(atoms)
