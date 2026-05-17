"""Torsion extraction and pseudorotation helpers."""

import functools
import math

import numpy as np
import torch

from ..torsion_constants import (
    N_TORSIONS,
    TOR_BRIDGE_PHASE,
    TOR_CHI,
    TOR_DELTA,
    TOR_ETA_P,
    TOR_GAMMA,
    TOR_PSEUDOROTATION_PHASE,
)
from .primitives import GEO_EPS_SQ, _dihedral_rad, dihedral_rad

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
_TEMPLATE_RENAMES = {'O1P': 'OP1', 'O2P': 'OP2', 'O1A': 'OP1', 'O2A': 'OP2'}


def pseudorotation_phase_rad_from_nus(nu_deg):
    """Return MDAnalysis `phase_as` phase angle in radians from five endocyclic torsions."""
    nu = np.asarray(nu_deg, dtype=np.float64)
    b = np.dot(nu, np.sin(_RING_ANGLES)) * (-2.0 / 5.0)
    a = np.dot(nu, np.cos(_RING_ANGLES)) * (2.0 / 5.0)
    return float(np.arctan2(b, a))


def pucker_amplitude_rad(nu_deg, P_rad):
    """Return τ_m in radians with pseudorotation phase held fixed."""
    nu = np.deg2rad(np.asarray(nu_deg, dtype=np.float64))
    cos_phase = np.cos(P_rad + _PSEUDOROTATION_OFFSETS)
    denom = float(np.dot(cos_phase, cos_phase)) + 1e-12
    tau = float(np.dot(nu, cos_phase) / denom)
    return abs(tau)


def _chi_quads(base_one_letter):
    if base_one_letter in ('A', 'G'):
        return ("O4'", "C1'", 'N9', 'C4')
    return ("O4'", "C1'", 'N1', 'C2')


@functools.lru_cache(maxsize=None)
def _bridge_template_bond_lengths(base_one_letter: str) -> tuple[float, float]:
    from pynamod.atomic_analysis.nucleotides_parser import get_base_u  # lazy import

    template: dict[str, np.ndarray] = {}
    for atom in get_base_u(base_one_letter):  # type: ignore[union-attr]
        raw_name = getattr(atom, 'name', None)
        if raw_name is None:
            continue
        atom_name = (
            _TEMPLATE_RENAMES[raw_name]
            if raw_name in _TEMPLATE_RENAMES
            else raw_name.rstrip('AB')
        )
        if 'H' in atom_name or getattr(atom, 'element', None) in {'H', 'D'}:
            continue
        template[atom_name] = np.asarray(atom.position, dtype=np.float64)
    r_b = float(np.linalg.norm(template['P'] - template["O5'"]))
    r_a = r_b
    return r_a, r_b


def _safe_normalize_torch(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=eps)


def bridge_circle_geometry_torch(
    anchor_a: torch.Tensor,
    anchor_b: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = anchor_b - anchor_a
    d = axis.norm(dim=-1).clamp(min=eps)
    e = axis / d.unsqueeze(-1)
    a = (r_a * r_a - r_b * r_b + d * d) / (2.0 * d)
    center = anchor_a + a.unsqueeze(-1) * e
    # Infeasible/degenerate bridge circles have radius^2 = 0. Keep sqrt backward finite.
    radius = torch.sqrt((r_a * r_a - a * a).clamp(min=0.0) + GEO_EPS_SQ)

    x_axis = torch.zeros_like(e)
    x_axis[..., 0] = 1.0
    y_axis = torch.zeros_like(e)
    y_axis[..., 1] = 1.0
    ref = torch.where(e[..., 0].abs().unsqueeze(-1) >= 0.9, y_axis, x_axis)
    u = _safe_normalize_torch(torch.linalg.cross(ref, e, dim=-1), eps=eps)
    v = torch.linalg.cross(e, u, dim=-1)
    return center, radius, u, v


def phosphate_from_bridge_phase_torch(
    anchor_a: torch.Tensor,
    anchor_b: torch.Tensor,
    phase: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    center, radius, u, v = bridge_circle_geometry_torch(
        anchor_a,
        anchor_b,
        r_a,
        r_b,
        eps=eps,
    )
    return center + radius.unsqueeze(-1) * (
        torch.cos(phase).unsqueeze(-1) * u
        + torch.sin(phase).unsqueeze(-1) * v
    )


def bridge_phase_from_points_torch(
    anchor_a: torch.Tensor,
    anchor_b: torch.Tensor,
    phosphate: torch.Tensor,
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    center, _radius, u, v = bridge_circle_geometry_torch(
        anchor_a,
        anchor_b,
        r_a,
        r_b,
        eps=eps,
    )
    q = phosphate - center
    x = (q * u).sum(dim=-1)
    y = (q * v).sum(dim=-1)
    # atan2(0, 0) is finite forward but has undefined backward; use the same guard as dihedral_rad.
    ok = (x * x + y * y) > GEO_EPS_SQ
    x = torch.where(ok, x, torch.ones_like(x))
    y = torch.where(ok, y, torch.zeros_like(y))
    return torch.atan2(y, x)


def bridge_phase_from_points(
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    phosphate: np.ndarray,
    r_a: float,
    r_b: float,
) -> float:
    phase = bridge_phase_from_points_torch(
        torch.as_tensor(anchor_a, dtype=torch.float64).reshape(1, 3),
        torch.as_tensor(anchor_b, dtype=torch.float64).reshape(1, 3),
        torch.as_tensor(phosphate, dtype=torch.float64).reshape(1, 3),
        torch.tensor([r_a], dtype=torch.float64),
        torch.tensor([r_b], dtype=torch.float64),
    )
    return float(phase.item())


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

    _set(TOR_GAMMA, o5, c5, c4, c3)
    _set(TOR_DELTA, c5, c4, c3, o3_cur)
    if o3_prev is not None and p_cur is not None and o5 is not None:
        r_a, r_b = _bridge_template_bond_lengths(base_one_letter)
        torsions[TOR_BRIDGE_PHASE] = bridge_phase_from_points(o3_prev, o5, p_cur, r_a, r_b)
        mask[TOR_BRIDGE_PHASE] = True

    op1 = g(xyz_by_name_cur, 'OP1')
    _set(TOR_ETA_P, c5, o5, p_cur, op1)  # dihedral(C5', O5', P, OP1)

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
