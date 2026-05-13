"""Torsion definitions, sugar puckering (Altona–Sundaralingam / MDAnalysis conventions), and angle wrapping."""

import functools
import math
from typing import Any, Optional, cast

import numpy as np
import torch

from .torsion_constants import (N_LATENT, N_TORSIONS, N_TORSIONS_LATENT,
                                TOR_ALPHA, TOR_BETA, TOR_CHI, TOR_EPS,
                                TOR_GAMMA, TOR_PUCKER_P, TOR_ZETA,
                                TORSION_IS_CIRCULAR, TORSION_NAMES)


def world_to_local_points(
    x_world: torch.Tensor,
    origin: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    """Row-vector convention: ``local = (world - origin) @ frame`` (same as training frames)."""
    d = x_world - _expand_point_origin(origin, x_world)
    if frame.dim() == 2:
        return torch.matmul(d, frame)
    return torch.matmul(d.unsqueeze(-2), _expand_point_frame(frame, d)).squeeze(-2)


def local_to_world_points(
    x_local: torch.Tensor,
    origin: torch.Tensor,
    frame: torch.Tensor,
) -> torch.Tensor:
    """Row-vector convention: ``world = local @ frame.T + origin``."""
    r = frame.transpose(-2, -1)
    origin_exp = _expand_point_origin(origin, x_local)
    if frame.dim() == 2:
        return torch.matmul(x_local, r) + origin_exp
    return torch.matmul(
        x_local.unsqueeze(-2),
        _expand_point_frame(r, x_local),
    ).squeeze(-2) + origin_exp


world_to_local_torch = world_to_local_points
local_to_world_torch = local_to_world_points


_GEO_EPS = 1e-8
_GEO_EPS_NP = 1e-12
_PHI_PHOS_GRID = 128
_HALF_PI = math.pi / 2.0
_PSEUDOROTATION_OFFSETS = (
    np.array([0.0, 4.0, 8.0, 2.0, 6.0], dtype=np.float64) * (math.pi / 5.0)
)


def _expand_point_origin(origin: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    while origin.ndim < points.ndim:
        origin = origin.unsqueeze(-2)
    return origin


def _expand_point_frame(frame: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    while frame.ndim < points.ndim + 1:
        frame = frame.unsqueeze(-3)
    return frame


def wrap_angle_rad(x):
    """Map angles to (-π, π]."""
    return np.arctan2(np.sin(x), np.cos(x))


def _bond_angle_np(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle ∠(a–b–c) in radians."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba)) * float(np.linalg.norm(bc)) + _GEO_EPS_NP
    cos_t = float(np.dot(ba, bc)) / denom
    cos_t = float(np.clip(cos_t, -1.0 + 1e-9, 1.0 - 1e-9))
    return float(np.arccos(cos_t))


def _pseudorotation_nus_numpy(P_rad: float, tau_m: float) -> np.ndarray:
    """Return ν₂…ν₁ cycle from pseudorotation phase/amplitude."""
    return float(tau_m) * np.cos(float(P_rad) + _PSEUDOROTATION_OFFSETS)


def nerf_place(a, b, c, r, theta, psi):
    """Place D given prior atoms A–B–C, bond C–D length r, interior angle ∠(B–C–D)=theta, dihedral(A,B,C,D)=psi (rad)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    c = np.asarray(c, dtype=np.float64).reshape(3)
    ba = a - b
    bc = c - b
    bc_u = bc / (np.linalg.norm(bc) + _GEO_EPS_NP)
    n = np.cross(ba, bc_u)
    nn = np.linalg.norm(n)
    if nn < 1e-10:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        n = n / nn
    m = np.cross(n, bc_u)
    return c + r * (
        np.cos(math.pi - theta) * bc_u
        + np.sin(math.pi - theta) * (np.cos(psi) * n + np.sin(psi) * m)
    )


def nerf_place_torch(
    a: 'torch.Tensor',  # [B, 3]
    b: 'torch.Tensor',  # [B, 3]
    c: 'torch.Tensor',  # [B, 3]
    r: 'torch.Tensor',  # [B]
    theta: 'torch.Tensor',  # [B]
    psi: 'torch.Tensor',    # [B]
) -> 'torch.Tensor':        # [B, 3]
    """Batched differentiable NERF. Mirrors nerf_place exactly."""
    import torch
    ba = a - b
    bc = c - b
    bc_u = bc / (bc.norm(dim=-1, keepdim=True) + _GEO_EPS_NP)
    n = torch.linalg.cross(ba, bc_u)
    nn = n.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(n)
    fallback[..., 0] = 1.0
    n = torch.where(nn < 1e-10, fallback, n / (nn + _GEO_EPS_NP))
    m = torch.linalg.cross(n, bc_u)
    r_ = r.unsqueeze(-1)
    return c + r_ * (
        torch.cos(theta.new_tensor(math.pi) - theta).unsqueeze(-1) * bc_u
        + torch.sin(theta.new_tensor(math.pi) - theta).unsqueeze(-1) * (
            torch.cos(psi).unsqueeze(-1) * n
            + torch.sin(psi).unsqueeze(-1) * m
        )
    )


def dihedral_rad(p0, p1, p2, p3):
    """Signed dihedral angle (radians) for points p0–p3 (column vectors in R^3)."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_u = n1 / (np.linalg.norm(n1) + _GEO_EPS_NP)
    n2_u = n2 / (np.linalg.norm(n2) + _GEO_EPS_NP)
    m1 = np.cross(n1_u, b2 / (np.linalg.norm(b2) + _GEO_EPS_NP))
    x = np.dot(n1_u, n2_u)
    y = np.dot(m1, n2_u)
    return float(np.arctan2(y, x))


def close_phosphate_bridge(
    o3: np.ndarray,
    o5: np.ndarray,
    c5: np.ndarray,
    r_o3p: float,
    r_po5: float,
    alpha_hint: float,
) -> np.ndarray:
    """Place P on the O3'–O5' intersecting circle matching O3'–P and P–O5' lengths.

    Picks the point whose α dihedral O3'–P–O5'–C5' best matches ``alpha_hint``.
    """
    o3 = np.asarray(o3, dtype=np.float64).reshape(3)
    o5 = np.asarray(o5, dtype=np.float64).reshape(3)
    c5 = np.asarray(c5, dtype=np.float64).reshape(3)
    d = float(np.linalg.norm(o5 - o3))
    sum_l = r_o3p + r_po5
    diff_l = abs(r_o3p - r_po5)
    eps = 1e-12
    if d > sum_l + eps or d < diff_l - eps:
        o5_minus_o3 = o5 - o3
        dn = float(np.linalg.norm(o5_minus_o3))
        if dn < eps:
            return o3.copy()
        return o3 + o5_minus_o3 * (r_o3p / sum_l)
    a = (r_o3p * r_o3p - r_po5 * r_po5 + d * d) / (2.0 * d)
    h = float(np.sqrt(max(r_o3p * r_o3p - a * a, 0.0)))
    axis = (o5 - o3) / d
    midpoint = o3 + a * axis
    tmp = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(axis[0]) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    v1 = np.cross(tmp, axis)
    v1n = float(np.linalg.norm(v1))
    if v1n < eps:
        v1 = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), axis)
        v1n = float(np.linalg.norm(v1))
    v1 = v1 / v1n
    v2 = np.cross(axis, v1)

    n_samples = 360
    best_t = 0.0
    best_err = float('inf')
    for k in range(n_samples):
        t = 2.0 * np.pi * k / n_samples
        p_t = midpoint + h * (np.cos(t) * v1 + np.sin(t) * v2)
        dihed = dihedral_rad(o3, p_t, o5, c5)
        err = abs(wrap_angle_rad(dihed - alpha_hint))
        if err < best_err:
            best_err = err
            best_t = t
    return midpoint + h * (np.cos(best_t) * v1 + np.sin(best_t) * v2)


_RING_TORSION_DEFS = [
    ("C1'", "C2'", "C3'", "C4'"),
    ("C2'", "C3'", "C4'", "O4'"),
    ("C3'", "C4'", "O4'", "C1'"),
    ("C4'", "O4'", "C1'", "C2'"),
    ("O4'", "C1'", "C2'", "C3'"),
]
_RING_ANGLES = 2.0 * 2.0 * math.pi * np.arange(5, dtype=np.float64) / 5.0


def pseudorotation_P_rad_from_nus(nu_deg):
    """MDAnalysis `phase_as` phase angle in radians from five endocyclic torsions (degrees)."""
    nu = np.asarray(nu_deg, dtype=np.float64)
    B = np.dot(nu, np.sin(_RING_ANGLES)) * (-2.0 / 5.0)
    A = np.dot(nu, np.cos(_RING_ANGLES)) * (2.0 / 5.0)
    return float(np.arctan2(B, A))


def pucker_amplitude_rad(nu_deg, P_rad):
    """Amplitude τ_m (rad): LS fit ν_i ≈ τ_m cos(P + 2π i/5) with P fixed; five ν in degree (MDAnalysis order)."""
    nu = np.deg2rad(np.asarray(nu_deg, dtype=np.float64))
    idx = np.arange(5, dtype=np.float64)
    phases = P_rad + (2.0 * math.pi * idx / 5.0)
    c = np.cos(phases)
    den = float(np.dot(c, c)) + 1e-12
    tau = float(np.dot(nu, c) / np.sqrt(den))
    return abs(tau)


def _chi_quads(base_one_letter):
    if base_one_letter in ('A', 'G'):
        return ("O4'", "C1'", "N9", "C4")
    return ("O4'", "C1'", "N1", "C2")


def nucleotide_torsions_numpy(
    xyz_by_name_cur,
    xyz_by_name_prev,
    xyz_by_name_next,
    base_one_letter,
):
    """Return (torsions [N_TORSIONS], mask [N_TORSIONS], tau_m [rad], tau_m_valid).

    tau_m_valid is False when sugar ring ν torsions were incomplete.
    """

    def g(d, name):
        if d is None or name not in d:
            return None
        return np.asarray(d[name], dtype=np.float64).reshape(3)

    t = np.zeros(N_TORSIONS, dtype=np.float64)
    m = np.zeros(N_TORSIONS, dtype=bool)

    o3_prev = g(xyz_by_name_prev, "O3'")
    p_c = g(xyz_by_name_cur, 'P')
    o5 = g(xyz_by_name_cur, "O5'")
    c5 = g(xyz_by_name_cur, "C5'")
    c4 = g(xyz_by_name_cur, "C4'")
    c3 = g(xyz_by_name_cur, "C3'")
    o3_c = g(xyz_by_name_cur, "O3'")
    p_next = g(xyz_by_name_next, 'P')
    o5_next = g(xyz_by_name_next, "O5'")

    # α O3'(i-1)–P–O5'–C5'
    if all(x is not None for x in (o3_prev, p_c, o5, c5)):
        t[TOR_ALPHA] = dihedral_rad(o3_prev, p_c, o5, c5)
        m[TOR_ALPHA] = True
    # β P–O5'–C5'–C4'
    if all(x is not None for x in (p_c, o5, c5, c4)):
        t[TOR_BETA] = dihedral_rad(p_c, o5, c5, c4)
        m[TOR_BETA] = True
    # γ O5'–C5'–C4'–C3'
    if all(x is not None for x in (o5, c5, c4, c3)):
        t[TOR_GAMMA] = dihedral_rad(o5, c5, c4, c3)
        m[TOR_GAMMA] = True
    # ε C4'–C3'–O3'–P(i+1)
    if all(x is not None for x in (c4, c3, o3_c, p_next)):
        t[TOR_EPS] = dihedral_rad(c4, c3, o3_c, p_next)
        m[TOR_EPS] = True
    # ζ C3'–O3'–P(i+1)–O5'(i+1)
    if all(x is not None for x in (c3, o3_c, p_next, o5_next)):
        t[TOR_ZETA] = dihedral_rad(c3, o3_c, p_next, o5_next)
        m[TOR_ZETA] = True

    o4 = g(xyz_by_name_cur, "O4'")
    c1 = g(xyz_by_name_cur, "C1'")
    a0, a1, a2, a3 = _chi_quads(base_one_letter)
    ap2 = g(xyz_by_name_cur, a2)
    ap3 = g(xyz_by_name_cur, a3)
    if all(x is not None for x in (o4, c1, ap2, ap3)):
        t[TOR_CHI] = dihedral_rad(o4, c1, ap2, ap3)
        m[TOR_CHI] = True

    nu_deg = []
    for a0n, a1n, a2n, a3n in _RING_TORSION_DEFS:
        pts = [g(xyz_by_name_cur, a0n), g(xyz_by_name_cur, a1n), g(xyz_by_name_cur, a2n), g(xyz_by_name_cur, a3n)]
        if not all(x is not None for x in pts):
            nu_deg = None
            break
        nu_deg.append(float(np.degrees(dihedral_rad(pts[0], pts[1], pts[2], pts[3]))))
    tau_m_val = 0.0
    if nu_deg is not None:
        nu_arr = np.asarray(nu_deg, dtype=np.float64)
        P_rad = pseudorotation_P_rad_from_nus(nu_arr)
        t[TOR_PUCKER_P] = P_rad
        m[TOR_PUCKER_P] = True
        tau_m_val = float(pucker_amplitude_rad(nu_arr, P_rad))
    tau_m_valid = nu_deg is not None
    return t, m, tau_m_val, tau_m_valid


# Cache template atom dicts per restype to avoid repeated pynamod calls.
_template_cache: dict = {}


_template_tau_m: dict = {}  # per-restype canonical ring amplitude τm_AS (radians)


def _get_template(restype: str) -> 'dict[str, np.ndarray]':
    """Return heavy-atom positions (local frame, Å) for the canonical nucleotide."""
    if restype in _template_cache:
        return _template_cache[restype]
    from pynamod.atomic_analysis.nucleotides_parser import \
        get_base_u  # lazy import
    _RENAMES = {'O1P': 'OP1', 'O2P': 'OP2', 'O1A': 'OP1', 'O2A': 'OP2'}
    tpl: dict = {}
    for atom in get_base_u(restype):  # type: ignore[union-attr]
        if 'H' in atom.name or getattr(atom, 'element', None) in {'H', 'D'}:
            continue
        nm = _RENAMES.get(atom.name, atom.name.rstrip('AB'))
        tpl[nm] = np.asarray(atom.position, dtype=np.float64)
    # Compute canonical ring amplitude τm_AS = sqrt(2/5 · Σ νₖ²) from template.
    # This is consistent with the A-S inverse formula νₖ = τm cos(P + offsets[k]).
    nu_rad = np.array([
        dihedral_rad(tpl[q[0]], tpl[q[1]], tpl[q[2]], tpl[q[3]])
        for q in _RING_TORSION_DEFS
    ])
    _template_tau_m[restype] = float(np.sqrt(0.4 * float(np.dot(nu_rad, nu_rad))))
    _template_cache[restype] = tpl
    return tpl


@functools.lru_cache(maxsize=None)
def _get_template_tensors(device_str: str) -> dict:
    """Return all template geometry constants as float32 tensors on `device`.

    All tensors have shape [4, ...] — one entry per restype in order A C G T.
    Cached per device string; call once per device.
    """
    import torch
    device = torch.device(device_str)
    restypes = ['A', 'C', 'G', 'T']

    def _bl(tpl, a, b):
        return float(np.linalg.norm(tpl[a] - tpl[b]))

    def _ba(tpl, a, b, c):
        return _bond_angle_np(tpl[a], tpl[b], tpl[c])

    def _dr(tpl, a, b, c, d):
        return float(dihedral_rad(tpl[a], tpl[b], tpl[c], tpl[d]))

    keys_3d = ['c1', 'o4', 'c2_ref', 'c4_ref', 'chi_n', 'chi_c']
    keys_1d = [
        'bl_c2_c1', 'ba_o4_c1_c2',
        'bl_c3_c2', 'ba_c1_c2_c3',
        'bl_c4_c3', 'ba_c2_c3_c4',
        'bl_c5_c4', 'ba_c3_c4_c5', 'psi_c5',
        'bl_o5_c5', 'ba_c4_c5_o5',
        'bl_o3_c3', 'ba_c4_c3_o3',
        'r_po3',
        'bond_p_o5',
        'bond_p_o3_inter',
        'tpl_o3_sep_p',
        'bl_op1', 'bl_op2',
        'ang_op1', 'ang_op2',
        'psi_op1', 'psi_op2',
        'psi_o3', 'psi_o3_ring', 'bl_o4_c4', 'ba_c1_o4_c4', 'ba_o4_c4_c3',
        'ring_chiral_triple',
    ]

    rows_3d = {k: [] for k in keys_3d}
    rows_1d = {k: [] for k in keys_1d}

    for rt in restypes:
        tpl = _get_template(rt)
        rows_3d['c1'].append(tpl["C1'"])
        rows_3d['o4'].append(tpl["O4'"])
        rows_3d['c2_ref'].append(tpl["C2'"])
        rows_3d['c4_ref'].append(tpl["C4'"])
        if rt in ('A', 'G'):
            rows_3d['chi_n'].append(tpl['N9'])
            rows_3d['chi_c'].append(tpl['C4'])
        else:
            rows_3d['chi_n'].append(tpl['N1'])
            rows_3d['chi_c'].append(tpl['C2'])

        rows_1d['bl_c2_c1'].append(_bl(tpl, "C2'", "C1'"))
        rows_1d['ba_o4_c1_c2'].append(_ba(tpl, "O4'", "C1'", "C2'"))
        rows_1d['bl_c3_c2'].append(_bl(tpl, "C3'", "C2'"))
        rows_1d['ba_c1_c2_c3'].append(_ba(tpl, "C1'", "C2'", "C3'"))
        rows_1d['bl_c4_c3'].append(_bl(tpl, "C4'", "C3'"))
        rows_1d['ba_c2_c3_c4'].append(_ba(tpl, "C2'", "C3'", "C4'"))
        rows_1d['bl_c5_c4'].append(_bl(tpl, "C5'", "C4'"))
        rows_1d['ba_c3_c4_c5'].append(_ba(tpl, "C3'", "C4'", "C5'"))
        rows_1d['psi_c5'].append(_dr(tpl, "O4'", "C3'", "C4'", "C5'"))
        rows_1d['bl_o5_c5'].append(_bl(tpl, "O5'", "C5'"))
        rows_1d['ba_c4_c5_o5'].append(_ba(tpl, "C4'", "C5'", "O5'"))
        rows_1d['bl_o3_c3'].append(_bl(tpl, "O3'", "C3'"))
        rows_1d['ba_c4_c3_o3'].append(_ba(tpl, "C4'", "C3'", "O3'"))
        po5_len = _bl(tpl, 'P', "O5'")
        rows_1d['r_po3'].append(po5_len)
        rows_1d['bond_p_o5'].append(po5_len)
        # O3'_i–P_{i+1} phosphodiester: do not use same-residue |O3'−P| (not a bonded geometry).
        rows_1d['bond_p_o3_inter'].append(po5_len)
        rows_1d['tpl_o3_sep_p'].append(_bl(tpl, "O3'", 'P'))
        rows_1d['bl_op1'].append(_bl(tpl, 'P', 'OP1'))
        rows_1d['bl_op2'].append(_bl(tpl, 'P', 'OP2'))
        rows_1d['ang_op1'].append(_ba(tpl, "O5'", 'P', 'OP1'))
        rows_1d['ang_op2'].append(_ba(tpl, "O5'", 'P', 'OP2'))
        rows_1d['psi_op1'].append(_dr(tpl, "O3'", "O5'", 'P', 'OP1'))
        rows_1d['psi_op2'].append(_dr(tpl, "O3'", "O5'", 'P', 'OP2'))
        rows_1d['psi_o3'].append(_dr(tpl, "C5'", "C4'", "C3'", "O3'"))
        rows_1d['psi_o3_ring'].append(_dr(tpl, "O4'", "C4'", "C3'", "O3'"))
        rows_1d['bl_o4_c4'].append(_bl(tpl, "O4'", "C4'"))
        rows_1d['ba_c1_o4_c4'].append(_ba(tpl, "C1'", "O4'", "C4'"))
        rows_1d['ba_o4_c4_c3'].append(_ba(tpl, "O4'", "C4'", "C3'"))
        _v2 = tpl["C2'"] - tpl["O4'"]
        _v3 = tpl["C3'"] - tpl["O4'"]
        _v4 = tpl["C4'"] - tpl["O4'"]
        rows_1d['ring_chiral_triple'].append(
            float(np.dot(np.cross(_v2, _v3), _v4)),
        )

    out = {}
    for k, v in rows_3d.items():
        out[k] = torch.tensor(np.array(v), dtype=torch.float32, device=device)
    for k, v in rows_1d.items():
        out[k] = torch.tensor(v, dtype=torch.float32, device=device)
    return out


# --- Torch geometry decoder (sugar ring: planar canonical + pseudorotation; phosphate uses φ grid) ---


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp(min=eps))


def _orthonormal_basis_from_axis(
    axis: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return unit axis and two perpendicular unit vectors (batch [...])."""
    u = _safe_normalize(axis, eps=eps)
    tmp = torch.zeros_like(u)
    tmp[..., 0] = 1.0
    use_alt = u[..., 0].abs() >= 0.9
    tmp[use_alt, 0] = 0.0
    tmp[use_alt, 1] = 1.0
    e1 = torch.linalg.cross(tmp, u, dim=-1)
    e1 = _safe_normalize(e1, eps=eps)
    e2 = torch.linalg.cross(u, e1, dim=-1)
    return u, e1, e2


def dihedral_rad_torch(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
) -> torch.Tensor:
    """Batched signed dihedral in radians; shapes broadcast to [..., 3]."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)
    b2u = b2 / (b2.norm(dim=-1, keepdim=True) + _GEO_EPS)
    n1n = n1.norm(dim=-1, keepdim=True) + _GEO_EPS
    n2n = n2.norm(dim=-1, keepdim=True) + _GEO_EPS
    n1u = n1 / n1n
    n2u = n2 / n2n
    m1 = torch.linalg.cross(n1u, b2u)
    x = (n1u * n2u).sum(dim=-1)
    y = (m1 * n2u).sum(dim=-1)
    return torch.atan2(y, x)


def _bond_angle_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps: float = _GEO_EPS,
) -> torch.Tensor:
    """Interior angle ∠(a–b–c) in radians; batched last dim 3."""
    ba = a - b
    bc = c - b
    denom = ba.norm(dim=-1) * bc.norm(dim=-1) + eps
    cos_t = (ba * bc).sum(dim=-1) / denom
    cos_t = cos_t.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos_t)


def wrap_dihedral_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = (a - b + torch.pi) % (2.0 * torch.pi) - torch.pi
    return d


def nus_rad_from_P_tau_torch(
    P_rad: torch.Tensor,
    tau_m: torch.Tensor,
) -> torch.Tensor:
    """Five endocyclic ν (rad) in the same order as _RING_TORSION_DEFS (ν₂…ν₁ cycle)."""
    p = P_rad.reshape(-1, 1)
    t = tau_m.reshape(-1, 1)
    offs = _pseudorotation_offsets_torch(str(P_rad.device), P_rad.dtype)
    return t * torch.cos(p + offs.unsqueeze(0))


pseudorotation_to_nus_torch = nus_rad_from_P_tau_torch


def _ring_dihedrals_from_coords_torch(
    ring_atoms: dict[str, torch.Tensor],
) -> torch.Tensor:
    """[B, 5] dihedrals in _RING_TORSION_DEFS order."""
    names = _RING_TORSION_DEFS
    out = []
    for a0n, a1n, a2n, a3n in names:
        out.append(
            dihedral_rad_torch(
                ring_atoms[a0n], ring_atoms[a1n],
                ring_atoms[a2n], ring_atoms[a3n],
            ).unsqueeze(-1),
        )
    return torch.cat(out, dim=-1)


def sugar_ring_torsions_torch(atoms: dict[str, torch.Tensor]) -> torch.Tensor:
    """Endocyclic ν₀…ν₄ (rad) in ``_RING_TORSION_DEFS`` order; same convention as ``pseudorotation_to_nus_torch``."""
    return _ring_dihedrals_from_coords_torch(atoms)


def signed_tetra_volume(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Signed volume ∝ (b−a)·((c−a)×(d−a)); shape matches broadcast of inputs."""
    return (torch.linalg.cross(b - a, c - a, dim=-1) * (d - a)).sum(dim=-1)


def _rodrigues_rotate_point_torch(
    v: torch.Tensor,
    k: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Rotate vectors v [...,3] around unit axes k [...,3] by theta [...]."""
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cross = torch.linalg.cross(k, v, dim=-1)
    dot = (k * v).sum(dim=-1, keepdim=True)
    return (
        v * cos_t.unsqueeze(-1)
        + cross * sin_t.unsqueeze(-1)
        + k * dot * (1.0 - cos_t.unsqueeze(-1))
    )


def _apply_chi_rotation_to_sugar_ring_torch(
    ring: dict[str, torch.Tensor],
    chi_target: torch.Tensor,
    n_atom: torch.Tensor,
    c_atom: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Align χ(O4', C1', N, base_C) with chi_target by rotating ring atoms around C1'→N."""
    c1 = ring["C1'"]
    o4 = ring["O4'"]
    axis = n_atom - c1
    axn = axis.norm(dim=-1, keepdim=True).clamp(min=_GEO_EPS)
    u = axis / axn
    chi_meas = dihedral_rad_torch(o4, c1, n_atom, c_atom)
    dchi = wrap_dihedral_diff_torch(chi_target, chi_meas)
    out = dict(ring)
    for nm in ("O4'", "C2'", "C3'", "C4'"):
        v = ring[nm] - c1
        out[nm] = _rodrigues_rotate_point_torch(v, u, dchi) + c1
    return out


# Indices 0..4 correspond to this atom order (canonical planar + z puckering).
_SUGAR_RING_BUILD_ATOM_ORDER: tuple[str, ...] = (
    "C4'", "O4'", "C1'", "C2'", "C3'",
)

_PLANAR_SUGAR_SPEC: Optional[dict[str, Any]] = None


def _fallback_mean_ring_xyz_acgt_order() -> np.ndarray:
    """Approximate mean sugar ring (Å) in ``_SUGAR_RING_BUILD_ATOM_ORDER`` when pynamod is unavailable.

    Puckered 5-ring (not flat) so PCA normal and chirality are defined; used only if template import fails.
    """
    ang = np.linspace(0.0, 2.0 * math.pi, 6, dtype=np.float64)[:-1] + 0.25
    r0 = 1.28
    r_var = np.array([0.03, -0.02, 0.01, -0.01, 0.02], dtype=np.float64)
    x = (r0 + r_var) * np.cos(ang)
    y = (r0 + r_var) * np.sin(ang)
    z = 0.045 * np.sin(2.0 * ang)
    verts_ring = np.stack([x, y, z], axis=1)
    # ring visit order O4', C1', C2', C3', C4' == angular indices 0..4
    ring_to_build = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    return verts_ring[ring_to_build]


def _mean_sugar_ring_xyz_from_templates() -> np.ndarray:
    ATOM_ORDER = _SUGAR_RING_BUILD_ATOM_ORDER
    restypes = ['A', 'C', 'G', 'T']
    acc: list[np.ndarray] = []
    for rt in restypes:
        tpl = _get_template(rt)
        acc.append(np.stack([tpl[n] for n in ATOM_ORDER], axis=0))
    return np.mean(np.stack(acc, axis=0), axis=0)


def _calibrate_planar_sugar_from_pts(pts_mean: np.ndarray) -> dict[str, Any]:
    """One-time fit: cyclic phase/coset + z-scale vs ``nus`` Altona–Sundaralingam formula (numpy)."""
    ATOM_ORDER = _SUGAR_RING_BUILD_ATOM_ORDER
    pts = np.asarray(pts_mean, dtype=np.float64)
    ctr = pts.mean(axis=0)
    X = pts - ctr
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    e1 = Vt[0].astype(np.float64)
    e2 = Vt[1].astype(np.float64)
    en = Vt[2].astype(np.float64)
    xy = np.stack([(X @ e1), (X @ e2)], axis=1)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    tpl_e = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in pairs], dtype=np.float64)
    can_e = np.array([np.linalg.norm(xy[i] - xy[j]) for i, j in pairs], dtype=np.float64).clip(min=1e-9)
    xy *= float(np.mean(tpl_e / can_e))
    base_phases = np.array([0.0, 4.0, 8.0, 2.0, 6.0], dtype=np.float64) * (np.pi / 5.0)
    P_grid = np.linspace(-math.pi, math.pi, 9, dtype=np.float64)
    tau_grid = np.array([0.28, 0.4, 0.55, 0.7], dtype=np.float64)

    def _ring_dict_from_xyz(pos: np.ndarray) -> dict[str, np.ndarray]:
        return {ATOM_ORDER[j]: pos[j] for j in range(5)}

    def _nus_np(d: dict[str, np.ndarray]) -> np.ndarray:
        return np.array(
            [
                dihedral_rad(d[a0], d[a1], d[a2], d[a3])
                for a0, a1, a2, a3 in _RING_TORSION_DEFS
            ],
            dtype=np.float64,
        )

    rch = float(
        np.dot(
            np.cross(pts[3] - pts[1], pts[4] - pts[1]),
            pts[0] - pts[1],
        ),
    )

    def _mean_nu_err_for(
        z_scale_: float,
        shift_: int,
        sp_: float,
        sz_: float,
        phase0_: float,
    ) -> float:
        rolled = np.roll(base_phases, shift_)
        phases_np = rolled * sz_ + phase0_
        errs: list[float] = []
        for Pv in P_grid:
            for tv in tau_grid:
                z = z_scale_ * tv * np.cos(sp_ * Pv + phases_np)
                pos = np.stack(
                    [
                        ctr + xy[j, 0] * e1 + xy[j, 1] * e2 + z[j] * en
                        for j in range(5)
                    ],
                    axis=0,
                )
                nu_act = _nus_np(_ring_dict_from_xyz(pos))
                nu_tgt = tv * np.cos(Pv + base_phases)
                dnu = np.arctan2(
                    np.sin(nu_act - nu_tgt),
                    np.cos(nu_act - nu_tgt),
                )
                errs.append(float(np.mean(np.abs(dnu))))
        return float(np.mean(errs))

    best: Optional[tuple[float, float, int, float, float, float, float]] = None
    phase0_grid = np.linspace(0.0, 2.0 * math.pi / 5.0, 5, endpoint=False)
    for shift in range(5):
        for sp in (1.0, -1.0):
            for sz in (1.0, -1.0):
                for phase0 in phase0_grid:
                    for z_scale in np.geomspace(0.06, 4.0, 24):
                        m_err = _mean_nu_err_for(z_scale, shift, sp, sz, float(phase0))
                        if best is None or m_err < best[0]:
                            best = (
                                m_err,
                                m_err,
                                shift,
                                sp,
                                sz,
                                float(phase0),
                                float(z_scale),
                            )
    if best is None:
        raise RuntimeError('planar sugar calibration failed (grid).')
    _, _, shift, sp, sz, phase0_sel, z_coarse = best
    z_lo = z_coarse * 0.75
    z_hi = z_coarse * 1.35
    z_fine = np.linspace(z_lo, z_hi, 24, dtype=np.float64)
    best_m = best[0]
    z_scale = z_coarse
    for z_try in z_fine:
        m_err = _mean_nu_err_for(float(z_try), shift, sp, sz, phase0_sel)
        if m_err < best_m:
            best_m = m_err
            z_scale = float(z_try)
    phases_out = np.roll(base_phases, shift) * sz + phase0_sel
    en_u = en.copy()
    z_test = z_scale * 0.5 * np.cos(sp * 0.0 + phases_out)
    pos_t = np.stack(
        [
            ctr + xy[j, 0] * e1 + xy[j, 1] * e2 + z_test[j] * en_u
            for j in range(5)
        ],
        axis=0,
    )
    tr = float(
        np.dot(
            np.cross(pos_t[3] - pos_t[1], pos_t[4] - pos_t[1]),
            pos_t[0] - pos_t[1],
        ),
    )
    if np.sign(tr) != np.sign(rch):
        en_u = -en_u
    return {
        'xy': xy.astype(np.float32),
        'center': ctr.astype(np.float32),
        'e1': e1.astype(np.float32),
        'e2': e2.astype(np.float32),
        'en': en_u.astype(np.float32),
        'phase_rad': phases_out.astype(np.float32),
        'z_scale': z_scale,
        'sign_P': float(sp),
    }


def _calibrate_planar_sugar_spec() -> dict[str, Any]:
    try:
        pts = _mean_sugar_ring_xyz_from_templates()
    except (ImportError, ModuleNotFoundError, OSError):
        pts = _fallback_mean_ring_xyz_acgt_order()
    return _calibrate_planar_sugar_from_pts(pts)


def _planar_sugar_spec() -> dict[str, Any]:
    global _PLANAR_SUGAR_SPEC
    if _PLANAR_SUGAR_SPEC is None:
        _PLANAR_SUGAR_SPEC = _calibrate_planar_sugar_spec()
    return _PLANAR_SUGAR_SPEC


@functools.lru_cache(maxsize=None)
def _pseudorotation_offsets_torch(
    device_str: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        _PSEUDOROTATION_OFFSETS,
        device=torch.device(device_str),
        dtype=dtype,
    )


@functools.lru_cache(maxsize=None)
def _scalar_constant_torch(
    value: float,
    device_str: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(value, device=torch.device(device_str), dtype=dtype)


@functools.lru_cache(maxsize=None)
def _phi_grid_torch(
    device_str: str,
    dtype: torch.dtype,
    n_grid: int,
) -> torch.Tensor:
    return torch.linspace(
        0.0,
        2.0 * math.pi,
        n_grid,
        device=torch.device(device_str),
        dtype=dtype,
    )


@functools.lru_cache(maxsize=None)
def _planar_sugar_spec_tensors(
    device_str: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    spec = _planar_sugar_spec()
    device = torch.device(device_str)
    return {
        'xy': torch.as_tensor(spec['xy'], device=device, dtype=dtype),
        'center': torch.as_tensor(spec['center'], device=device, dtype=dtype),
        'e1': torch.as_tensor(spec['e1'], device=device, dtype=dtype),
        'e2': torch.as_tensor(spec['e2'], device=device, dtype=dtype),
        'en': torch.as_tensor(spec['en'], device=device, dtype=dtype),
        'phase_rad': torch.as_tensor(spec['phase_rad'], device=device, dtype=dtype),
        'z_scale': float(spec['z_scale']),
        'sign_P': float(spec['sign_P']),
    }


def _template_select(
    tc: dict[str, torch.Tensor],
    key: str,
    restype_indices: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    out = tc[key][restype_indices.long()]
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype=dtype)
    return out


def sugar_ring_from_xy_z_torch(
    xy: torch.Tensor,
    z: torch.Tensor,
    center: torch.Tensor,
    e1: torch.Tensor,
    e2: torch.Tensor,
    en: torch.Tensor,
) -> torch.Tensor:
    """Build 3D ring atoms from in-plane ``xy`` and out-of-plane ``z``.

    Args:
        xy: ``[5, 2]`` for ``_SUGAR_RING_BUILD_ATOM_ORDER``.
        z: ``[..., 5]`` out-of-plane displacement per atom.
        center, e1, e2, en: ``[3]`` PCA frame in Å.

    Returns:
        ``[..., 5, 3]`` coordinates in the same order as ``_SUGAR_RING_BUILD_ATOM_ORDER``.
    """
    if xy.shape != (5, 2):
        raise ValueError(f'xy must be [5,2], got {tuple(xy.shape)}')
    base = (
        center
        + xy[:, 0].unsqueeze(-1) * e1
        + xy[:, 1].unsqueeze(-1) * e2
    )
    nd = z.ndim - 1
    base_exp = base.view((1,) * nd + (5, 3)).expand(z.shape[:-1] + (5, 3))
    z3 = z.unsqueeze(-1) * en
    return base_exp + z3


def build_sugar_ring_closed_form_torch(
    chi: torch.Tensor,
    P: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
    *,
    geometry: Optional[dict] = None,
    bond_lengths: Optional[dict] = None,
    bond_angles: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Closed-form furanose ring: NeRF from template reference frame + χ rotation.

    Builds the ring using exact template bond lengths and angles.  C3' and C4' are
    placed via NERF with torsions ν₄ and ν₀ from the Altona-Sundaralingam formula
    (ν_k = τ_m cos(P + phase_k)).  The ring closes to within the AS approximation
    residual (< 0.03 Å for typical RNA puckering).

    No grid scan, no discrete branch minimization, no cone–sphere sugar solver. Returns
    ``O4'``, ``C1'``, ``C2'``, ``C3'``, ``C4'``.
    """
    g = geometry or {}
    _ = bond_lengths, bond_angles
    if P.shape != tau_m.shape or P.shape != restype_indices.shape:
        raise ValueError('P, tau_m, restype_indices must share the same leading shape.')
    if P.ndim == 0:
        chi = chi.unsqueeze(0)
        P = P.unsqueeze(0)
        tau_m = tau_m.unsqueeze(0)
        restype_indices = restype_indices.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    orig = P.shape
    N = int(P.numel())
    dev = P.device
    dtype = P.dtype
    dev_str = str(dev)
    chi_f = chi.reshape(N)
    P_f = P.reshape(N)
    tm_f = tau_m.reshape(N)
    ri = restype_indices.reshape(N).long()
    tc = g.get('template_tensors')
    if tc is None:
        tc = _get_template_tensors(dev_str)

    def _g1(key: str) -> torch.Tensor:
        return _template_select(tc, key, ri, dtype=dtype)

    # PCA ring: gives P-appropriate 3D positions for all ring atoms.
    spec = g.get('planar_spec_tensors')
    if spec is None:
        spec = _planar_sugar_spec_tensors(dev_str, dtype)
    xy = cast(torch.Tensor, spec['xy'])
    ctr = cast(torch.Tensor, spec['center'])
    e1v = cast(torch.Tensor, spec['e1'])
    e2v = cast(torch.Tensor, spec['e2'])
    env = cast(torch.Tensor, spec['en'])
    phases = cast(torch.Tensor, spec['phase_rad'])
    z_sc = float(spec['z_scale'])
    sgn_p = float(spec['sign_P'])

    z = z_sc * tm_f.clamp(min=1e-4).unsqueeze(-1) * torch.cos(
        sgn_p * P_f.unsqueeze(-1) + phases.unsqueeze(0)
    )
    stacked = sugar_ring_from_xy_z_torch(xy, z, ctr, e1v, e2v, env)  # [N, 5, 3]
    # _SUGAR_RING_BUILD_ATOM_ORDER: C4'=0, O4'=1, C1'=2, C2'=3, C3'=4

    # Translate-only: move C1' to template position, preserve P-appropriate O4'/C2' directions.
    c1 = _g1('c1')
    transl = c1 - stacked[:, 2]
    bl_o4c1 = (_g1('c1') - _g1('o4')).norm(dim=-1)

    # Rescale O4' and C2' to exact template bond lengths, keeping P-dependent directions.
    o4 = c1 + bl_o4c1.unsqueeze(-1) * _safe_normalize(stacked[:, 1] + transl - c1)
    c2 = c1 + _g1('bl_c2_c1').unsqueeze(-1) * _safe_normalize(stacked[:, 3] + transl - c1)

    # ν₁ = dihedral(O4',C1',C2',C3'); ν₂ = dihedral(C1',C2',C3',C4') — AS formula.
    half_pi = _scalar_constant_torch(_HALF_PI, dev_str, dtype)
    nus = nus_rad_from_P_tau_torch(P_f, tm_f)  # [N, 5]

    # Place C3' via NeRF: exact bl_c3c2, exact ν₁ (index 4 in _RING_TORSION_DEFS order).
    c3 = nerf_place_torch(
        o4, c1, c2,
        _g1('bl_c3_c2'), _g1('ba_c1_c2_c3'),
        nus[:, 4] - half_pi,
    )

    # Place ideal C4' via NeRF with ν₂ (index 0); used only to initialise circle basis.
    ideal_c4 = nerf_place_torch(
        c1, c2, c3,
        _g1('bl_c4_c3'), _g1('ba_c2_c3_c4'),
        nus[:, 0] - half_pi,
    )
    bl_c4c3 = _g1('bl_c4_c3')   # [N]
    bl_o4c4 = _g1('bl_o4_c4')  # [N]

    # Sphere-sphere intersection: circle of C4' satisfying |C4'–C3'| = bl_c4c3 and |C4'–O4'| = bl_o4c4.
    c3o4 = o4 - c3
    d_sq = (c3o4 * c3o4).sum(dim=-1)
    d = d_sq.clamp(min=1e-8).sqrt()
    d_hat = c3o4 / d.unsqueeze(-1)
    h = (d_sq + bl_c4c3 ** 2 - bl_o4c4 ** 2) / (2.0 * d)  # foot distance from C3' to circle centre
    r_c = (bl_c4c3 ** 2 - h ** 2).clamp(min=0.0).sqrt()    # circle radius
    M = c3 + h.unsqueeze(-1) * d_hat                        # circle centre

    # Circle orthonormal basis from ideal_c4 direction (initial orientation only).
    vp = ideal_c4 - M
    vp = vp - (vp * d_hat).sum(dim=-1, keepdim=True) * d_hat
    e1 = _safe_normalize(vp)
    e2 = torch.linalg.cross(d_hat, e1, dim=-1)

    # Analytic φ for exact ν₂ = dihedral(C1',C2',C3',C4'(φ)) = nus[:,0] on the circle.
    # n2(φ) = cross(C3'−C2', h·d̂ + r_c·(cos·e1 + sin·e2))
    #       = h·A + r_c·(cos·B + sin·C)  where A,B,C below.
    bc32 = c3 - c2
    _A = torch.linalg.cross(bc32, d_hat, dim=-1)
    _B = torch.linalg.cross(bc32, e1, dim=-1)
    _C = torch.linalg.cross(bc32, e2, dim=-1)
    n12 = _safe_normalize(torch.linalg.cross(c1 - c2, bc32, dim=-1))
    m12 = torch.linalg.cross(n12, _safe_normalize(bc32), dim=-1)
    a0 = h * (n12 * _A).sum(dim=-1)
    a1 = h * (m12 * _A).sum(dim=-1)
    b0 = r_c * (n12 * _B).sum(dim=-1)
    b1 = r_c * (m12 * _B).sum(dim=-1)
    g0 = r_c * (n12 * _C).sum(dim=-1)
    g1 = r_c * (m12 * _C).sum(dim=-1)
    nu2_tgt = nus[:, 0]
    ct = torch.cos(nu2_tgt)
    st = torch.sin(nu2_tgt)
    Pc = ct * b1 - st * b0
    Qc = ct * g1 - st * g0
    Rc = ct * a1 - st * a0
    pq = (Pc ** 2 + Qc ** 2).clamp(min=1e-8).sqrt()
    phi_base = torch.atan2(Qc, Pc)
    phi_off = torch.acos((-Rc / pq).clamp(-1.0, 1.0))
    phi1 = phi_base + phi_off
    phi2 = phi_base - phi_off
    c4_a = M + r_c.unsqueeze(-1) * (torch.cos(phi1).unsqueeze(-1) * e1 + torch.sin(phi1).unsqueeze(-1) * e2)
    c4_b = M + r_c.unsqueeze(-1) * (torch.cos(phi2).unsqueeze(-1) * e1 + torch.sin(phi2).unsqueeze(-1) * e2)
    # Pick the solution closer to ideal_c4 (preserves ring chirality).
    closer = ((c4_a - ideal_c4).norm(dim=-1) <= (c4_b - ideal_c4).norm(dim=-1)).unsqueeze(-1)
    c4 = torch.where(closer, c4_a, c4_b)

    ring_pre = {"C1'": c1, "O4'": o4, "C2'": c2, "C3'": c3, "C4'": c4}
    n_atom = _g1('chi_n').reshape(N, 3)
    c_atom = _g1('chi_c').reshape(N, 3)
    out = _apply_chi_rotation_to_sugar_ring_torch(ring_pre, chi_f, n_atom, c_atom)
    for nm in ("C1'", "C2'", "C3'", "C4'", "O4'"):
        out[nm] = out[nm].reshape(*orig, 3)
    if squeeze_batch:
        out = {k: v.squeeze(0) for k, v in out.items()}
    return out


def add_exocyclic_sugar_atoms_torch(
    ring_atoms: dict[str, torch.Tensor],
    *,
    restype_indices: Optional[torch.Tensor] = None,
    geometry: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Add O3' and C5' using template stereochemistry (fixed ψ from template)."""
    g = geometry or {}
    ri = restype_indices if restype_indices is not None else ring_atoms.get('_ri')
    if ri is None:
        raise ValueError('Pass restype_indices= or ring_atoms["_ri"] (long [B]).')
    o4 = ring_atoms["O4'"]
    c3 = ring_atoms["C3'"]
    c4 = ring_atoms["C4'"]
    if o4.ndim == 1:
        o4 = o4.unsqueeze(0)
        c3 = c3.unsqueeze(0)
        c4 = c4.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    dev = o4.device
    dtype = o4.dtype
    dev_str = str(dev)
    ri = ri.long()
    tc = g.get('template_tensors')
    if tc is None:
        tc = _get_template_tensors(dev_str)
    half_pi = _scalar_constant_torch(_HALF_PI, dev_str, dtype)

    c5 = nerf_place_torch(
        o4, c3, c4,
        _template_select(tc, 'bl_c5_c4', ri, dtype=dtype),
        _template_select(tc, 'ba_c3_c4_c5', ri, dtype=dtype),
        _template_select(tc, 'psi_c5', ri, dtype=dtype) - half_pi,
    )
    o3 = nerf_place_torch(
        o4, c4, c3,
        _template_select(tc, 'bl_o3_c3', ri, dtype=dtype),
        _template_select(tc, 'ba_c4_c3_o3', ri, dtype=dtype),
        _template_select(tc, 'psi_o3_ring', ri, dtype=dtype) - half_pi,
    )

    out = dict(ring_atoms)
    if squeeze:
        out["C5'"] = c5.squeeze(0)
        out["O3'"] = o3.squeeze(0)
    else:
        out["C5'"] = c5
        out["O3'"] = o3
    return out


def add_o5_from_gamma_torch(
    atoms: dict[str, torch.Tensor],
    gamma: torch.Tensor,
    *,
    restype_indices: Optional[torch.Tensor] = None,
    geometry: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Place O5' using γ = O5'–C5'–C4'–C3' (NeRF chain, same bond/angle prior as main decoder)."""
    gctx = geometry or {}
    c3 = atoms["C3'"]
    c4 = atoms["C4'"]
    c5 = atoms["C5'"]
    g = gamma
    if c3.ndim == 1:
        c3 = c3.unsqueeze(0)
        c4 = c4.unsqueeze(0)
        c5 = c5.unsqueeze(0)
        g = g.reshape(1)
        sq = True
    else:
        sq = False
    ri = restype_indices if restype_indices is not None else atoms.get('_ri')
    if ri is None:
        raise ValueError('Pass restype_indices= or atoms["_ri"].')
    ri = ri.long()
    dev = c3.device
    dtype = c3.dtype
    dev_str = str(dev)
    tc = gctx.get('template_tensors')
    if tc is None:
        tc = _get_template_tensors(dev_str)
    half_pi = _scalar_constant_torch(_HALF_PI, dev_str, dtype)

    o5 = nerf_place_torch(
        c3,
        c4,
        c5,
        _template_select(tc, 'bl_o5_c5', ri, dtype=dtype),
        _template_select(tc, 'ba_c4_c5_o5', ri, dtype=dtype),
        g - half_pi,
    )
    out = dict(atoms)
    if sq:
        out["O5'"] = o5.squeeze(0)
    else:
        out["O5'"] = o5
    return out


def close_phosphate_bridge_multi_torch(
    prev_atoms: dict[str, torch.Tensor],
    next_atoms: dict[str, torch.Tensor],
    epsilon_prev: torch.Tensor,
    zeta_prev: torch.Tensor,
    alpha_next: torch.Tensor,
    beta_next: torch.Tensor,
    *,
    geometry: Optional[dict] = None,
    weight_epsilon: torch.Tensor | float = 1.0,
    weight_zeta: torch.Tensor | float = 1.0,
    weight_alpha: torch.Tensor | float = 1.0,
    weight_beta: torch.Tensor | float = 1.0,
) -> dict[str, torch.Tensor]:
    """Place P on the O3'prev–O5'next circle; scan φ to match up to four target dihedrals."""
    g = geometry or {}
    dev = epsilon_prev.device
    dtype = epsilon_prev.dtype
    dev_str = str(dev)

    def _broadcast_bridge_weight(w, batch_sz: int) -> torch.Tensor:
        """Scalar or per-batch weight column [B, 1] for loss grid broadcast."""
        import torch as _t
        if isinstance(w, _t.Tensor):
            t = w.reshape(-1).to(device=dev, dtype=dtype)
            if t.numel() == 1:
                t = t.expand(batch_sz)
            elif t.numel() != batch_sz:
                raise ValueError(
                    f'Bridge weight tensor must have 1 or B={batch_sz} elements, got {t.numel()}',
                )
            return t.reshape(batch_sz, 1)
        return _t.full((batch_sz, 1), float(w), device=dev, dtype=dtype)
    o3p = prev_atoms["O3'"]
    c3p = prev_atoms["C3'"]
    c4p = prev_atoms["C4'"]
    o5n = next_atoms["O5'"]
    c5n = next_atoms["C5'"]
    c4n = next_atoms["C4'"]

    ri = g.get('restype_indices_next')
    if ri is None:
        ri = next_atoms.get('_ri')
    if ri is None:
        raise ValueError('close_phosphate_bridge_multi_torch needs restype_indices (next) in geometry or _ri on next_atoms.')
    ri = ri.long()
    tc = g.get('template_tensors')
    if tc is None:
        tc = _get_template_tensors(dev_str)
    r_o3p = _template_select(tc, 'bond_p_o3_inter', ri, dtype=dtype)
    l_po5 = _template_select(tc, 'bond_p_o5', ri, dtype=dtype)

    d_vec = o5n - o3p
    d_o3_o5 = d_vec.norm(dim=-1).clamp(min=_GEO_EPS)
    axis, v1, v2 = _orthonormal_basis_from_axis(d_vec, eps=_GEO_EPS)

    sum_l = r_o3p + l_po5
    diff_l = (r_o3p - l_po5).abs()
    feasible = (d_o3_o5 >= diff_l - 1e-6) & (d_o3_o5 <= sum_l + 1e-6)

    a_coef = (r_o3p * r_o3p - l_po5 * l_po5 + d_o3_o5 * d_o3_o5) / (2.0 * d_o3_o5)
    h_sq = (r_o3p * r_o3p - a_coef * a_coef).clamp(min=0.0)
    h_val = torch.sqrt(h_sq)

    midpoint = o3p + a_coef.unsqueeze(-1) * axis

    n_grid = int(g.get('n_phi', _PHI_PHOS_GRID))
    ts = _phi_grid_torch(dev_str, dtype, n_grid)
    if o3p.ndim >= 2:
        B = int(o3p.shape[0])
    else:
        B = 1
        o3p = o3p.unsqueeze(0)
        c3p = c3p.unsqueeze(0)
        c4p = c4p.unsqueeze(0)
        o5n = o5n.unsqueeze(0)
        c5n = c5n.unsqueeze(0)
        c4n = c4n.unsqueeze(0)
        epsilon_prev = epsilon_prev.reshape(1)
        zeta_prev = zeta_prev.reshape(1)
        alpha_next = alpha_next.reshape(1)
        beta_next = beta_next.reshape(1)
        r_o3p = r_o3p.reshape(1)
        l_po5 = l_po5.reshape(1)
        d_o3_o5 = d_o3_o5.reshape(1)
        feasible = feasible.reshape(1)
        a_coef = a_coef.reshape(1)
        h_val = h_val.reshape(1)
        midpoint = midpoint.unsqueeze(0)
        axis = axis.unsqueeze(0)
        v1 = v1.unsqueeze(0)
        v2 = v2.unsqueeze(0)
        ri = ri.reshape(1)

    B = int(o3p.shape[0])
    p_cands = (
        midpoint.unsqueeze(1)
        + h_val.unsqueeze(1).unsqueeze(-1)
        * (
            torch.cos(ts).view(1, -1, 1) * v1.unsqueeze(1)
            + torch.sin(ts).view(1, -1, 1) * v2.unsqueeze(1)
        )
    )

    c4p_g = c4p.unsqueeze(1).expand(-1, n_grid, -1)
    c3p_g = c3p.unsqueeze(1).expand(-1, n_grid, -1)
    o3p_g = o3p.unsqueeze(1).expand(-1, n_grid, -1)
    o5n_g = o5n.unsqueeze(1).expand(-1, n_grid, -1)
    c5n_g = c5n.unsqueeze(1).expand(-1, n_grid, -1)
    c4n_g = c4n.unsqueeze(1).expand(-1, n_grid, -1)
    eps_m = dihedral_rad_torch(
        c4p_g.reshape(-1, 3),
        c3p_g.reshape(-1, 3),
        o3p_g.reshape(-1, 3),
        p_cands.reshape(-1, 3),
    ).reshape(B, n_grid)

    ze_m = dihedral_rad_torch(
        c3p_g.reshape(-1, 3),
        o3p_g.reshape(-1, 3),
        p_cands.reshape(-1, 3),
        o5n_g.reshape(-1, 3),
    ).reshape(B, n_grid)

    al_m = dihedral_rad_torch(
        o3p_g.reshape(-1, 3),
        p_cands.reshape(-1, 3),
        o5n_g.reshape(-1, 3),
        c5n_g.reshape(-1, 3),
    ).reshape(B, n_grid)

    be_m = dihedral_rad_torch(
        p_cands.reshape(-1, 3),
        o5n_g.reshape(-1, 3),
        c5n_g.reshape(-1, 3),
        c4n_g.reshape(-1, 3),
    ).reshape(B, n_grid)

    we = _broadcast_bridge_weight(weight_epsilon, B)
    wz = _broadcast_bridge_weight(weight_zeta, B)
    wa = _broadcast_bridge_weight(weight_alpha, B)
    wb = _broadcast_bridge_weight(weight_beta, B)

    loss = (
        we * wrap_dihedral_diff_torch(eps_m, epsilon_prev.unsqueeze(-1)) ** 2
        + wz * wrap_dihedral_diff_torch(ze_m, zeta_prev.unsqueeze(-1)) ** 2
        + wa * wrap_dihedral_diff_torch(al_m, alpha_next.unsqueeze(-1)) ** 2
        + wb * wrap_dihedral_diff_torch(be_m, beta_next.unsqueeze(-1)) ** 2
    )

    best = loss.argmin(dim=-1)
    idx = torch.arange(B, device=dev, dtype=torch.long)
    p_pick = p_cands[idx, best]

    # Infeasible: midpoint along axis (deterministic fallback)
    not_feas = ~feasible
    if not_feas.any():
        sum_l = (r_o3p + l_po5).unsqueeze(-1).clamp(min=_GEO_EPS)
        fb = o3p + (o5n - o3p) * (r_o3p.unsqueeze(-1) / sum_l)
        p_pick = torch.where(not_feas.unsqueeze(-1), fb, p_pick)

    half_pi = _scalar_constant_torch(_HALF_PI, dev_str, dtype)
    bl_op1 = _template_select(tc, 'bl_op1', ri, dtype=dtype)
    bl_op2 = _template_select(tc, 'bl_op2', ri, dtype=dtype)
    ang_op1 = _template_select(tc, 'ang_op1', ri, dtype=dtype)
    ang_op2 = _template_select(tc, 'ang_op2', ri, dtype=dtype)
    psi_op1 = _template_select(tc, 'psi_op1', ri, dtype=dtype)
    psi_op2 = _template_select(tc, 'psi_op2', ri, dtype=dtype)

    op1 = nerf_place_torch(o3p, o5n, p_pick, bl_op1, ang_op1, psi_op1 - half_pi)
    op2 = nerf_place_torch(o3p, o5n, p_pick, bl_op2, ang_op2, psi_op2 - half_pi)

    out = {'P': p_pick, 'OP1': op1, 'OP2': op2}
    if B == 1 and epsilon_prev.ndim == 0:
        out = {k: v.squeeze(0) for k, v in out.items()}
    return out


def build_backbone_from_torsions(
    torsions: np.ndarray,
    restype: str,
    o3_prev_local: Optional[np.ndarray] = None,
    tau_m: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Build backbone atom positions in the local nucleotide frame from seven torsion angles.

    Parameters
    ----------
    torsions : np.ndarray, shape [N_TORSIONS]
        Order: [α, β, γ, ε, ζ, χ, P], radians.
    restype : str
        One-letter nucleotide code: 'A' | 'C' | 'G' | 'T'.
    o3_prev_local : optional (3,)
        Previous nucleotide's O3' in this residue's local frame; enables α-based P placement.
    tau_m : optional float
        Predicted puckering amplitude (rad); fallback: template amplitude.

    Returns
    -------
    dict[str, np.ndarray]
        atom_name -> xyz (Å) in the local nucleotide frame.
    """
    tpl = _get_template(restype)
    ri = {'A': 0, 'C': 1, 'G': 2, 'T': 3}[restype]
    tc = _get_template_tensors('cpu')

    # Extract relevant torsion angles
    alpha = float(torsions[TOR_ALPHA])
    beta = float(torsions[TOR_BETA])
    gamma = float(torsions[TOR_GAMMA])
    P_rad = float(torsions[TOR_PUCKER_P])
    if tau_m is not None:
        _TAU_DNA = float(tau_m)
    else:
        _TAU_DNA = float(_template_tau_m.get(restype, 0.611))

    nus = _pseudorotation_nus_numpy(P_rad, _TAU_DNA)
    nu0, _nu1, _nu2, nu3, nu4 = nus

    # Anchor atoms — fixed in local frame from canonical template
    c1 = tpl["C1'"].copy()
    o4 = tpl["O4'"].copy()
    c4_ref = tpl["C4'"].copy()

    # nerf_place(A, B, C, r, θ, ψ) satisfies dihedral(A,B,C,D) = ψ + π/2
    _HP = _HALF_PI

    # ── Sugar ring ──────────────────────────────────────────────────────────
    c2 = nerf_place(
        c4_ref, o4, c1,
        float(tc['bl_c2_c1'][ri].item()),
        _bond_angle_np(o4, c1, tpl["C2'"]),
        nu3 - _HP,
    )
    c3 = nerf_place(
        o4, c1, c2,
        float(tc['bl_c3_c2'][ri].item()),
        float(tc['ba_c1_c2_c3'][ri].item()),
        nu4 - _HP,
    )
    c4 = nerf_place(
        c1, c2, c3,
        float(tc['bl_c4_c3'][ri].item()),
        float(tc['ba_c2_c3_c4'][ri].item()),
        nu0 - _HP,
    )

    chi_tgt = float(torsions[TOR_CHI])
    _o4n, _c1n, n_nm, c_nm = _chi_quads(restype)
    _ = _o4n, _c1n
    n_xyz = np.asarray(tpl[n_nm], dtype=np.float64).reshape(3)
    c_xyz = np.asarray(tpl[c_nm], dtype=np.float64).reshape(3)
    chi_meas = dihedral_rad(o4, c1, n_xyz, c_xyz)
    dchi = float(wrap_angle_rad(chi_tgt - chi_meas))
    ax = n_xyz - c1
    la = float(np.linalg.norm(ax))
    if la > 1e-10:
        u_ax = ax / la

        def _rot_vec(v: np.ndarray) -> np.ndarray:
            vv = np.asarray(v, dtype=np.float64).reshape(3) - c1
            co = np.cos(dchi)
            si = np.sin(dchi)
            vc = np.cross(u_ax, vv)
            dp = float(np.dot(u_ax, vv))
            return (vv * co + vc * si + u_ax * dp * (1.0 - co)) + c1

        o4 = _rot_vec(o4)
        c2 = _rot_vec(c2)
        c3 = _rot_vec(c3)
        c4 = _rot_vec(c4)

    psi_c5 = float(tc['psi_c5'][ri].item())
    c5 = nerf_place(
        o4, c3, c4,
        float(tc['bl_c5_c4'][ri].item()),
        float(tc['ba_c3_c4_c5'][ri].item()),
        psi_c5 - _HP,
    )

    o5_gamma = nerf_place(
        c3, c4, c5,
        float(tc['bl_o5_c5'][ri].item()),
        float(tc['ba_c4_c5_o5'][ri].item()),
        gamma - _HP,
    )

    psi_o3 = float(tc['psi_o3_ring'][ri].item())
    o3 = nerf_place(
        o4, c4, c3,
        float(tc['bl_o3_c3'][ri].item()),
        float(tc['ba_c4_c3_o3'][ri].item()),
        psi_o3 - _HP,
    )

    out: dict[str, np.ndarray] = {
        "C1'": c1.copy(),
        "C2'": c2,
        "C3'": c3,
        "C4'": c4,
        "C5'": c5,
        "O4'": o4.copy(),
        "O3'": o3,
    }

    p_built: Optional[np.ndarray] = None
    o3p: Optional[np.ndarray] = None
    o5 = o5_gamma.copy()

    if o3_prev_local is not None:
        o3p = np.asarray(o3_prev_local, dtype=np.float64).reshape(3)
        # O3'_i–P_{i+1}: use phosphodiester target (Å), same canonical value as `_get_template_tensors` ``bond_p_o3_inter``.
        l_po5 = float(tc['bond_p_o5'][ri].item())
        r_o3p_bridge = float(tc['bond_p_o3_inter'][ri].item())
        theta_po3 = np.deg2rad(119.0)
        d_o3_o5 = float(np.linalg.norm(o3p - o5_gamma))
        if abs(r_o3p_bridge - l_po5) <= d_o3_o5 <= r_o3p_bridge + l_po5:
            p_built = close_phosphate_bridge(
                o3p, o5_gamma, c5, r_o3p_bridge, l_po5, alpha,
            )
        else:
            # ψ = −α aligns dihedral(O3′, P, O5′, C5′) with α given nerf_place's +π/2 offset.
            p_built = nerf_place(
                c5, o5_gamma, o3p, r_o3p_bridge, theta_po3, -alpha - _HP,
            )
        o5 = o5_gamma

    out["O5'"] = o5

    if p_built is not None and o3p is not None:
        o5_f = out["O5'"]
        r_p_op1 = float(tc['bl_op1'][ri].item())
        r_p_op2 = float(tc['bl_op2'][ri].item())
        ang_op1 = float(tc['ang_op1'][ri].item())
        ang_op2 = float(tc['ang_op2'][ri].item())
        psi_op1 = float(tc['psi_op1'][ri].item())
        psi_op2 = float(tc['psi_op2'][ri].item())
        op1 = nerf_place(o3p, o5_f, p_built, r_p_op1, ang_op1, psi_op1 - _HP)
        op2 = nerf_place(o3p, o5_f, p_built, r_p_op2, ang_op2, psi_op2 - _HP)
        out['P'] = p_built
        out['OP1'] = op1
        out['OP2'] = op2
    else:
        for nm in ('P', 'OP1', 'OP2'):
            if nm in tpl:
                out[nm] = tpl[nm].copy()

    return out


_BACKBONE_ATOM_ORDER = (
    "C1'", "C2'", "C3'", "C4'", "C5'", "OP1", "OP2", "P", "O3'", "O4'", "O5'",
)
_BACKBONE_NAME_TO_INDEX = {nm: j for j, nm in enumerate(_BACKBONE_ATOM_ORDER)}
_LOCAL_BACKBONE_ATOM_ORDER = ("C1'", "C2'", "C3'", "C4'", "C5'", "O4'", "O3'", "O5'")
_LOCAL_BACKBONE_INDEX = tuple(_BACKBONE_NAME_TO_INDEX[nm] for nm in _LOCAL_BACKBONE_ATOM_ORDER)
_BRIDGE_PREV_ATOM_ORDER = ("O3'", "C3'", "C4'")
_BRIDGE_NEXT_ATOM_ORDER = ("O5'", "C5'", "C4'")
_PHOSPHATE_ATOM_ORDER = ('P', 'OP1', 'OP2')
_PHOSPHATE_ATOM_INDEX = tuple(_BACKBONE_NAME_TO_INDEX[nm] for nm in _PHOSPHATE_ATOM_ORDER)


def build_backbone_from_torsions_torch(
    torsions: 'torch.Tensor',
    tau_m: 'torch.Tensor',
    restype_indices: 'torch.Tensor',
    o3_prev_local: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Batched decoder: deterministic sugar ring closure, stereo exocyclic O3′/C5′, γ→O5′.

    Phosphate uses ``close_phosphate_bridge_multi_torch``; without prior-nucleotide ε/ζ in this API,
    only α and β targets are weighted (incoming bridge from o3_prev in target frame).
    """
    import torch

    ri = restype_indices.long()
    geometry = {'template_tensors': _get_template_tensors(str(torsions.device))}
    chi = torsions[:, TOR_CHI]
    P = torsions[:, TOR_PUCKER_P]
    gamma = torsions[:, TOR_GAMMA]
    alpha = torsions[:, TOR_ALPHA]
    beta = torsions[:, TOR_BETA]
    eps_lat = torsions[:, TOR_EPS]
    zet_lat = torsions[:, TOR_ZETA]
    ring = build_sugar_ring_closed_form_torch(chi, P, tau_m, ri, geometry=geometry)
    atoms = add_exocyclic_sugar_atoms_torch(ring, restype_indices=ri, geometry=geometry)
    atoms = add_o5_from_gamma_torch(atoms, gamma, restype_indices=ri, geometry=geometry)

    out = {
        "C1'": atoms["C1'"],
        "C2'": atoms["C2'"],
        "C3'": atoms["C3'"],
        "C4'": atoms["C4'"],
        "C5'": atoms["C5'"],
        "O4'": atoms["O4'"],
        "O3'": atoms["O3'"],
        "O5'": atoms["O5'"],
    }

    if o3_prev_local is None:
        return out

    prev_atoms = {
        "O3'": o3_prev_local,
        "C3'": out["C3'"],
        "C4'": out["C4'"],
    }
    next_atoms: dict[str, torch.Tensor] = {
        "O5'": out["O5'"],
        "C5'": out["C5'"],
        "C4'": out["C4'"],
        "_ri": ri,
    }
    phosph = close_phosphate_bridge_multi_torch(
        prev_atoms,
        next_atoms,
        eps_lat,
        zet_lat,
        alpha,
        beta,
        geometry={
            'restype_indices_next': ri,
            'template_tensors': geometry['template_tensors'],
        },
        weight_epsilon=0.0,
        weight_zeta=0.0,
        weight_alpha=1.0,
        weight_beta=1.0,
    )
    out['P'] = phosph['P']
    out['OP1'] = phosph['OP1']
    out['OP2'] = phosph['OP2']
    return out


def build_batch_window_backbone_from_torsions_torch(
    torsions: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
    nt_origins_world: torch.Tensor,
    nt_frames_world: torch.Tensor,
    torsion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """World backbone ``[B, W, n_bb, 3]``; atom order matches ``_BACKBONE_ATOM_ORDER``.

    For each window position ``k>=1`` and batch row ``b``, place ``P_k``/``OP*`` using ε,ζ from
    residue ``k−1`` and α,β from ``k`` (per-sample mask when ``torsion_mask`` is set).
    """
    import torch

    if torsions.dim() != 3 or torsions.shape[-1] != N_TORSIONS:
        raise ValueError(
            f'Expected torsions [B,W,{N_TORSIONS}], got {tuple(torsions.shape)}',
        )
    B, W, _ = torsions.shape
    dev = torsions.device
    dtype = torsions.dtype
    n_bb = len(_BACKBONE_ATOM_ORDER)
    BW = B * W
    phosphate_index = torch.tensor(_PHOSPHATE_ATOM_INDEX, device=dev, dtype=torch.long)

    flat_t = torsions.reshape(BW, N_TORSIONS)
    flat_tm = tau_m.reshape(BW)
    flat_ri = restype_indices.reshape(BW).long()
    geometry = {
        'template_tensors': _get_template_tensors(str(dev)),
        'planar_spec_tensors': _planar_sugar_spec_tensors(str(dev), dtype),
    }
    chi = flat_t[:, TOR_CHI]
    P_rad = flat_t[:, TOR_PUCKER_P]
    ring = build_sugar_ring_closed_form_torch(chi, P_rad, flat_tm, flat_ri, geometry=geometry)
    atoms = add_exocyclic_sugar_atoms_torch(ring, restype_indices=flat_ri, geometry=geometry)
    atoms = add_o5_from_gamma_torch(
        atoms, flat_t[:, TOR_GAMMA], restype_indices=flat_ri, geometry=geometry,
    )

    Ri_flat = nt_frames_world.reshape(BW, 3, 3)
    ok_flat = nt_origins_world.reshape(BW, 3)

    bb_flat = torch.full((BW, n_bb, 3), float('nan'), device=dev, dtype=dtype)
    local_atoms = torch.stack([atoms[nm] for nm in _LOCAL_BACKBONE_ATOM_ORDER], dim=1)
    bb_flat[:, _LOCAL_BACKBONE_INDEX] = local_to_world_points(local_atoms, ok_flat, Ri_flat)

    bb = bb_flat.view(B, W, n_bb, 3)

    for k in range(1, W):
        exec_ok = (
            torch.isfinite(bb[:, k - 1, _BACKBONE_NAME_TO_INDEX["O3'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k - 1, _BACKBONE_NAME_TO_INDEX["C3'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k - 1, _BACKBONE_NAME_TO_INDEX["C4'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k, _BACKBONE_NAME_TO_INDEX["O5'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k, _BACKBONE_NAME_TO_INDEX["C5'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k, _BACKBONE_NAME_TO_INDEX["C4'"]]).all(dim=-1)
        )
        if not bool(exec_ok.any().item()):
            continue
        idx = torch.nonzero(exec_ok, as_tuple=False).squeeze(-1)
        ok = nt_origins_world[idx, k]
        Rk = nt_frames_world[idx, k]

        prev_world = torch.stack(
            [bb[idx, k - 1, _BACKBONE_NAME_TO_INDEX[nm]] for nm in _BRIDGE_PREV_ATOM_ORDER],
            dim=1,
        )
        next_world = torch.stack(
            [bb[idx, k, _BACKBONE_NAME_TO_INDEX[nm]] for nm in _BRIDGE_NEXT_ATOM_ORDER],
            dim=1,
        )
        prev_local = world_to_local_points(prev_world, ok, Rk)
        next_local = world_to_local_points(next_world, ok, Rk)
        prev_loc = {
            nm: prev_local[:, atom_idx]
            for atom_idx, nm in enumerate(_BRIDGE_PREV_ATOM_ORDER)
        }
        next_loc = {
            nm: next_local[:, atom_idx]
            for atom_idx, nm in enumerate(_BRIDGE_NEXT_ATOM_ORDER)
        }
        next_loc['_ri'] = restype_indices[idx, k]

        nv = int(idx.shape[0])
        if torsion_mask is None:
            we = torch.ones(nv, device=dev, dtype=dtype)
            wz = torch.ones(nv, device=dev, dtype=dtype)
            wa = torch.ones(nv, device=dev, dtype=dtype)
            wb = torch.ones(nv, device=dev, dtype=dtype)
        else:
            we = torsion_mask[idx, k - 1, TOR_EPS].to(dtype=dtype)
            wz = torsion_mask[idx, k - 1, TOR_ZETA].to(dtype=dtype)
            wa = torsion_mask[idx, k, TOR_ALPHA].to(dtype=dtype)
            wb = torsion_mask[idx, k, TOR_BETA].to(dtype=dtype)

        phosph = close_phosphate_bridge_multi_torch(
            prev_loc,
            next_loc,
            torsions[idx, k - 1, TOR_EPS],
            torsions[idx, k - 1, TOR_ZETA],
            torsions[idx, k, TOR_ALPHA],
            torsions[idx, k, TOR_BETA],
            geometry={
                'restype_indices_next': restype_indices[idx, k],
                'template_tensors': geometry['template_tensors'],
            },
            weight_epsilon=we,
            weight_zeta=wz,
            weight_alpha=wa,
            weight_beta=wb,
        )
        phosph_local = torch.stack([phosph[nm] for nm in _PHOSPHATE_ATOM_ORDER], dim=1)
        bb[idx.unsqueeze(-1), k, phosphate_index] = local_to_world_points(phosph_local, ok, Rk)

    return bb


def build_window_backbone_from_torsions_torch(
    torsions: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
    nt_origins_world: torch.Tensor,
    nt_frames_world: torch.Tensor,
    torsion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """World backbone [W, n_bb, 3] with the same atom order as ``bbregen.schema.BACKBONE_ATOMS``.

    For each k>=1, place P_k / OP1 / OP2 using ε,ζ from residue k−1 and α,β from k (masked).
    Bridge geometry uses k−1 world O3′,C3′,C4′ expressed in nucleotide k local frame.
    """
    W = int(torsions.shape[0])
    Ri = nt_frames_world
    if Ri.dim() == 2:
        Ri = Ri.view(W, 3, 3)
    bb_b = build_batch_window_backbone_from_torsions_torch(
        torsions.unsqueeze(0),
        tau_m.unsqueeze(0),
        restype_indices.unsqueeze(0),
        nt_origins_world.unsqueeze(0),
        Ri.unsqueeze(0),
        torsion_mask.unsqueeze(0) if torsion_mask is not None else None,
    )
    return bb_b.squeeze(0)


def build_chain_backbone_from_predictions(
    theta: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
    nt_origins_world: torch.Tensor,
    nt_frames_world: torch.Tensor,
    torsion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """World backbone from wrapped torsions + τ_m for a single window/chain slice (training builder).

    Shapes: ``theta`` is ``[N, N_TORSIONS]`` or ``[1, N, N_TORSIONS]``; ``tau_m`` matches the
    leading structure (``[N]`` or ``[1, N]``). Returns ``[N, n_bb, 3]`` (atom order =
    ``bbregen.schema.BACKBONE_ATOMS`` / ``_BACKBONE_ATOM_ORDER``). For B>1 batches use
    ``build_batch_window_backbone_from_torsions_torch`` directly.
    """
    if theta.dim() == 2:
        if tau_m.dim() != 1 or restype_indices.dim() != 1:
            raise ValueError(
                f'Expected tau_m [N], restype [N] with theta [N,7]; '
                f'got tau {tuple(tau_m.shape)}, ri {tuple(restype_indices.shape)}',
            )
        if nt_origins_world.shape[0] != theta.shape[0] or nt_frames_world.shape[0] != theta.shape[0]:
            raise ValueError('nt_origins_world / nt_frames_world must have length N matching theta')
        return build_window_backbone_from_torsions_torch(
            theta,
            tau_m,
            restype_indices,
            nt_origins_world,
            nt_frames_world,
            torsion_mask,
        )
    if theta.dim() == 3:
        if theta.shape[0] != 1:
            raise ValueError(
                'build_chain_backbone_from_predictions supports batch size 1 only; '
                'use build_batch_window_backbone_from_torsions_torch for B>1',
            )
        tm = tau_m.squeeze(0) if tau_m.dim() == 2 else tau_m
        ri = restype_indices.squeeze(0) if restype_indices.dim() == 2 else restype_indices
        oo = nt_origins_world.squeeze(0) if nt_origins_world.dim() == 3 else nt_origins_world
        rr = nt_frames_world.squeeze(0) if nt_frames_world.dim() == 4 else nt_frames_world
        m = torsion_mask.squeeze(0) if torsion_mask is not None and torsion_mask.dim() == 3 else torsion_mask
        return build_window_backbone_from_torsions_torch(
            theta.squeeze(0), tm, ri, oo, rr, m,
        )
    raise ValueError(f'theta must be [N, {N_TORSIONS}] or [1, N, {N_TORSIONS}], got {tuple(theta.shape)}')
