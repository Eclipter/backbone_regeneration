"""Torsion definitions, sugar puckering (Altona–Sundaralingam / MDAnalysis-compatible), and angle wrapping."""

from __future__ import annotations

import functools
from typing import Optional

import numpy as np
import torch

N_TORSIONS = 7
# 0:α 1:β 2:γ 3:ε 4:ζ 5:χ 6:P   (backbone δ removed from generative set)

# True = circular (S¹); used for circular-aware diffusion
TORSION_IS_CIRCULAR = torch.tensor(
    [True, True, True, True, True, True, True],
    dtype=torch.bool,
)  # α β γ ε ζ χ P
TOR_ALPHA = 0
TOR_BETA = 1
TOR_GAMMA = 2
TOR_EPS = 3
TOR_ZETA = 4
TOR_CHI = 5
TOR_PUCKER_P = 6

N_TORSIONS_LATENT = N_TORSIONS + 1  # 8: 7 angles + log τ_m


def legacy_torsion8_to_new7(t_old: np.ndarray) -> np.ndarray:
    """Convert legacy 8-vector (…,δ,…) to 7-vector by dropping index 3 (δ)."""
    t_old = np.asarray(t_old)
    if t_old.shape[-1] == N_TORSIONS:
        return t_old
    if t_old.shape[-1] != 8:
        raise ValueError(
            f'Expected last dim 7 or 8, got {t_old.shape[-1]}',
        )
    return np.concatenate([t_old[..., :3], t_old[..., 4:]], axis=-1)


def wrap_angle_rad(x):
    """Map angles to (-π, π]."""
    return np.arctan2(np.sin(x), np.cos(x))


def nerf_place(a, b, c, r, theta, psi):
    """Place D given prior atoms A–B–C, bond C–D length r, interior angle ∠(B–C–D)=theta, dihedral(A,B,C,D)=psi (rad)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    c = np.asarray(c, dtype=np.float64).reshape(3)
    ba = a - b
    bc = c - b
    bc_u = bc / (np.linalg.norm(bc) + 1e-12)
    n = np.cross(ba, bc_u)
    nn = np.linalg.norm(n)
    if nn < 1e-10:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        n = n / nn
    m = np.cross(n, bc_u)
    return c + r * (
        np.cos(np.pi - theta) * bc_u
        + np.sin(np.pi - theta) * (np.cos(psi) * n + np.sin(psi) * m)
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
    bc_u = bc / (bc.norm(dim=-1, keepdim=True) + 1e-12)
    n = torch.linalg.cross(ba, bc_u)
    nn = n.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(n)
    fallback[..., 0] = 1.0
    n = torch.where(nn < 1e-10, fallback, n / (nn + 1e-12))
    m = torch.linalg.cross(n, bc_u)
    r_ = r.unsqueeze(-1)
    return c + r_ * (
        torch.cos(torch.pi - theta).unsqueeze(-1) * bc_u
        + torch.sin(torch.pi - theta).unsqueeze(-1) * (
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
    n1_u = n1 / (np.linalg.norm(n1) + 1e-12)
    n2_u = n2 / (np.linalg.norm(n2) + 1e-12)
    m1 = np.cross(n1_u, b2 / (np.linalg.norm(b2) + 1e-12))
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
        delta = o5 - o3
        dn = float(np.linalg.norm(delta))
        if dn < eps:
            return o3.copy()
        return o3 + delta * (r_o3p / sum_l)
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
_RING_ANGLES = 2.0 * 2.0 * np.pi * np.arange(5, dtype=np.float64) / 5.0


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
    phases = P_rad + (2.0 * np.pi * idx / 5.0)
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
        va, vc = tpl[a] - tpl[b], tpl[c] - tpl[b]
        cos_t = np.dot(va, vc) / (
            np.linalg.norm(va) * np.linalg.norm(vc) + 1e-12
        )
        return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))

    def _dr(tpl, a, b, c, d):
        return float(dihedral_rad(tpl[a], tpl[b], tpl[c], tpl[d]))

    keys_3d = ['c1', 'o4', 'c4_ref']
    keys_1d = [
        'bl_c2_c1', 'ba_o4_c1_c2',
        'bl_c3_c2', 'ba_c1_c2_c3',
        'bl_c4_c3', 'ba_c2_c3_c4',
        'bl_c5_c4', 'ba_c3_c4_c5', 'psi_c5',
        'bl_o5_c5', 'ba_c4_c5_o5',
        'bl_o3_c3', 'ba_c4_c3_o3',
        'r_po3',
        'bl_o3_p',
        'bl_op1', 'bl_op2',
        'ang_op1', 'ang_op2',
        'psi_op1', 'psi_op2',
        'psi_o3', 'bl_o4_c4', 'ring_chiral_triple',
    ]

    rows_3d = {k: [] for k in keys_3d}
    rows_1d = {k: [] for k in keys_1d}

    for rt in restypes:
        tpl = _get_template(rt)
        rows_3d['c1'].append(tpl["C1'"])
        rows_3d['o4'].append(tpl["O4'"])
        rows_3d['c4_ref'].append(tpl["C4'"])

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
        rows_1d['r_po3'].append(_bl(tpl, 'P', "O5'"))
        rows_1d['bl_o3_p'].append(_bl(tpl, "O3'", 'P'))
        rows_1d['bl_op1'].append(_bl(tpl, 'P', 'OP1'))
        rows_1d['bl_op2'].append(_bl(tpl, 'P', 'OP2'))
        rows_1d['ang_op1'].append(_ba(tpl, "O5'", 'P', 'OP1'))
        rows_1d['ang_op2'].append(_ba(tpl, "O5'", 'P', 'OP2'))
        rows_1d['psi_op1'].append(_dr(tpl, "O3'", "O5'", 'P', 'OP1'))
        rows_1d['psi_op2'].append(_dr(tpl, "O3'", "O5'", 'P', 'OP2'))
        rows_1d['psi_o3'].append(_dr(tpl, "C5'", "C4'", "C3'", "O3'"))
        rows_1d['bl_o4_c4'].append(_bl(tpl, "O4'", "C4'"))
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


# --- Torch geometry decoder (analytic ring + stereo exocyclic + multi-torsion phosphate) ---

_GEO_EPS = 1e-8
_PHI_SUGAR_GRID = 128
_PHI_PHOS_GRID = 64


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


def wrap_dihedral_diff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = (a - b + torch.pi) % (2.0 * torch.pi) - torch.pi
    return d


def nus_rad_from_P_tau_torch(
    P_rad: torch.Tensor,
    tau_m: torch.Tensor,
) -> torch.Tensor:
    """Five endocyclic ν (rad) in the same order as _RING_TORSION_DEFS (ν₂…ν₁ cycle)."""
    dev = P_rad.device
    dtype = P_rad.dtype
    p = P_rad.reshape(-1, 1)
    t = tau_m.reshape(-1, 1)
    offs = torch.tensor(
        [0.0, 4.0, 8.0, 2.0, 6.0],
        device=dev, dtype=dtype,
    ) * (torch.pi / 5.0)
    return t * torch.cos(p + offs.unsqueeze(0))


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


def build_sugar_ring_analytic_torch(
    chi: torch.Tensor,
    P: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
    *,
    bond_lengths: Optional[dict] = None,
    bond_angles: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Build sugar ring atoms in local base frame (χ reserved; ring anchors match template).

    Returns O4', C1', C2', C3', C4' with shapes [..., 3].
    """
    _ = bond_lengths, bond_angles, chi  # API compatibility; χ does not alter sugar-only coords here
    if P.shape != tau_m.shape or P.shape != restype_indices.shape:
        raise ValueError('P, tau_m, restype_indices must share the same leading shape.')
    if P.ndim == 0:
        P = P.unsqueeze(0)
        tau_m = tau_m.unsqueeze(0)
        restype_indices = restype_indices.unsqueeze(0)
    B = int(P.shape[0])
    dev = P.device
    dtype = P.dtype
    ri = restype_indices.long()
    tc = _get_template_tensors(str(dev))

    c1 = tc['c1'][ri].reshape(B, 3)
    o4 = tc['o4'][ri].reshape(B, 3)
    c4_ref = tc['c4_ref'][ri].reshape(B, 3)
    axis = c1 - o4
    axis_u = axis / (axis.norm(dim=-1, keepdim=True) + _GEO_EPS)

    _HP = torch.tensor(np.pi / 2.0, device=dev, dtype=dtype)
    tau_c = tau_m.clamp(min=1e-4)
    nu_tgt = nus_rad_from_P_tau_torch(P, tau_c)
    nu3 = nu_tgt[:, 1]
    nu4 = nu_tgt[:, 2]
    nu0 = nu_tgt[:, 3]

    def bl(key: str) -> torch.Tensor:
        return tc[key][ri]

    def ba(key: str) -> torch.Tensor:
        return tc[key][ri]

    bl_o4c4 = tc['bl_o4_c4'][ri]

    # Grid over rotation of the template C4' reference used in the first NeRF pivot (closure DOF)
    n_grid = _PHI_SUGAR_GRID
    phis = torch.linspace(
        0.0, 2.0 * torch.pi, n_grid, device=dev, dtype=dtype,
    )

    chir_ref = tc['ring_chiral_triple'][ri].reshape(-1, 1).expand(-1, n_grid)

    v_rot = (c4_ref - o4).reshape(B, 3)
    c4_exp = o4.unsqueeze(1) + _rodrigues_rotate_point_torch(
        v_rot.unsqueeze(1).expand(-1, n_grid, -1),
        axis_u.unsqueeze(1).expand(-1, n_grid, -1),
        phis.view(1, -1).expand(B, -1),
    )

    # Broadcast batch × grid
    c1g = c1.unsqueeze(1).expand(-1, n_grid, -1)
    o4g = o4.unsqueeze(1).expand(-1, n_grid, -1)
    nu3g = nu3.unsqueeze(1).expand(-1, n_grid)
    nu4g = nu4.unsqueeze(1).expand(-1, n_grid)
    nu0g = nu0.unsqueeze(1).expand(-1, n_grid)

    c2 = nerf_place_torch(
        c4_exp.reshape(B * n_grid, 3),
        o4g.reshape(B * n_grid, 3),
        c1g.reshape(B * n_grid, 3),
        bl('bl_c2_c1').unsqueeze(1).expand(-1, n_grid).reshape(-1),
        ba('ba_o4_c1_c2').unsqueeze(1).expand(-1, n_grid).reshape(-1),
        (nu3g - _HP).reshape(-1),
    ).reshape(B, n_grid, 3)

    c3 = nerf_place_torch(
        o4g.reshape(B * n_grid, 3),
        c1g.reshape(B * n_grid, 3),
        c2.reshape(B * n_grid, 3),
        bl('bl_c3_c2').unsqueeze(1).expand(-1, n_grid).reshape(-1),
        ba('ba_c1_c2_c3').unsqueeze(1).expand(-1, n_grid).reshape(-1),
        (nu4g - _HP).reshape(-1),
    ).reshape(B, n_grid, 3)

    c4 = nerf_place_torch(
        c1g.reshape(B * n_grid, 3),
        c2.reshape(B * n_grid, 3),
        c3.reshape(B * n_grid, 3),
        bl('bl_c4_c3').unsqueeze(1).expand(-1, n_grid).reshape(-1),
        ba('ba_c2_c3_c4').unsqueeze(1).expand(-1, n_grid).reshape(-1),
        (nu0g - _HP).reshape(-1),
    ).reshape(B, n_grid, 3)

    v2 = c2 - o4g
    v3 = c3 - o4g
    v4 = c4 - o4g
    triple = (torch.linalg.cross(v2, v3) * v4).sum(dim=-1)

    ra = {}
    for nm, tv in zip(("C1'", "C2'", "C3'", "C4'", "O4'"), (c1g, c2, c3, c4, o4g)):
        ra[nm] = tv
    nu_meas = []
    for a0n, a1n, a2n, a3n in _RING_TORSION_DEFS:
        p0 = ra[a0n]
        p1 = ra[a1n]
        p2 = ra[a2n]
        p3 = ra[a3n]
        nu_meas.append(
            dihedral_rad_torch(
                p0.reshape(B * n_grid, 3), p1.reshape(B * n_grid, 3),
                p2.reshape(B * n_grid, 3), p3.reshape(B * n_grid, 3),
            ).reshape(B, n_grid, 1),
        )
    nu_meas_t = torch.cat(nu_meas, dim=-1)

    nu_tgt_g = nu_tgt.unsqueeze(1).expand(-1, n_grid, -1)
    d_nu = wrap_dihedral_diff_torch(nu_meas_t, nu_tgt_g)
    err_nu = (d_nu ** 2).sum(dim=-1)

    d_close = (c4 - o4g).norm(dim=-1) - bl_o4c4.unsqueeze(-1)
    err_close = d_close ** 2

    chir_miss = (torch.sign(triple) - torch.sign(chir_ref)).abs()

    loss = 2.5 * err_nu + 0.2 * err_close + 1e-3 * chir_miss

    best = loss.argmin(dim=-1)
    idx = torch.arange(B, device=dev, dtype=torch.long)
    out = {
        "C1'": c1,
        "C2'": c2[idx, best],
        "C3'": c3[idx, best],
        "C4'": c4[idx, best],
        "O4'": o4,
    }
    return out


def add_exocyclic_sugar_atoms_torch(
    ring_atoms: dict[str, torch.Tensor],
    *,
    restype_indices: Optional[torch.Tensor] = None,
    geometry: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Add O3' and C5' using template stereochemistry (fixed ψ from template)."""
    _ = geometry
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
    ri = ri.long()
    tc = _get_template_tensors(str(dev))
    _HP = torch.tensor(np.pi / 2.0, device=dev, dtype=dtype)

    def bl(key: str) -> torch.Tensor:
        return tc[key][ri]

    def ba(key: str) -> torch.Tensor:
        return tc[key][ri]

    c5 = nerf_place_torch(
        o4, c3, c4,
        bl('bl_c5_c4'), ba('ba_c3_c4_c5'),
        tc['psi_c5'][ri] - _HP,
    )
    o3 = nerf_place_torch(
        c5, c4, c3,
        bl('bl_o3_c3'), ba('ba_c4_c3_o3'),
        tc['psi_o3'][ri] - _HP,
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
    """Place O5' using γ = O5'–C5'–C4'–C3' (NeRF chain, same bond/angle prior as legacy)."""
    _ = geometry
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
    tc = _get_template_tensors(str(dev))
    _HP = torch.tensor(np.pi / 2.0, device=dev, dtype=dtype)

    def bl(key: str) -> torch.Tensor:
        return tc[key][ri]

    def ba(key: str) -> torch.Tensor:
        return tc[key][ri]

    o5 = nerf_place_torch(c3, c4, c5, bl('bl_o5_c5'), ba('ba_c4_c5_o5'), g - _HP)
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
    weight_epsilon=1.0,
    weight_zeta=1.0,
    weight_alpha=1.0,
    weight_beta=1.0,
) -> dict[str, torch.Tensor]:
    """Place P on the O3'prev–O5'next circle; scan φ to match up to four target dihedrals."""
    g = geometry or {}
    dev = epsilon_prev.device
    dtype = epsilon_prev.dtype

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
    tc = _get_template_tensors(str(dev))
    r_o3p = tc['bl_o3_p'][ri]
    l_po5 = tc['r_po3'][ri]

    d_vec = o5n - o3p
    d_o3_o5 = d_vec.norm(dim=-1).clamp(min=_GEO_EPS)
    axis = d_vec / d_o3_o5.unsqueeze(-1)

    sum_l = r_o3p + l_po5
    diff_l = (r_o3p - l_po5).abs()
    feasible = (d_o3_o5 >= diff_l - 1e-6) & (d_o3_o5 <= sum_l + 1e-6)

    a_coef = (r_o3p * r_o3p - l_po5 * l_po5 + d_o3_o5 * d_o3_o5) / (2.0 * d_o3_o5)
    h_sq = (r_o3p * r_o3p - a_coef * a_coef).clamp(min=0.0)
    h_val = torch.sqrt(h_sq)

    midpoint = o3p + a_coef.unsqueeze(-1) * axis

    tmp = torch.zeros_like(axis)
    tmp[..., 0] = 1.0
    use_y = axis[..., 0].abs() >= 0.9
    tmp[use_y, 0] = 0.0
    tmp[use_y, 1] = 1.0
    v1 = torch.linalg.cross(tmp, axis)
    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + _GEO_EPS)
    v2 = torch.linalg.cross(axis, v1)

    n_grid = int(g.get('n_phi', _PHI_PHOS_GRID))
    ts = torch.linspace(0.0, 2.0 * torch.pi, n_grid, device=dev, dtype=dtype)
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

    p0 = c4p.unsqueeze(1)
    p1 = c3p.unsqueeze(1)
    p2 = o3p.unsqueeze(1)
    p3 = p_cands
    eps_m = dihedral_rad_torch(
        p0.reshape(-1, 3), p1.reshape(-1, 3),
        p2.reshape(-1, 3), p3.reshape(-1, 3),
    ).reshape(B, n_grid)

    ze_m = dihedral_rad_torch(
        c3p.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
        o3p.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
        p_cands.reshape(-1, 3),
        o5n.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
    ).reshape(B, n_grid)

    al_m = dihedral_rad_torch(
        o3p.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
        p_cands.reshape(-1, 3),
        o5n.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
        c5n.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
    ).reshape(B, n_grid)

    be_m = dihedral_rad_torch(
        p_cands.reshape(-1, 3),
        o5n.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
        c5n.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
        c4n.unsqueeze(1).expand(-1, n_grid, -1).reshape(-1, 3),
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

    _HP = torch.tensor(np.pi / 2.0, device=dev, dtype=dtype)
    bl_op1 = tc['bl_op1'][ri]
    bl_op2 = tc['bl_op2'][ri]
    ang_op1 = tc['ang_op1'][ri]
    ang_op2 = tc['ang_op2'][ri]
    psi_op1 = tc['psi_op1'][ri]
    psi_op2 = tc['psi_op2'][ri]

    op1 = nerf_place_torch(o3p, o5n, p_pick, bl_op1, ang_op1, psi_op1 - _HP)
    op2 = nerf_place_torch(o3p, o5n, p_pick, bl_op2, ang_op2, psi_op2 - _HP)

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

    def _blen(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _bangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba, bc = a - b, c - b
        cos_t = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
        return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))

    # Extract relevant torsion angles
    alpha = float(torsions[TOR_ALPHA])
    beta = float(torsions[TOR_BETA])
    gamma = float(torsions[TOR_GAMMA])
    P_rad = float(torsions[TOR_PUCKER_P])
    if tau_m is not None:
        _TAU_DNA = float(tau_m)
    else:
        _TAU_DNA = float(_template_tau_m.get(restype, 0.611))

    # _RING_TORSION_DEFS starts at standard ν₂ (C1'-C2'-C3'-C4').
    # Mapping: our k → A-S index: 0→ν₂, 1→ν₃, 2→ν₄, 3→ν₀, 4→ν₁.
    _NU_OFFSETS = np.array([0.0, 4.0, 8.0, 2.0, 6.0]) * np.pi / 5.0
    nus = _TAU_DNA * np.cos(P_rad + _NU_OFFSETS)
    nu0, _nu1, _nu2, nu3, nu4 = nus

    # Anchor atoms — fixed in local frame from canonical template
    c1 = tpl["C1'"].copy()
    o4 = tpl["O4'"].copy()
    c4_ref = tpl["C4'"].copy()

    # nerf_place(A, B, C, r, θ, ψ) satisfies dihedral(A,B,C,D) = ψ + π/2
    _HP = np.pi / 2.0

    # ── Sugar ring ──────────────────────────────────────────────────────────
    c2 = nerf_place(
        c4_ref, o4, c1,
        _blen(tpl["C2'"], c1),
        _bangle(o4, c1, tpl["C2'"]),
        nu3 - _HP,
    )
    c3 = nerf_place(
        o4, c1, c2,
        _blen(tpl["C3'"], tpl["C2'"]),
        _bangle(c1, tpl["C2'"], tpl["C3'"]),
        nu4 - _HP,
    )
    c4 = nerf_place(
        c1, c2, c3,
        _blen(tpl["C4'"], tpl["C3'"]),
        _bangle(tpl["C2'"], tpl["C3'"], tpl["C4'"]),
        nu0 - _HP,
    )

    psi_c5 = dihedral_rad(o4, tpl["C3'"], tpl["C4'"], tpl["C5'"])
    c5 = nerf_place(
        o4, c3, c4,
        _blen(tpl["C5'"], tpl["C4'"]),
        _bangle(tpl["C3'"], tpl["C4'"], tpl["C5'"]),
        psi_c5 - _HP,
    )

    o5_gamma = nerf_place(
        c3, c4, c5,
        _blen(tpl["O5'"], tpl["C5'"]),
        _bangle(tpl["C4'"], tpl["C5'"], tpl["O5'"]),
        gamma - _HP,
    )

    psi_o3 = dihedral_rad(tpl["C5'"], tpl["C4'"], tpl["C3'"], tpl["O3'"])
    o3 = nerf_place(
        c5, c4, c3,
        _blen(tpl["O3'"], tpl["C3'"]),
        _bangle(tpl["C4'"], tpl["C3'"], tpl["O3'"]),
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
    r_po3 = 0.0
    o5 = o5_gamma.copy()

    if o3_prev_local is not None:
        o3p = np.asarray(o3_prev_local, dtype=np.float64).reshape(3)
        r_po3 = _blen(tpl['P'], tpl["O5'"])
        theta_po3 = np.deg2rad(119.0)
        l_po5 = _blen(tpl['P'], tpl["O5'"])
        d_o3_o5 = float(np.linalg.norm(o3p - o5_gamma))
        if abs(r_po3 - l_po5) <= d_o3_o5 <= r_po3 + l_po5:
            p_built = close_phosphate_bridge(
                o3p, o5_gamma, c5, r_po3, l_po5, alpha,
            )
        else:
            # ψ = −α aligns dihedral(O3′, P, O5′, C5′) with α given nerf_place's +π/2 offset.
            p_built = nerf_place(
                c5, o5_gamma, o3p, r_po3, theta_po3, -alpha - _HP,
            )
        o5 = o5_gamma

    out["O5'"] = o5

    if p_built is not None and o3p is not None:
        o5_f = out["O5'"]
        r_p_op1 = _blen(tpl['P'], tpl["OP1"])
        r_p_op2 = _blen(tpl['P'], tpl["OP2"])
        ang_op1 = _bangle(tpl["O5'"], tpl['P'], tpl["OP1"])
        ang_op2 = _bangle(tpl["O5'"], tpl['P'], tpl["OP2"])
        psi_op1 = dihedral_rad(tpl["O3'"], tpl["O5'"], tpl['P'], tpl["OP1"])
        psi_op2 = dihedral_rad(tpl["O3'"], tpl["O5'"], tpl['P'], tpl["OP2"])
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


def build_backbone_from_torsions_legacy_torch(
    torsions: 'torch.Tensor',          # [B, N_TORSIONS]
    tau_m: 'torch.Tensor',             # [B]
    restype_indices: 'torch.Tensor',   # [B] long, 0=A 1=C 2=G 3=T
    o3_prev_local: Optional[torch.Tensor] = None,  # [B, 3] or None
) -> dict[str, torch.Tensor]:
    """Legacy NeRF sugar + template-ψ O3′ + α-only phosphate circle search (preserved for comparison)."""
    import torch

    B = torsions.shape[0]
    dev = torsions.device
    _HP = torch.pi / 2.0

    tc = _get_template_tensors(str(dev))
    ri = restype_indices  # [B]

    # Anchor atoms from template (same for all samples of same restype)
    c1 = tc['c1'][ri]      # [B, 3]
    o4 = tc['o4'][ri]      # [B, 3]
    c4_ref = tc['c4_ref'][ri]  # [B, 3]

    # Torsions
    alpha = torsions[:, TOR_ALPHA]
    gamma = torsions[:, TOR_GAMMA]
    P_rad = torsions[:, TOR_PUCKER_P]
    tau_m_c = tau_m.clamp(min=1e-3)

    # Altona-Sundaralingam puckering [B, 5]
    offsets = torch.tensor(
        [0., 4., 8., 2., 6.],
        dtype=torch.float32, device=dev,
    ) * (torch.pi / 5.0)
    nus = tau_m_c.unsqueeze(-1) * torch.cos(
        P_rad.unsqueeze(-1) + offsets.unsqueeze(0),
    )
    nu0, nu3, nu4 = nus[:, 3], nus[:, 1], nus[:, 2]

    def bl(key):
        return tc[key][ri]

    def ba(key):
        return tc[key][ri]

    # Sugar ring — all NeRF calls are fully batched [B, 3]
    c2 = nerf_place_torch(
        c4_ref, o4, c1,
        bl('bl_c2_c1'), ba('ba_o4_c1_c2'), nu3 - _HP,
    )
    c3 = nerf_place_torch(
        o4, c1, c2,
        bl('bl_c3_c2'), ba('ba_c1_c2_c3'), nu4 - _HP,
    )
    c4 = nerf_place_torch(
        c1, c2, c3,
        bl('bl_c4_c3'), ba('ba_c2_c3_c4'), nu0 - _HP,
    )
    c5 = nerf_place_torch(
        o4, c3, c4,
        bl('bl_c5_c4'), ba('ba_c3_c4_c5'),
        tc['psi_c5'][ri] - _HP,
    )
    o5 = nerf_place_torch(
        c3, c4, c5,
        bl('bl_o5_c5'), ba('ba_c4_c5_o5'), gamma - _HP,
    )
    o3 = nerf_place_torch(
        c5, c4, c3,
        bl('bl_o3_c3'), ba('ba_c4_c3_o3'), tc['psi_o3'][ri] - _HP,
    )

    out = {
        "C1'": c1, "C2'": c2, "C3'": c3, "C4'": c4,
        "C5'": c5, "O4'": o4, "O3'": o3, "O5'": o5,
    }

    if o3_prev_local is None:
        return out

    o3p = o3_prev_local  # [B, 3]

    # P placement via NeRF from O3'_prev
    r_o3p = bl('bl_o3_p')  # |O3'−P|
    l_po5 = bl('r_po3')  # |P−O5'|
    theta_po3 = torch.full(
        (B,),
        float(np.deg2rad(119.0)),
        dtype=torch.float32, device=dev,
    )
    p_built = nerf_place_torch(
        c5, o5, o3p, r_o3p, theta_po3,
        -alpha - _HP,
    )  # [B, 3]

    # Vectorized circle closure for feasible samples
    d_o3_o5 = (o3p - o5).norm(dim=-1)  # [B]
    feasible = (d_o3_o5 >= (r_o3p - l_po5).abs()) \
        & (d_o3_o5 <= r_o3p + l_po5)

    if feasible.any():
        d_f = d_o3_o5[feasible]
        o3f = o3p[feasible]
        o5f = o5[feasible]
        c5f = c5[feasible]
        r_f = r_o3p[feasible]
        l_f = l_po5[feasible]
        alp_f = alpha[feasible]

        a_coef = (r_f**2 - l_f**2 + d_f**2) / (2.0 * d_f)
        h_val = (r_f**2 - a_coef**2).clamp(min=0.0).sqrt()
        axis = (o5f - o3f) / (d_f.unsqueeze(-1) + 1e-12)
        mid = o3f + a_coef.unsqueeze(-1) * axis

        tmp = torch.zeros_like(axis)
        tmp[:, 0] = 1.0
        use_y = axis[:, 0].abs() >= 0.9
        tmp[use_y, 0] = 0.0
        tmp[use_y, 1] = 1.0
        v1 = torch.linalg.cross(tmp, axis)
        v1 = v1 / (v1.norm(dim=-1, keepdim=True) + 1e-12)
        v2 = torch.linalg.cross(axis, v1)

        N_SAMP = 360
        ts = torch.linspace(
            0, 2 * torch.pi, N_SAMP,
            dtype=torch.float32, device=dev,
        )
        p_cands = (
            mid.unsqueeze(1)
            + h_val[:, None, None]
            * (
                torch.cos(ts)[None, :, None] * v1.unsqueeze(1)
                + torch.sin(ts)[None, :, None] * v2.unsqueeze(1)
            )
        )

        def _batch_dihedral(p0, p1, p2, p3):
            b1 = p1 - p0.unsqueeze(1)
            b2 = p2.unsqueeze(1) - p1
            b3 = p3.unsqueeze(1) - p2.unsqueeze(1)
            n1 = torch.linalg.cross(b1, b2)
            n2 = torch.linalg.cross(b2, b3)
            b2u = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-12)
            m1 = torch.linalg.cross(
                n1 / (n1.norm(dim=-1, keepdim=True) + 1e-12),
                b2u,
            )
            n1u = n1 / (n1.norm(dim=-1, keepdim=True) + 1e-12)
            n2u = n2 / (n2.norm(dim=-1, keepdim=True) + 1e-12)
            x = (n1u * n2u).sum(-1)
            y = (m1 * n2u).sum(-1)
            return torch.atan2(y, x)

        dihed = _batch_dihedral(o3f, p_cands, o5f, c5f)
        err = (dihed - alp_f.unsqueeze(-1)).remainder(2 * torch.pi)
        err = torch.where(err > torch.pi, 2 * torch.pi - err, err)
        best = err.argmin(dim=-1)
        p_closed = p_cands[
            torch.arange(p_cands.shape[0], device=dev),
            best,
        ]

        p_built = p_built.clone()
        p_built[feasible] = p_closed

    op1 = nerf_place_torch(
        o3p, o5, p_built,
        bl('bl_op1'), ba('ang_op1'),
        tc['psi_op1'][ri] - _HP,
    )
    op2 = nerf_place_torch(
        o3p, o5, p_built,
        bl('bl_op2'), ba('ang_op2'),
        tc['psi_op2'][ri] - _HP,
    )
    out['P'] = p_built
    out['OP1'] = op1
    out['OP2'] = op2

    return out


_BACKBONE_ATOM_ORDER = (
    "C1'", "C2'", "C3'", "C4'", "C5'", "OP1", "OP2", "P", "O3'", "O4'", "O5'",
)


def build_backbone_from_torsions_torch(
    torsions: 'torch.Tensor',
    tau_m: 'torch.Tensor',
    restype_indices: 'torch.Tensor',
    o3_prev_local: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Batched decoder: analytic sugar ring, stereo exocyclic O3′/C5′, γ→O5′.

    Phosphate uses ``close_phosphate_bridge_multi_torch``; without prior-nucleotide ε/ζ in this API,
    only α and β targets are weighted (incoming bridge from o3_prev in target frame).
    """
    import torch

    ri = restype_indices.long()
    chi = torsions[:, TOR_CHI]
    P = torsions[:, TOR_PUCKER_P]
    gamma = torsions[:, TOR_GAMMA]
    alpha = torsions[:, TOR_ALPHA]
    beta = torsions[:, TOR_BETA]
    eps_lat = torsions[:, TOR_EPS]
    zet_lat = torsions[:, TOR_ZETA]
    ring = build_sugar_ring_analytic_torch(chi, P, tau_m, ri)
    atoms = add_exocyclic_sugar_atoms_torch(ring, restype_indices=ri)
    atoms = add_o5_from_gamma_torch(atoms, gamma, restype_indices=ri)

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
        geometry={'restype_indices_next': ri},
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
    name_to_j = {nm: j for j, nm in enumerate(_BACKBONE_ATOM_ORDER)}
    BW = B * W

    flat_t = torsions.reshape(BW, N_TORSIONS)
    flat_tm = tau_m.reshape(BW)
    flat_ri = restype_indices.reshape(BW).long()
    chi = flat_t[:, TOR_CHI]
    P_rad = flat_t[:, TOR_PUCKER_P]
    ring = build_sugar_ring_analytic_torch(chi, P_rad, flat_tm, flat_ri)
    atoms = add_exocyclic_sugar_atoms_torch(ring, restype_indices=flat_ri)
    atoms = add_o5_from_gamma_torch(
        atoms, flat_t[:, TOR_GAMMA], restype_indices=flat_ri,
    )

    Ri_flat = nt_frames_world.reshape(BW, 3, 3)
    ok_flat = nt_origins_world.reshape(BW, 3)

    bb_flat = torch.full((BW, n_bb, 3), float('nan'), device=dev, dtype=dtype)
    for nm in ("C1'", "C2'", "C3'", "C4'", "C5'", "O4'", "O3'", "O5'"):
        local = atoms[nm]
        bb_flat[:, name_to_j[nm]] = torch.bmm(
            local.unsqueeze(1),
            Ri_flat.transpose(-2, -1),
        ).squeeze(1) + ok_flat

    bb = bb_flat.view(B, W, n_bb, 3)

    for k in range(1, W):
        exec_ok = (
            torch.isfinite(bb[:, k - 1, name_to_j["O3'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k - 1, name_to_j["C3'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k - 1, name_to_j["C4'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k, name_to_j["O5'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k, name_to_j["C5'"]]).all(dim=-1)
            & torch.isfinite(bb[:, k, name_to_j["C4'"]]).all(dim=-1)
        )
        if not bool(exec_ok.any().item()):
            continue
        idx = torch.nonzero(exec_ok, as_tuple=False).squeeze(-1)
        ok = nt_origins_world[idx, k]
        Rk = nt_frames_world[idx, k]

        def _wl(vw: torch.Tensor) -> torch.Tensor:
            dv = vw - ok
            return torch.bmm(dv.unsqueeze(1), Rk.transpose(-2, -1)).squeeze(1)

        prev_loc = {
            "O3'": _wl(bb[idx, k - 1, name_to_j["O3'"]]),
            "C3'": _wl(bb[idx, k - 1, name_to_j["C3'"]]),
            "C4'": _wl(bb[idx, k - 1, name_to_j["C4'"]]),
        }
        next_loc = {
            "O5'": _wl(bb[idx, k, name_to_j["O5'"]]),
            "C5'": _wl(bb[idx, k, name_to_j["C5'"]]),
            "C4'": _wl(bb[idx, k, name_to_j["C4'"]]),
            '_ri': restype_indices[idx, k],
        }

        if torsion_mask is None:
            we = wz = wa = wb = 1.0
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
            geometry={'restype_indices_next': restype_indices[idx, k]},
            weight_epsilon=we,
            weight_zeta=wz,
            weight_alpha=wa,
            weight_beta=wb,
        )
        loc_p = phosph['P']
        loc_o1 = phosph['OP1']
        loc_o2 = phosph['OP2']
        bb[idx, k, name_to_j['P']] = torch.bmm(
            loc_p.unsqueeze(1), Rk.transpose(-2, -1),
        ).squeeze(1) + ok
        bb[idx, k, name_to_j['OP1']] = torch.bmm(
            loc_o1.unsqueeze(1), Rk.transpose(-2, -1),
        ).squeeze(1) + ok
        bb[idx, k, name_to_j['OP2']] = torch.bmm(
            loc_o2.unsqueeze(1), Rk.transpose(-2, -1),
        ).squeeze(1) + ok

    return bb


def build_window_backbone_from_torsions_torch(
    torsions: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
    nt_origins_world: torch.Tensor,
    nt_frames_world: torch.Tensor,
    torsion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """World backbone [W, n_bb, 3] with the same atom order as ``utils.backbone_atoms``.

    For each k>=1, place P_k / OP1 / OP2 using ε,ζ from residue k−1 and α,β from k (masked).
    Bridge geometry uses k−1 world O3′,C3′,C4′ expressed in nucleotide k local frame.
    """
    import torch

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
