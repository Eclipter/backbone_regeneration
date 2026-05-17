"""Torsion definitions, sugar puckering (Altona–Sundaralingam / MDAnalysis conventions), and angle wrapping."""

import functools
import math
from typing import Any, Optional, cast

import numpy as np
import torch

from ..torsion_constants import (
    N_TORSIONS,
    TOR_ALPHA,
    TOR_BETA,
    TOR_CHI,
    TOR_DELTA,
    TOR_EPS,
    TOR_ETA_P,
    TOR_GAMMA,
    TOR_PSEUDOROTATION_PHASE,
    TOR_ZETA,
)
from .primitives import (
    _bond_angle,
    _dihedral_rad,
    dihedral_rad,
    local_to_world_points,
    nerf_place,
    rodrigues_rotate_point,
    world_to_local_points,
    wrap_angle_rad,
    wrap_dihedral_diff,
)
from .torsions import (
    RING_TORSION_DEFS as _RING_TORSION_DEFS,
    _PSEUDOROTATION_OFFSETS,
    nucleotide_torsions,
    nus_rad_from_phase_and_amplitude,
    phase_and_amplitude_to_nus,
    pseudorotation_phase_rad_from_nus,
    pucker_amplitude_rad,
    sugar_ring_torsions,
)

_GEO_EPS = 1e-8
_PHI_PHOS_GRID = 128
_HALF_PI = math.pi / 2.0


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
        _dihedral_rad(tpl[q[0]], tpl[q[1]], tpl[q[2]], tpl[q[3]])
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
    device = torch.device(device_str)
    restypes = ['A', 'C', 'G', 'T']

    def _bl(tpl, a, b):
        return float(np.linalg.norm(tpl[a] - tpl[b]))

    def _ba(tpl, a, b, c):
        return _bond_angle(tpl[a], tpl[b], tpl[c])

    def _dr(tpl, a, b, c, d):
        return _dihedral_rad(tpl[a], tpl[b], tpl[c], tpl[d])

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
        'psi_op1_c5', 'psi_op2_c5',
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
        rows_1d['psi_op1_c5'].append(_dr(tpl, "C5'", "O5'", 'P', 'OP1'))
        rows_1d['psi_op2_c5'].append(_dr(tpl, "C5'", "O5'", 'P', 'OP2'))
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
def _ring_dihedrals_from_coords(
    ring_atoms: dict[str, torch.Tensor],
) -> torch.Tensor:
    """[B, 5] dihedrals in _RING_TORSION_DEFS order."""
    names = _RING_TORSION_DEFS
    out = []
    for a0n, a1n, a2n, a3n in names:
        out.append(
            dihedral_rad(
                ring_atoms[a0n], ring_atoms[a1n],
                ring_atoms[a2n], ring_atoms[a3n],
            ).unsqueeze(-1),
        )
    return torch.cat(out, dim=-1)


def signed_tetra_volume(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Signed volume ∝ (b−a)·((c−a)×(d−a)); shape matches broadcast of inputs."""
    return (torch.linalg.cross(b - a, c - a, dim=-1) * (d - a)).sum(dim=-1)




def _apply_chi_rotation_to_sugar_ring(
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
    chi_meas = dihedral_rad(o4, c1, n_atom, c_atom)
    dchi = wrap_dihedral_diff(chi_target, chi_meas)
    out = dict(ring)
    for nm in ("O4'", "C2'", "C3'", "C4'"):
        v = ring[nm] - c1
        out[nm] = rodrigues_rotate_point(v, u, dchi) + c1
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

    def _ring_nus_from_xyz_dict(d: dict[str, np.ndarray]) -> np.ndarray:
        return np.array(
            [_dihedral_rad(d[a0], d[a1], d[a2], d[a3]) for a0, a1, a2, a3 in _RING_TORSION_DEFS],
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
                nu_act = _ring_nus_from_xyz_dict(_ring_dict_from_xyz(pos))
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
def _pseudorotation_offsets(
    device_str: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        _PSEUDOROTATION_OFFSETS,
        device=torch.device(device_str),
        dtype=dtype,
    )


@functools.lru_cache(maxsize=None)
def _scalar_constant(
    value: float,
    device_str: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(value, device=torch.device(device_str), dtype=dtype)


@functools.lru_cache(maxsize=None)
def _phi_grid(
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


def sugar_ring_from_xy_z(
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


def _map_sugar_local_ring_to_base_frame(
    ring_local: dict[str, torch.Tensor],
    *,
    c1_base: torch.Tensor,
    o4_base: torch.Tensor,
    c2_ref: torch.Tensor,
    c4_base: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Map a sugar-local gauge to the nucleotide base frame with a rigid transform.

    The ring-local convention pins C2' to ``(x2, -y2, 0)`` with ``z = 0`` by construction,
    so the world-frame y-axis must point towards the template C2' (not towards N).
    The legacy (C1', O4', N)-plane heuristic put C2' in that plane, which is geometrically
    wrong for D-deoxyribose and produced a ring rotated ~50° around the C1'-O4' axis vs
    the template (≈3.6 Å RMSD even on round-trip inputs).
    """
    x_axis = _safe_normalize(o4_base - c1_base)
    y_seed = c2_ref - c1_base
    y_seed = y_seed - (y_seed * x_axis).sum(dim=-1, keepdim=True) * x_axis
    # Flip sign because ring-local places C2' at y = -y2 (D-deoxyribose stereochemistry).
    y_seed = -y_seed
    alt_seed = c4_base - c1_base
    alt_seed = alt_seed - (alt_seed * x_axis).sum(dim=-1, keepdim=True) * x_axis
    _, basis_y, _ = _orthonormal_basis_from_axis(x_axis, eps=_GEO_EPS)
    use_alt = y_seed.norm(dim=-1, keepdim=True) < 1e-6
    y_seed = torch.where(use_alt, alt_seed, y_seed)
    use_basis = y_seed.norm(dim=-1, keepdim=True) < 1e-6
    y_axis = torch.where(use_basis, basis_y, y_seed)
    y_axis = _safe_normalize(y_axis)
    z_axis = _safe_normalize(torch.linalg.cross(x_axis, y_axis, dim=-1))
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    # Column-stacked basis: frame[..., :, j] is the j-th basis vector in world coords.
    # `local_to_world_points` does `local @ frame.T + origin`, which expects this convention
    # (the same one used by `nt_frames_world` across the dataset).
    frame = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return {
        name: local_to_world_points(coords, c1_base, frame)
        for name, coords in ring_local.items()
    }


def _flatten_sugar_builder_inputs(
    chi: torch.Tensor,
    P: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], bool]:
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
    if chi.shape != P.shape:
        raise ValueError('chi must share the same leading shape as P.')
    return (
        chi.reshape(-1),
        P.reshape(-1),
        tau_m.reshape(-1),
        restype_indices.reshape(-1).long(),
        tuple(P.shape),
        squeeze_batch,
    )


def _map_sugar_ring_local_to_output(
    ring_local: dict[str, torch.Tensor],
    chi_f: torch.Tensor,
    ri: torch.Tensor,
    tc: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype,
    orig: tuple[int, ...],
    squeeze_batch: bool,
) -> dict[str, torch.Tensor]:
    c1_base = _template_select(tc, 'c1', ri, dtype=dtype).reshape(-1, 3)
    o4_base = _template_select(tc, 'o4', ri, dtype=dtype).reshape(-1, 3)
    n_atom = _template_select(tc, 'chi_n', ri, dtype=dtype).reshape(-1, 3)
    c_atom = _template_select(tc, 'chi_c', ri, dtype=dtype).reshape(-1, 3)
    c2_ref = _template_select(tc, 'c2_ref', ri, dtype=dtype).reshape(-1, 3)
    c4_base = _template_select(tc, 'c4_ref', ri, dtype=dtype).reshape(-1, 3)
    ring_pre = _map_sugar_local_ring_to_base_frame(
        ring_local,
        c1_base=c1_base,
        o4_base=o4_base,
        c2_ref=c2_ref,
        c4_base=c4_base,
    )
    out = _apply_chi_rotation_to_sugar_ring(ring_pre, chi_f, n_atom, c_atom)
    for nm in ("C1'", "C2'", "C3'", "C4'", "O4'"):
        out[nm] = out[nm].reshape(*orig, 3)
    if squeeze_batch:
        out = {k: v.squeeze(0) for k, v in out.items()}
    return out


def build_sugar_ring_closed_form(
    chi: torch.Tensor,
    P: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
    *,
    geometry: Optional[dict] = None,
    bond_lengths: Optional[dict] = None,
    bond_angles: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Analytic finite-branch cyclic sugar builder.

    Exact constraints:
      - all five ring bond lengths;
      - ring closure;
      - ν₀ = dihedral(C1',C2',C3',C4');
      - ν₄ = dihedral(O4',C1',C2',C3');
      - ring chirality;
      - χ after the final rigid rotation around the glycosidic axis.

    Residual constraints:
      - ν₁, ν₂, ν₃ are evaluated after construction and used only for branch
        selection. With fixed template lengths and angles, enforcing all five ν
        exactly is generally overconstrained.
    """
    _ = bond_lengths, bond_angles
    g = geometry or {}
    chi_f, P_f, tm_f, ri, orig, squeeze_batch = _flatten_sugar_builder_inputs(
        chi,
        P,
        tau_m,
        restype_indices,
    )
    device_str = str(P_f.device)
    dtype = P_f.dtype
    tc = g.get('template_tensors')
    if tc is None:
        tc = _get_template_tensors(device_str)

    def _g1(key: str) -> torch.Tensor:
        return _template_select(tc, key, ri, dtype=dtype)

    half_pi = _scalar_constant(_HALF_PI, device_str, dtype)
    nus = nus_rad_from_phase_and_amplitude(P_f, tm_f)
    bl_o4_c1 = (_g1('o4') - _g1('c1')).norm(dim=-1)
    bl_c2_c1 = _g1('bl_c2_c1')
    ba_o4_c1_c2 = _g1('ba_o4_c1_c2')
    bl_c3_c2 = _g1('bl_c3_c2')
    ba_c1_c2_c3 = _g1('ba_c1_c2_c3')
    bl_c4_c3 = _g1('bl_c4_c3')
    bl_o4_c4 = _g1('bl_o4_c4')

    zero = torch.zeros_like(bl_o4_c1)
    c1_local = torch.stack([zero, zero, zero], dim=-1)
    o4_local = torch.stack([bl_o4_c1, zero, zero], dim=-1)
    x2 = bl_c2_c1 * torch.cos(ba_o4_c1_c2)
    y2 = bl_c2_c1 * torch.sin(ba_o4_c1_c2)
    # D-deoxyribose stereochemistry pins C2' to the negative-y half of the (C1', O4', N) plane:
    # the two mirror-twin sugar rings (related by 180° rotation around the C1'-O4' axis) share
    # identical endocyclic ν dihedrals, so the legacy 4-candidate scheme picked either by chance.
    # See `_TEMPLATE_C2_Y_SIGN` for the empirical verification across A/C/G/T.
    c2_local = torch.stack([x2, -y2, zero], dim=-1)
    c3_local = nerf_place(
        o4_local,
        c1_local,
        c2_local,
        bl_c3_c2,
        ba_c1_c2_c3,
        nus[:, 4] - half_pi,
    )
    d_vec = o4_local - c3_local
    d_sq = (d_vec * d_vec).sum(dim=-1)
    d = d_sq.clamp(min=1e-8).sqrt()
    d_hat = d_vec / d.unsqueeze(-1)
    h = (d_sq + bl_c4_c3 ** 2 - bl_o4_c4 ** 2) / (2.0 * d)
    # `+1e-12` floor avoids `sqrt'(0) = +inf` on the boundary (infeasible C3'-C4'-O4' triangle).
    r_c = ((bl_c4_c3 ** 2 - h ** 2).clamp(min=0.0) + 1e-12).sqrt()
    circle_center = c3_local + h.unsqueeze(-1) * d_hat
    _, e1, e2 = _orthonormal_basis_from_axis(d_vec, eps=_GEO_EPS)
    bc32 = c3_local - c2_local
    cross_a = torch.linalg.cross(bc32, d_hat, dim=-1)
    cross_b = torch.linalg.cross(bc32, e1, dim=-1)
    cross_c = torch.linalg.cross(bc32, e2, dim=-1)
    n12 = _safe_normalize(torch.linalg.cross(c1_local - c2_local, bc32, dim=-1))
    m12 = torch.linalg.cross(n12, _safe_normalize(bc32), dim=-1)
    a0 = h * (n12 * cross_a).sum(dim=-1)
    a1 = h * (m12 * cross_a).sum(dim=-1)
    b0 = r_c * (n12 * cross_b).sum(dim=-1)
    b1 = r_c * (m12 * cross_b).sum(dim=-1)
    g0 = r_c * (n12 * cross_c).sum(dim=-1)
    g1 = r_c * (m12 * cross_c).sum(dim=-1)
    nu0_tgt = nus[:, 0]
    ct = torch.cos(nu0_tgt)
    st = torch.sin(nu0_tgt)
    pc = ct * b1 - st * b0
    qc = ct * g1 - st * g0
    rc_coef = ct * a1 - st * a0
    pq = (pc ** 2 + qc ** 2).clamp(min=1e-8).sqrt()
    # FP32-safe clamp eps: `1 - 1e-8 == 1.0` in float32, so the original (-1.0, 1.0) clamp let
    # `acos(+/-1)` -> backward = -1/sqrt(1-x^2) = -inf. Use 1e-6 (>> FP32 eps ~1.19e-7).
    acos_arg = (-rc_coef / pq).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    phi_a = torch.atan2(qc, pc) + torch.acos(acos_arg)
    phi_b = torch.atan2(qc, pc) - torch.acos(acos_arg)
    c4_a = circle_center + r_c.unsqueeze(-1) * (
        torch.cos(phi_a).unsqueeze(-1) * e1 + torch.sin(phi_a).unsqueeze(-1) * e2
    )
    c4_b = circle_center + r_c.unsqueeze(-1) * (
        torch.cos(phi_b).unsqueeze(-1) * e1 + torch.sin(phi_b).unsqueeze(-1) * e2
    )

    # Two ring-closure candidates only (a vs b for C4' on the intersection circle).
    ring_candidates = {
        "C1'": torch.stack([c1_local, c1_local], dim=1),
        "O4'": torch.stack([o4_local, o4_local], dim=1),
        "C2'": torch.stack([c2_local, c2_local], dim=1),
        "C3'": torch.stack([c3_local, c3_local], dim=1),
        "C4'": torch.stack([c4_a, c4_b], dim=1),
    }
    flat_candidates = {name: coords.reshape(-1, 3) for name, coords in ring_candidates.items()}
    nu_act = sugar_ring_torsions(flat_candidates).reshape(-1, 2, 5)
    residual = wrap_dihedral_diff(nu_act, nus.unsqueeze(1))
    score = (residual * residual).sum(dim=-1)
    chirality = signed_tetra_volume(
        flat_candidates["O4'"],
        flat_candidates["C2'"],
        flat_candidates["C3'"],
        flat_candidates["C4'"],
    ).reshape(-1, 2)
    chirality_ok = chirality * _g1('ring_chiral_triple').unsqueeze(-1) > 0.0
    inf_score = torch.full_like(score, float('inf'))
    score_valid = torch.where(chirality_ok, score, inf_score)
    any_valid = chirality_ok.any(dim=-1)
    best_idx_valid = torch.where(score_valid[:, 1] < score_valid[:, 0], 1, 0)
    best_idx_raw = torch.where(score[:, 1] < score[:, 0], 1, 0)
    best_idx = torch.where(any_valid, best_idx_valid, best_idx_raw)
    gather_idx = best_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ring_local = {
        name: coords.gather(1, gather_idx).squeeze(1)
        for name, coords in ring_candidates.items()
    }
    return _map_sugar_ring_local_to_output(
        ring_local,
        chi_f,
        ri,
        tc,
        dtype=dtype,
        orig=orig,
        squeeze_batch=squeeze_batch,
    )


def add_exocyclic_sugar_atoms(
    ring_atoms: dict[str, torch.Tensor],
    delta: torch.Tensor,
    *,
    restype_indices: Optional[torch.Tensor] = None,
    geometry: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Add C5' from template geometry and O3' from the predicted δ torsion."""
    g = geometry or {}
    ri = restype_indices if restype_indices is not None else ring_atoms.get('_ri')
    if ri is None:
        raise ValueError('Pass restype_indices= or ring_atoms["_ri"] (long [B]).')
    o4 = ring_atoms["O4'"]
    c3 = ring_atoms["C3'"]
    c4 = ring_atoms["C4'"]
    d = delta
    if o4.ndim == 1:
        o4 = o4.unsqueeze(0)
        c3 = c3.unsqueeze(0)
        c4 = c4.unsqueeze(0)
        d = d.reshape(1)
        squeeze = True
    else:
        squeeze = False
    device = o4.device
    dtype = o4.dtype
    device_str = str(device)
    ri = ri.long()
    tc = g.get('template_tensors')
    if tc is None:
        tc = _get_template_tensors(device_str)
    half_pi = _scalar_constant(_HALF_PI, device_str, dtype)

    c5 = nerf_place(
        o4, c3, c4,
        _template_select(tc, 'bl_c5_c4', ri, dtype=dtype),
        _template_select(tc, 'ba_c3_c4_c5', ri, dtype=dtype),
        _template_select(tc, 'psi_c5', ri, dtype=dtype) - half_pi,
    )
    o3 = nerf_place(
        c5, c4, c3,
        _template_select(tc, 'bl_o3_c3', ri, dtype=dtype),
        _template_select(tc, 'ba_c4_c3_o3', ri, dtype=dtype),
        d - half_pi,
    )

    out = dict(ring_atoms)
    if squeeze:
        out["C5'"] = c5.squeeze(0)
        out["O3'"] = o3.squeeze(0)
    else:
        out["C5'"] = c5
        out["O3'"] = o3
    return out


def add_o5_from_gamma(
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
    device = c3.device
    dtype = c3.dtype
    device_str = str(device)
    tc = gctx.get('template_tensors')
    if tc is None:
        tc = _get_template_tensors(device_str)
    half_pi = _scalar_constant(_HALF_PI, device_str, dtype)

    o5 = nerf_place(
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


def close_phosphate_bridge_multi(
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
    device = epsilon_prev.device
    dtype = epsilon_prev.dtype
    device_str = str(device)

    def _broadcast_bridge_weight(w, batch_sz: int) -> torch.Tensor:
        """Scalar or per-batch weight column [B, 1] for loss grid broadcast."""
        if isinstance(w, torch.Tensor):
            t = w.reshape(-1).to(device=device, dtype=dtype)
            if t.numel() == 1:
                t = t.expand(batch_sz)
            elif t.numel() != batch_sz:
                raise ValueError(
                    f'Bridge weight tensor must have 1 or B={batch_sz} elements, got {t.numel()}',
                )
            return t.reshape(batch_sz, 1)
        return torch.full((batch_sz, 1), float(w), device=device, dtype=dtype)
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
        raise ValueError('close_phosphate_bridge_multi needs restype_indices (next) in geometry or _ri on next_atoms.')
    ri = ri.long()
    tc = g.get('template_tensors')
    if tc is None:
        tc = _get_template_tensors(device_str)
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
    # `sqrt(h_sq + eps^2)`: at degenerate (infeasible) triangles `h_sq=0` and `sqrt'(0) = +inf`,
    # which then propagates to NaN in upstream grads. Tiny additive eps regularises backward.
    h_val = torch.sqrt(h_sq + 1e-12)

    midpoint = o3p + a_coef.unsqueeze(-1) * axis

    n_grid = int(g.get('n_phi', _PHI_PHOS_GRID))
    ts = _phi_grid(device_str, dtype, n_grid)
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
    eps_m = dihedral_rad(
        c4p_g.reshape(-1, 3),
        c3p_g.reshape(-1, 3),
        o3p_g.reshape(-1, 3),
        p_cands.reshape(-1, 3),
    ).reshape(B, n_grid)

    ze_m = dihedral_rad(
        c3p_g.reshape(-1, 3),
        o3p_g.reshape(-1, 3),
        p_cands.reshape(-1, 3),
        o5n_g.reshape(-1, 3),
    ).reshape(B, n_grid)

    al_m = dihedral_rad(
        o3p_g.reshape(-1, 3),
        p_cands.reshape(-1, 3),
        o5n_g.reshape(-1, 3),
        c5n_g.reshape(-1, 3),
    ).reshape(B, n_grid)

    be_m = dihedral_rad(
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
        we * wrap_dihedral_diff(eps_m, epsilon_prev.unsqueeze(-1)) ** 2
        + wz * wrap_dihedral_diff(ze_m, zeta_prev.unsqueeze(-1)) ** 2
        + wa * wrap_dihedral_diff(al_m, alpha_next.unsqueeze(-1)) ** 2
        + wb * wrap_dihedral_diff(be_m, beta_next.unsqueeze(-1)) ** 2
    )

    best = loss.argmin(dim=-1)
    idx = torch.arange(B, device=device, dtype=torch.long)
    p_pick = p_cands[idx, best]

    # Infeasible: midpoint along axis (deterministic fallback)
    not_feas = ~feasible
    if not_feas.any():
        sum_l = (r_o3p + l_po5).unsqueeze(-1).clamp(min=_GEO_EPS)
        fb = o3p + (o5n - o3p) * (r_o3p.unsqueeze(-1) / sum_l)
        p_pick = torch.where(not_feas.unsqueeze(-1), fb, p_pick)

    half_pi = _scalar_constant(_HALF_PI, device_str, dtype)
    bl_op1 = _template_select(tc, 'bl_op1', ri, dtype=dtype)
    bl_op2 = _template_select(tc, 'bl_op2', ri, dtype=dtype)
    ang_op1 = _template_select(tc, 'ang_op1', ri, dtype=dtype)
    ang_op2 = _template_select(tc, 'ang_op2', ri, dtype=dtype)
    psi_op1 = _template_select(tc, 'psi_op1', ri, dtype=dtype)
    psi_op2 = _template_select(tc, 'psi_op2', ri, dtype=dtype)

    op1 = nerf_place(o3p, o5n, p_pick, bl_op1, ang_op1, psi_op1 - half_pi)
    op2 = nerf_place(o3p, o5n, p_pick, bl_op2, ang_op2, psi_op2 - half_pi)

    out = {'P': p_pick, 'OP1': op1, 'OP2': op2}
    if B == 1 and epsilon_prev.ndim == 0:
        out = {k: v.squeeze(0) for k, v in out.items()}
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

BACKBONE_ATOM_ORDER = _BACKBONE_ATOM_ORDER
BACKBONE_NAME_TO_INDEX = _BACKBONE_NAME_TO_INDEX
LOCAL_BACKBONE_ATOM_ORDER = _LOCAL_BACKBONE_ATOM_ORDER
LOCAL_BACKBONE_INDEX = _LOCAL_BACKBONE_INDEX
BRIDGE_PREV_ATOM_ORDER = _BRIDGE_PREV_ATOM_ORDER
BRIDGE_NEXT_ATOM_ORDER = _BRIDGE_NEXT_ATOM_ORDER
PHOSPHATE_ATOM_ORDER = _PHOSPHATE_ATOM_ORDER
PHOSPHATE_ATOM_INDEX = _PHOSPHATE_ATOM_INDEX


def build_backbone_from_torsions(
    torsions: 'torch.Tensor',
    tau_m: 'torch.Tensor',
    restype_indices: 'torch.Tensor',
    o3_prev_local: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Batched decoder: deterministic sugar ring closure, stereo exocyclic O3′/C5′, γ→O5′.

    Phosphate uses ``close_phosphate_bridge_multi``; without prior-nucleotide ε/ζ in this API,
    only α and β targets are weighted (incoming bridge from o3_prev in target frame).
    """
    ri = restype_indices.long()
    geometry = {'template_tensors': _get_template_tensors(str(torsions.device))}
    chi = torsions[:, TOR_CHI]
    P = torsions[:, TOR_PSEUDOROTATION_PHASE]
    gamma = torsions[:, TOR_GAMMA]
    delta = torsions[:, TOR_DELTA]
    alpha = torsions[:, TOR_ALPHA]
    beta = torsions[:, TOR_BETA]
    eps_lat = torsions[:, TOR_EPS]
    zet_lat = torsions[:, TOR_ZETA]
    ring = build_sugar_ring_closed_form(chi, P, tau_m, ri, geometry=geometry)
    atoms = add_exocyclic_sugar_atoms(ring, delta, restype_indices=ri, geometry=geometry)
    atoms = add_o5_from_gamma(atoms, gamma, restype_indices=ri, geometry=geometry)

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
    phosph = close_phosphate_bridge_multi(
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
    # Place OP1/OP2 using predicted eta_p = dihedral(C5', O5', P, OP1)
    # instead of the fixed canonical template dihedral.
    eta_p = torsions[:, TOR_ETA_P]
    tc = geometry['template_tensors']
    p_local = phosph['P']
    c5n = out["C5'"]
    o5n = out["O5'"]
    half_pi = torsions.new_tensor(_HALF_PI)
    bl_op1 = _template_select(tc, 'bl_op1', ri, dtype=torsions.dtype)
    bl_op2 = _template_select(tc, 'bl_op2', ri, dtype=torsions.dtype)
    ang_op1 = _template_select(tc, 'ang_op1', ri, dtype=torsions.dtype)
    ang_op2 = _template_select(tc, 'ang_op2', ri, dtype=torsions.dtype)
    psi_op1_c5 = _template_select(tc, 'psi_op1_c5', ri, dtype=torsions.dtype)
    psi_op2_c5 = _template_select(tc, 'psi_op2_c5', ri, dtype=torsions.dtype)
    delta_op_c5 = psi_op2_c5 - psi_op1_c5
    out['OP1'] = nerf_place(c5n, o5n, p_local, bl_op1, ang_op1, eta_p - half_pi)
    out['OP2'] = nerf_place(c5n, o5n, p_local, bl_op2, ang_op2, eta_p + delta_op_c5 - half_pi)
    return out


def build_batch_window_backbone_from_torsions(
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
    if torsions.dim() != 3 or torsions.shape[-1] != N_TORSIONS:
        raise ValueError(
            f'Expected torsions [B,W,{N_TORSIONS}], got {tuple(torsions.shape)}',
        )
    B, W, _ = torsions.shape
    device = torsions.device
    dtype = torsions.dtype
    n_bb = len(_BACKBONE_ATOM_ORDER)
    BW = B * W
    phosphate_index = torch.tensor(_PHOSPHATE_ATOM_INDEX, device=device, dtype=torch.long)

    flat_t = torsions.reshape(BW, N_TORSIONS)
    flat_tm = tau_m.reshape(BW)
    flat_ri = restype_indices.reshape(BW).long()
    geometry = {
        'template_tensors': _get_template_tensors(str(device)),
    }
    chi = flat_t[:, TOR_CHI]
    P_rad = flat_t[:, TOR_PSEUDOROTATION_PHASE]
    ring = build_sugar_ring_closed_form(chi, P_rad, flat_tm, flat_ri, geometry=geometry)
    atoms = add_exocyclic_sugar_atoms(
        ring,
        flat_t[:, TOR_DELTA],
        restype_indices=flat_ri,
        geometry=geometry,
    )
    atoms = add_o5_from_gamma(
        atoms, flat_t[:, TOR_GAMMA], restype_indices=flat_ri, geometry=geometry,
    )

    Ri_flat = nt_frames_world.reshape(BW, 3, 3)
    ok_flat = nt_origins_world.reshape(BW, 3)

    bb_flat = torch.full((BW, n_bb, 3), float('nan'), device=device, dtype=dtype)
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
            we = torch.ones(nv, device=device, dtype=dtype)
            wz = torch.ones(nv, device=device, dtype=dtype)
            wa = torch.ones(nv, device=device, dtype=dtype)
            wb = torch.ones(nv, device=device, dtype=dtype)
        else:
            we = torsion_mask[idx, k - 1, TOR_EPS].to(dtype=dtype)
            wz = torsion_mask[idx, k - 1, TOR_ZETA].to(dtype=dtype)
            wa = torsion_mask[idx, k, TOR_ALPHA].to(dtype=dtype)
            wb = torsion_mask[idx, k, TOR_BETA].to(dtype=dtype)

        tc_bb = geometry['template_tensors']
        ri_k = restype_indices[idx, k]
        phosph = close_phosphate_bridge_multi(
            prev_loc,
            next_loc,
            torsions[idx, k - 1, TOR_EPS],
            torsions[idx, k - 1, TOR_ZETA],
            torsions[idx, k, TOR_ALPHA],
            torsions[idx, k, TOR_BETA],
            geometry={
                'restype_indices_next': ri_k,
                'template_tensors': tc_bb,
            },
            weight_epsilon=we,
            weight_zeta=wz,
            weight_alpha=wa,
            weight_beta=wb,
        )
        # Re-place OP1/OP2 using predicted eta_p = dihedral(C5', O5', P, OP1).
        eta_p_k = torsions[idx, k, TOR_ETA_P]
        half_pi_k = torsions.new_tensor(_HALF_PI)
        psi_op1_c5_k = _template_select(tc_bb, 'psi_op1_c5', ri_k, dtype=dtype)
        psi_op2_c5_k = _template_select(tc_bb, 'psi_op2_c5', ri_k, dtype=dtype)
        delta_op_c5_k = psi_op2_c5_k - psi_op1_c5_k
        phosph['OP1'] = nerf_place(
            next_loc["C5'"], next_loc["O5'"], phosph['P'],
            _template_select(tc_bb, 'bl_op1', ri_k, dtype=dtype),
            _template_select(tc_bb, 'ang_op1', ri_k, dtype=dtype),
            eta_p_k - half_pi_k,
        )
        phosph['OP2'] = nerf_place(
            next_loc["C5'"], next_loc["O5'"], phosph['P'],
            _template_select(tc_bb, 'bl_op2', ri_k, dtype=dtype),
            _template_select(tc_bb, 'ang_op2', ri_k, dtype=dtype),
            eta_p_k + delta_op_c5_k - half_pi_k,
        )
        phosph_local = torch.stack([phosph[nm] for nm in _PHOSPHATE_ATOM_ORDER], dim=1)
        bb[idx.unsqueeze(-1), k, phosphate_index] = local_to_world_points(phosph_local, ok, Rk)

    return bb


def build_window_backbone_from_torsions(
    torsions: torch.Tensor,
    tau_m: torch.Tensor,
    restype_indices: torch.Tensor,
    nt_origins_world: torch.Tensor,
    nt_frames_world: torch.Tensor,
    torsion_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """World backbone [W, n_bb, 3] with the same atom order as ``base2backbone.schema.BACKBONE_ATOMS``.

    For each k>=1, place P_k / OP1 / OP2 using ε,ζ from residue k−1 and α,β from k (masked).
    Bridge geometry uses k−1 world O3′,C3′,C4′ expressed in nucleotide k local frame.
    """
    W = int(torsions.shape[0])
    Ri = nt_frames_world
    if Ri.dim() == 2:
        Ri = Ri.view(W, 3, 3)
    bb_b = build_batch_window_backbone_from_torsions(
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
    ``base2backbone.schema.BACKBONE_ATOMS`` / ``_BACKBONE_ATOM_ORDER``). For B>1 batches use
    ``build_batch_window_backbone_from_torsions`` directly.
    """
    if theta.dim() == 2:
        if tau_m.dim() != 1 or restype_indices.dim() != 1:
            raise ValueError(
                f'Expected tau_m [N], restype [N] with theta [N,{N_TORSIONS}]; '
                f'got tau {tuple(tau_m.shape)}, ri {tuple(restype_indices.shape)}',
            )
        if nt_origins_world.shape[0] != theta.shape[0] or nt_frames_world.shape[0] != theta.shape[0]:
            raise ValueError('nt_origins_world / nt_frames_world must have length N matching theta')
        return build_window_backbone_from_torsions(
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
                'use build_batch_window_backbone_from_torsions for B>1',
            )
        tm = tau_m.squeeze(0) if tau_m.dim() == 2 else tau_m
        ri = restype_indices.squeeze(0) if restype_indices.dim() == 2 else restype_indices
        oo = nt_origins_world.squeeze(0) if nt_origins_world.dim() == 3 else nt_origins_world
        rr = nt_frames_world.squeeze(0) if nt_frames_world.dim() == 4 else nt_frames_world
        m = torsion_mask.squeeze(0) if torsion_mask is not None and torsion_mask.dim() == 3 else torsion_mask
        return build_window_backbone_from_torsions(
            theta.squeeze(0), tm, ri, oo, rr, m,
        )
    raise ValueError(f'theta must be [N, {N_TORSIONS}] or [1, N, {N_TORSIONS}], got {tuple(theta.shape)}')


def build_backbone_local(
    torsions: np.ndarray,
    restype: str,
    *,
    o3_prev_local: Optional[np.ndarray] = None,
    tau_m: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Single residue on CPU: numpy torsion vector → numpy atom dict (scripts / tests)."""
    from base2backbone.geometry.templates import get_template_tau_m

    ri = torch.tensor([{'A': 0, 'C': 1, 'G': 2, 'T': 3}[restype]], dtype=torch.long)
    tm_val = float(tau_m) if tau_m is not None else float(get_template_tau_m(restype))
    theta = torch.from_numpy(np.asarray(torsions, dtype=np.float64)).unsqueeze(0).float()
    tm = torch.tensor([tm_val], dtype=torch.float32)
    o3_t = None
    if o3_prev_local is not None:
        o3_t = torch.from_numpy(
            np.asarray(o3_prev_local, dtype=np.float64).reshape(1, 3),
        ).float()
    bb_t = build_backbone_from_torsions(theta, tm, ri, o3_prev_local=o3_t)
    return {k: v.squeeze(0).detach().cpu().numpy() for k, v in bb_t.items()}
