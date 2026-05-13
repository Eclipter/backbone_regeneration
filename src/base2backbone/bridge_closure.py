"""Phosphate bridge closure loss: O3′_i – P_{i+1} – O5′_{i+1} (bonds, angles, wrapped torsions)."""

import functools
import math
from typing import Optional

import numpy as np
import torch

from .data import BACKBONE_ATOMS
from .geometry import (
    bond_angle,
    dihedral_rad,
    get_template,
    get_template_tensors,
    wrap_dihedral_diff,
)
from .geometry.primitives import _bond_angle
from .torsion_constants import (
    TOR_ALPHA,
    TOR_BETA,
    TOR_EPS,
    TOR_ZETA,
)

_BASE_LETTERS = ('A', 'C', 'G', 'T')

# Geometry hyperparameters (single source; training reads overrides via ``geometry`` dict).
CLOSURE_SIGMA_BOND_A = 0.035
CLOSURE_SIGMA_ANGLE_RAD = math.radians(4.0)
CLOSURE_SIGMA_TORSION_RAD = 0.35
CLOSURE_FAIL_THRESHOLD_BOND_SIGMA = 3.0
CLOSURE_FAIL_THRESHOLD_ANGLE_SIGMA = 3.0
CLOSURE_FAIL_THRESHOLD_TORSION_SIGMA = 3.0

_PAIR_BRIDGE_ANGLE_CACHE: Optional[np.ndarray] = None


def canonical_two_residue_bridge_positions(rest_prev: str, rest_next: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Prev / next backbone for a contiguous pair (prev O3' at origin).

    Next residue is placed as a rigid translate of ``rest_next`` aligned so ``P_next``
    sits on the ``O3'_{prev}-P_bridge`` chord at template ``bond_p_o3_inter[rest_next]``.
    This preserves intra-residue P– O5′ and sugar bond lengths unlike rescaling ``P``
    alone, which inflated ``bond_p_o5`` mismatch in closure loss fixtures.
    """
    tp = get_template(rest_prev)
    tn_raw = get_template(rest_next)
    o3_prev = np.asarray(tp["O3'"], dtype=np.float64)
    prev = {k: np.asarray(tp[k], dtype=np.float64) - o3_prev for k in BACKBONE_ATOMS if k in tp}

    _ri = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    inc = _ri[rest_next]
    tt = get_template_tensors('cpu')
    bond_o3_p = float(tt['bond_p_o3_inter'][inc].numpy())

    vp = np.asarray(tp['P'], dtype=np.float64) - o3_prev
    vn = vp / (np.linalg.norm(vp) + 1e-12)
    p_tgt = prev["O3'"] + vn * bond_o3_p

    p_tpl_next = np.asarray(tn_raw['P'], dtype=np.float64)
    nxt = {
        nm: np.asarray(tn_raw[nm], dtype=np.float64) - p_tpl_next + p_tgt
        for nm in BACKBONE_ATOMS
        if nm in tn_raw
    }
    return prev, nxt


def paired_bridge_corner_angles(rest_prev: str, rest_next: str) -> tuple[float, float, float]:
    """Three bridge-angle targets at residue boundary (prev → next), radians.

    Must match angles in ``compute_bridge_closure_loss``::
        ∠(C3'ᵢ,O3'ᵢ,Pᵢ₊₁), ∠(O3'ᵢ,Pᵢ₊₁,O5'ᵢ₊₁), ∠(Pᵢ₊₁,O5'ᵢ₊₁,C5'ᵢ₊₁).
    These differ from naive single-template corners that use intra-residue C3–O3–P_same.
    """
    prev, nxt = canonical_two_residue_bridge_positions(rest_prev, rest_next)
    c3_p = prev["C3'"]
    o3_p = prev["O3'"]
    p_n = nxt['P']
    o5_n = nxt["O5'"]
    c5_n = nxt["C5'"]
    return (
        _bond_angle(c3_p, o3_p, p_n),
        _bond_angle(o3_p, p_n, o5_n),
        _bond_angle(p_n, o5_n, c5_n),
    )


def _paired_bridge_corner_angles_lookup() -> np.ndarray:
    """``[4, 4, 3]``: prev restype × next restype × (angle1, angle2, angle3) in radians."""
    global _PAIR_BRIDGE_ANGLE_CACHE
    if _PAIR_BRIDGE_ANGLE_CACHE is not None:
        return _PAIR_BRIDGE_ANGLE_CACHE

    tbl = np.zeros((4, 4, 3), dtype=np.float64)
    for ip, rp in enumerate(_BASE_LETTERS):
        for jn, rn in enumerate(_BASE_LETTERS):
            a1, a2, a3 = paired_bridge_corner_angles(rp, rn)
            tbl[ip, jn, 0] = a1
            tbl[ip, jn, 1] = a2
            tbl[ip, jn, 2] = a3
    _PAIR_BRIDGE_ANGLE_CACHE = tbl
    return tbl


@functools.lru_cache(maxsize=None)
def _paired_bridge_corner_angles(
    device_str: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(
        _paired_bridge_corner_angles_lookup(),
        device=torch.device(device_str),
        dtype=dtype,
    )


def canonical_two_residue_bridge_bb_tensor(
        rest_prev: str = 'A',
        rest_next: str = 'A',
        *,
        dtype=torch.float64,
        device: torch.device | str = 'cpu',
) -> torch.Tensor:
    """Aligned with ``tests.test_bridge_closure_loss._ideal_bridge_bb_and_targets`` layout ``[2, n_bb, 3]``."""
    device = device if isinstance(device, torch.device) else torch.device(device)
    prev, nxt = canonical_two_residue_bridge_positions(rest_prev, rest_next)
    name_to_j = {nm: j for j, nm in enumerate(BACKBONE_ATOMS)}
    bb = torch.full((2, len(BACKBONE_ATOMS), 3), float('nan'), dtype=dtype, device=device)
    for nm, j in name_to_j.items():
        if nm in prev:
            bb[0, j] = torch.as_tensor(prev[nm], dtype=dtype, device=device)
        if nm in nxt:
            bb[1, j] = torch.as_tensor(nxt[nm], dtype=dtype, device=device)
    return bb


def compute_bridge_closure_loss(
    bb_xyz_world: torch.Tensor,
    target_torsions: torch.Tensor,
    torsion_mask: torch.Tensor,
    restype_indices: torch.Tensor,
    valid_nt_mask: Optional[torch.Tensor] = None,
    same_chain_mask: Optional[torch.Tensor] = None,
    valid_pair_mask: Optional[torch.Tensor] = None,
    valid_bridge_mask: Optional[torch.Tensor] = None,
    *,
    geometry: Optional[dict] = None,
    weights: Optional[dict] = None,
    grad_prop_tensor: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Bridge-level closure loss for consecutive pairs along sequence dimension.

    Parameters
    ----------
    bb_xyz_world
        ``[B, W, n_bb, 3]`` world coordinates (order matches ``base2backbone.data.BACKBONE_ATOMS``).
    target_torsions
        ``[B, W, N_TORSIONS]`` reference torsions (ε, ζ on residue i; α, β on i+1).
    torsion_mask
        ``[B, W, N_TORSIONS]`` boolean observability mask.
    restype_indices
        ``[B, W]`` residue-type indices into A/C/G/T templates.
    valid_nt_mask
        Unused API parameter. Prefer ``valid_pair_mask`` / ``valid_bridge_mask``.
    same_chain_mask
        Optional ``[B, W-1]`` mask for consecutive pairs. If ``None``, adjacent positions
        are treated as same-chain neighbors (sliding windows are contiguous).
    valid_pair_mask
        Optional ``[B, W-1]``. If set, only these candidate bridges are counted toward
        loss and ``closure_valid_bridge_fraction`` (e.g. target-adjacent pairs).
    valid_bridge_mask
        Optional ``[B, W-1]``. When set, ``bridge_ok = valid_bridge_mask & (torsions & atoms)``
        — use for explicit bridge gating instead of blanket per-nucleotide masks.

    Returns
    -------
    dict
        ``closure_loss``, component losses, ``closure_valid_bridge_fraction``,
        ``closure_num_valid_bridges``, ``closure_fail_rate``, and MAE metrics (Å / deg).
    """
    g = geometry or {}
    w = weights or {}
    wb = float(w.get('bond', 1.0))
    wa = float(w.get('angle', 1.0))
    wt = float(w.get('torsion', 1.0))

    sigma_d = float(g.get('sigma_bond', CLOSURE_SIGMA_BOND_A))
    sigma_a = float(g.get('sigma_angle_rad', CLOSURE_SIGMA_ANGLE_RAD))
    sigma_t = float(g.get('sigma_torsion_rad', CLOSURE_SIGMA_TORSION_RAD))
    thr_b = float(g.get('fail_threshold_bond_sigma', CLOSURE_FAIL_THRESHOLD_BOND_SIGMA))
    thr_a = float(g.get('fail_threshold_angle_sigma', CLOSURE_FAIL_THRESHOLD_ANGLE_SIGMA))
    thr_t = float(g.get('fail_threshold_torsion_sigma', CLOSURE_FAIL_THRESHOLD_TORSION_SIGMA))

    eps = float(g.get('eps', 1e-8))
    device = bb_xyz_world.device
    dtype = bb_xyz_world.dtype

    def _zero() -> torch.Tensor:
        if grad_prop_tensor is not None:
            return grad_prop_tensor.sum() * 0.0
        return torch.zeros((), device=device, dtype=dtype)

    if bb_xyz_world.dim() != 4:
        raise ValueError(f'Expected bb_xyz_world [B,W,n_bb,3], got {tuple(bb_xyz_world.shape)}')
    B, W, n_bb, _ = bb_xyz_world.shape
    if W < 2:
        z = _zero()
        zf = torch.zeros((), device=device, dtype=dtype)
        return {
            'closure_loss': z,
            'closure_bond_loss': z,
            'closure_angle_loss': z,
            'closure_torsion_loss': z,
            'closure_valid_bridge_fraction': zf,
            'closure_num_valid_bridges': zf,
            'closure_fail_rate': zf,
            'bridge_bond_mae': zf,
            'bridge_angle_mae_deg': zf,
            'bridge_torsion_mae_deg': zf,
        }

    name_to_j = {nm: j for j, nm in enumerate(BACKBONE_ATOMS)}
    j_c4, j_c3, j_o3 = name_to_j["C4'"], name_to_j["C3'"], name_to_j["O3'"]
    j_p, j_o5, j_c5 = name_to_j['P'], name_to_j["O5'"], name_to_j["C5'"]

    _ = valid_nt_mask

    bb = bb_xyz_world
    prev_tm = torsion_mask[:, :-1]
    curr_tm = torsion_mask[:, 1:]
    tgt_prev = target_torsions[:, :-1]
    tgt_curr = target_torsions[:, 1:]

    pair_gate = torch.ones(B, W - 1, dtype=torch.bool, device=device)
    if same_chain_mask is not None:
        pair_gate = pair_gate & same_chain_mask
    if valid_pair_mask is not None:
        if valid_pair_mask.shape != (B, W - 1):
            raise ValueError(
                f'valid_pair_mask must be [B, W-1]=[{B}, {W - 1}], got {tuple(valid_pair_mask.shape)}',
            )
        pair_gate = pair_gate & valid_pair_mask

    tor_obs = (
        prev_tm[..., TOR_EPS]
        & prev_tm[..., TOR_ZETA]
        & curr_tm[..., TOR_ALPHA]
        & curr_tm[..., TOR_BETA]
    )
    tor_fin = (
        torch.isfinite(tgt_prev[..., TOR_EPS])
        & torch.isfinite(tgt_prev[..., TOR_ZETA])
        & torch.isfinite(tgt_curr[..., TOR_ALPHA])
        & torch.isfinite(tgt_curr[..., TOR_BETA])
    )

    atoms_fin = (
        torch.isfinite(bb[:, :-1, j_c4]).all(dim=-1)
        & torch.isfinite(bb[:, :-1, j_c3]).all(dim=-1)
        & torch.isfinite(bb[:, :-1, j_o3]).all(dim=-1)
        & torch.isfinite(bb[:, 1:, j_p]).all(dim=-1)
        & torch.isfinite(bb[:, 1:, j_o5]).all(dim=-1)
        & torch.isfinite(bb[:, 1:, j_c5]).all(dim=-1)
        & torch.isfinite(bb[:, 1:, j_c4]).all(dim=-1)
    )

    if valid_bridge_mask is not None:
        if valid_bridge_mask.shape != (B, W - 1):
            raise ValueError(
                f'valid_bridge_mask must be [B, W-1]=[{B}, {W - 1}], got {tuple(valid_bridge_mask.shape)}',
            )
        bridge_gate = valid_bridge_mask
    else:
        bridge_gate = pair_gate

    bridge_ok = bridge_gate & tor_obs & tor_fin & atoms_fin

    c4_p = bb[:, :-1, j_c4]
    c3_p = bb[:, :-1, j_c3]
    o3_p = bb[:, :-1, j_o3]
    p_n = bb[:, 1:, j_p]
    o5_n = bb[:, 1:, j_o5]
    c5_n = bb[:, 1:, j_c5]
    c4_n = bb[:, 1:, j_c4]

    ri_next = restype_indices[:, 1:].long()
    ri_prev = restype_indices[:, :-1].long()

    tc = get_template_tensors(str(device))
    d0_o3p = tc['bond_p_o3_inter'][ri_next]
    d0_po5 = tc['bond_p_o5'][ri_next]

    d1 = (p_n - o3_p).norm(dim=-1)
    d2 = (o5_n - p_n).norm(dim=-1)
    bond_sq = ((d1 - d0_o3p) / (sigma_d + eps)) ** 2 + ((d2 - d0_po5) / (sigma_d + eps)) ** 2

    ang_lut = _paired_bridge_corner_angles(str(device), dtype)

    ar_angle1 = ang_lut[ri_prev.long(), ri_next.long(), 0]
    ar_angle2 = ang_lut[ri_prev.long(), ri_next.long(), 1]
    ar_angle3 = ang_lut[ri_prev.long(), ri_next.long(), 2]

    a1 = bond_angle(c3_p, o3_p, p_n, eps)
    a2 = bond_angle(o3_p, p_n, o5_n, eps)
    a3 = bond_angle(p_n, o5_n, c5_n, eps)
    angle_sq = (
        ((a1 - ar_angle1) / (sigma_a + eps)) ** 2
        + ((a2 - ar_angle2) / (sigma_a + eps)) ** 2
        + ((a3 - ar_angle3) / (sigma_a + eps)) ** 2
    )

    eps_pred = dihedral_rad(c4_p, c3_p, o3_p, p_n)
    ze_pred = dihedral_rad(c3_p, o3_p, p_n, o5_n)
    al_pred = dihedral_rad(o3_p, p_n, o5_n, c5_n)
    be_pred = dihedral_rad(p_n, o5_n, c5_n, c4_n)

    e_eps = wrap_dihedral_diff(eps_pred, tgt_prev[..., TOR_EPS])
    e_ze = wrap_dihedral_diff(ze_pred, tgt_prev[..., TOR_ZETA])
    e_al = wrap_dihedral_diff(al_pred, tgt_curr[..., TOR_ALPHA])
    e_be = wrap_dihedral_diff(be_pred, tgt_curr[..., TOR_BETA])

    torsion_sq = (
        (e_eps / (sigma_t + eps)) ** 2
        + (e_ze / (sigma_t + eps)) ** 2
        + (e_al / (sigma_t + eps)) ** 2
        + (e_be / (sigma_t + eps)) ** 2
    )

    if valid_pair_mask is not None:
        n_br = float(max(valid_pair_mask.float().sum().item(), 1.0))
    else:
        n_br = float(max(B * (W - 1), 1))
    n_ok = bridge_ok.float().sum().clamp(min=0.0)

    bond_norm = torch.maximum(
        (d1 - d0_o3p).abs() / (sigma_d + eps),
        (d2 - d0_po5).abs() / (sigma_d + eps),
    )
    angle_norm = torch.maximum(
        torch.maximum(
            (a1 - ar_angle1).abs() / (sigma_a + eps),
            (a2 - ar_angle2).abs() / (sigma_a + eps),
        ),
        (a3 - ar_angle3).abs() / (sigma_a + eps),
    )
    tor_abs = torch.maximum(
        torch.maximum(e_eps.abs(), e_ze.abs()),
        torch.maximum(e_al.abs(), e_be.abs()),
    )
    torsion_norm = tor_abs / (sigma_t + eps)

    viol = (
        (bond_norm > thr_b)
        | (angle_norm > thr_a)
        | (torsion_norm > thr_t)
    ) & bridge_ok

    if not bridge_ok.any():
        z = _zero()
        zf = torch.zeros((), device=device, dtype=dtype)
        return {
            'closure_loss': z,
            'closure_bond_loss': z,
            'closure_angle_loss': z,
            'closure_torsion_loss': z,
            'closure_valid_bridge_fraction': zf,
            'closure_num_valid_bridges': zf,
            'closure_fail_rate': zf,
            'bridge_bond_mae': zf,
            'bridge_angle_mae_deg': zf,
            'bridge_torsion_mae_deg': zf,
        }

    m = bridge_ok.float()
    bond_mean = (bond_sq * m).sum() / (m.sum() + eps)
    angle_mean = (angle_sq * m).sum() / (m.sum() + eps)
    torsion_mean = (torsion_sq * m).sum() / (m.sum() + eps)

    closure = wb * bond_mean + wa * angle_mean + wt * torsion_mean

    bond_mae = ((d1 - d0_o3p).abs() + (d2 - d0_po5).abs()) * 0.5
    ang_mae = (a1 - ar_angle1).abs() + (a2 - ar_angle2).abs() + (a3 - ar_angle3).abs()
    ang_mae = ang_mae / 3.0
    tor_mae = (e_eps.abs() + e_ze.abs() + e_al.abs() + e_be.abs()) / 4.0

    valid_frac = n_ok / bb_xyz_world.new_tensor(n_br)
    fail_rate = viol.float().sum() / (n_ok + eps)

    rad2deg = torch.tensor(180.0 / math.pi, device=device, dtype=dtype)

    return {
        'closure_loss': closure,
        'closure_bond_loss': bond_mean,
        'closure_angle_loss': angle_mean,
        'closure_torsion_loss': torsion_mean,
        'closure_valid_bridge_fraction': valid_frac,
        'closure_num_valid_bridges': n_ok,
        'closure_fail_rate': fail_rate,
        'bridge_bond_mae': (bond_mae * m).sum() / (m.sum() + eps),
        'bridge_angle_mae_deg': ((ang_mae * m).sum() / (m.sum() + eps)) * rad2deg,
        'bridge_torsion_mae_deg': ((tor_mae * m).sum() / (m.sum() + eps)) * rad2deg,
    }
