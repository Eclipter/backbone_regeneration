"""Phosphate bridge closure loss: O3′_i – P_{i+1} – O5′_{i+1} (bonds, angles, wrapped torsions)."""

import math
from typing import Optional

import numpy as np
import torch

from torsion_geometry import (
    TOR_ALPHA,
    TOR_BETA,
    TOR_EPS,
    TOR_ZETA,
    _BACKBONE_ATOM_ORDER,
    _get_template,
    _get_template_tensors,
    dihedral_rad_torch,
    wrap_dihedral_diff_torch,
)
from utils import backbone_atoms

# Geometry hyperparameters (single source; training reads overrides via ``geometry`` dict).
CLOSURE_SIGMA_BOND_A = 0.035
CLOSURE_SIGMA_ANGLE_RAD = math.radians(4.0)
CLOSURE_SIGMA_TORSION_RAD = 0.35
CLOSURE_FAIL_THRESHOLD_BOND_SIGMA = 3.0
CLOSURE_FAIL_THRESHOLD_ANGLE_SIGMA = 3.0
CLOSURE_FAIL_THRESHOLD_TORSION_SIGMA = 3.0

_BRIDGE_ANGLE_REF_CACHE: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None


def _bridge_template_angle_refs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference angles (rad) per restype index 0..3 from canonical templates."""
    global _BRIDGE_ANGLE_REF_CACHE
    if _BRIDGE_ANGLE_REF_CACHE is not None:
        return _BRIDGE_ANGLE_REF_CACHE

    def _ang(tpl: dict, a: str, b: str, c: str) -> float:
        va = tpl[a] - tpl[b]
        vc = tpl[c] - tpl[b]
        la = float(np.linalg.norm(va)) + 1e-12
        lc = float(np.linalg.norm(vc)) + 1e-12
        cos_t = float(np.dot(va, vc) / (la * lc))
        cos_t = float(np.clip(cos_t, -1.0 + 1e-9, 1.0 - 1e-9))
        return float(np.arccos(cos_t))

    c3_o3_p, o3_p_o5, p_o5_c5 = [], [], []
    for rt in ('A', 'C', 'G', 'T'):
        tpl = _get_template(rt)
        c3_o3_p.append(_ang(tpl, "C3'", "O3'", 'P'))
        o3_p_o5.append(_ang(tpl, "O3'", 'P', "O5'"))
        p_o5_c5.append(_ang(tpl, 'P', "O5'", "C5'"))
    _BRIDGE_ANGLE_REF_CACHE = (
        np.asarray(c3_o3_p, dtype=np.float64),
        np.asarray(o3_p_o5, dtype=np.float64),
        np.asarray(p_o5_c5, dtype=np.float64),
    )
    return _BRIDGE_ANGLE_REF_CACHE


def _bond_angle_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, eps: float) -> torch.Tensor:
    """Interior angle ∠(a–b–c) in radians; batched last dim 3."""
    ba = a - b
    bc = c - b
    denom = ba.norm(dim=-1) * bc.norm(dim=-1) + eps
    cos_t = (ba * bc).sum(dim=-1) / denom
    cos_t = cos_t.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos_t)


def compute_bridge_closure_loss(
    bb_xyz_world: torch.Tensor,
    target_torsions: torch.Tensor,
    torsion_mask: torch.Tensor,
    valid_nt_mask: torch.Tensor,
    restype_indices: torch.Tensor,
    same_chain_mask: Optional[torch.Tensor] = None,
    valid_pair_mask: Optional[torch.Tensor] = None,
    *,
    geometry: Optional[dict] = None,
    weights: Optional[dict] = None,
    grad_prop_tensor: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Bridge-level closure loss for consecutive pairs along sequence dimension.

    Parameters
    ----------
    bb_xyz_world
        ``[B, W, n_bb, 3]`` world coordinates (order matches ``utils.backbone_atoms``).
    target_torsions
        ``[B, W, N_TORSIONS]`` reference torsions (ε, ζ on residue i; α, β on i+1).
    torsion_mask
        ``[B, W, N_TORSIONS]`` boolean observability mask.
    valid_nt_mask
        ``[B, W]`` True when a residue may participate in a bridge (non-terminal, etc.).
    same_chain_mask
        Optional ``[B, W-1]`` mask for consecutive pairs. If ``None``, adjacent positions
        are treated as same-chain neighbors (sliding windows are contiguous).
    valid_pair_mask
        Optional ``[B, W-1]``. If set, only these candidate bridges are counted toward
        loss and ``closure_valid_bridge_fraction`` (e.g. target-adjacent pairs).

    Returns
    -------
    dict
        ``closure_loss``, component losses, ``closure_valid_bridge_fraction``,
        ``closure_fail_rate``, and MAE metrics (Å / deg).
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
    dev = bb_xyz_world.device
    dtype = bb_xyz_world.dtype

    def _zero() -> torch.Tensor:
        if grad_prop_tensor is not None:
            return grad_prop_tensor.sum() * 0.0
        return torch.zeros((), device=dev, dtype=dtype)

    if bb_xyz_world.dim() != 4:
        raise ValueError(f'Expected bb_xyz_world [B,W,n_bb,3], got {tuple(bb_xyz_world.shape)}')
    B, W, n_bb, _ = bb_xyz_world.shape
    if W < 2:
        z = _zero()
        zf = torch.zeros((), device=dev, dtype=dtype)
        return {
            'closure_loss': z,
            'closure_bond_loss': z,
            'closure_angle_loss': z,
            'closure_torsion_loss': z,
            'closure_valid_bridge_fraction': zf,
            'closure_fail_rate': zf,
            'bridge_bond_mae': zf,
            'bridge_angle_mae_deg': zf,
            'bridge_torsion_mae_deg': zf,
        }

    assert tuple(backbone_atoms) == _BACKBONE_ATOM_ORDER
    name_to_j = {nm: j for j, nm in enumerate(backbone_atoms)}
    j_c4, j_c3, j_o3 = name_to_j["C4'"], name_to_j["C3'"], name_to_j["O3'"]
    j_p, j_o5, j_c5 = name_to_j['P'], name_to_j["O5'"], name_to_j["C5'"]

    bb = bb_xyz_world
    prev_tm = torsion_mask[:, :-1]
    curr_tm = torsion_mask[:, 1:]
    tgt_prev = target_torsions[:, :-1]
    tgt_curr = target_torsions[:, 1:]

    pair_valid = valid_nt_mask[:, :-1] & valid_nt_mask[:, 1:]
    if same_chain_mask is not None:
        pair_valid = pair_valid & same_chain_mask
    if valid_pair_mask is not None:
        if valid_pair_mask.shape != (B, W - 1):
            raise ValueError(
                f'valid_pair_mask must be [B, W-1]=[{B}, {W - 1}], got {tuple(valid_pair_mask.shape)}',
            )
        pair_valid = pair_valid & valid_pair_mask

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

    bridge_ok = pair_valid & tor_obs & tor_fin & atoms_fin

    c4_p = bb[:, :-1, j_c4]
    c3_p = bb[:, :-1, j_c3]
    o3_p = bb[:, :-1, j_o3]
    p_n = bb[:, 1:, j_p]
    o5_n = bb[:, 1:, j_o5]
    c5_n = bb[:, 1:, j_c5]
    c4_n = bb[:, 1:, j_c4]

    ri_next = restype_indices[:, 1:].long()
    ri_prev = restype_indices[:, :-1].long()

    tc = _get_template_tensors(str(dev))
    d0_o3p = tc['bl_o3_p'][ri_next]
    d0_po5 = tc['r_po3'][ri_next]

    d1 = (p_n - o3_p).norm(dim=-1)
    d2 = (o5_n - p_n).norm(dim=-1)
    bond_sq = ((d1 - d0_o3p) / (sigma_d + eps)) ** 2 + ((d2 - d0_po5) / (sigma_d + eps)) ** 2

    ang_ref_c3, ang_ref_o3p, ang_ref_po5 = _bridge_template_angle_refs()
    ar_c3 = torch.as_tensor(ang_ref_c3, device=dev, dtype=dtype)[ri_prev]
    ar_o3 = torch.as_tensor(ang_ref_o3p, device=dev, dtype=dtype)[ri_next]
    ar_po5 = torch.as_tensor(ang_ref_po5, device=dev, dtype=dtype)[ri_next]

    a1 = _bond_angle_torch(c3_p, o3_p, p_n, eps)
    a2 = _bond_angle_torch(o3_p, p_n, o5_n, eps)
    a3 = _bond_angle_torch(p_n, o5_n, c5_n, eps)
    angle_sq = (
        ((a1 - ar_c3) / (sigma_a + eps)) ** 2
        + ((a2 - ar_o3) / (sigma_a + eps)) ** 2
        + ((a3 - ar_po5) / (sigma_a + eps)) ** 2
    )

    eps_pred = dihedral_rad_torch(c4_p, c3_p, o3_p, p_n)
    ze_pred = dihedral_rad_torch(c3_p, o3_p, p_n, o5_n)
    al_pred = dihedral_rad_torch(o3_p, p_n, o5_n, c5_n)
    be_pred = dihedral_rad_torch(p_n, o5_n, c5_n, c4_n)

    e_eps = wrap_dihedral_diff_torch(eps_pred, tgt_prev[..., TOR_EPS])
    e_ze = wrap_dihedral_diff_torch(ze_pred, tgt_prev[..., TOR_ZETA])
    e_al = wrap_dihedral_diff_torch(al_pred, tgt_curr[..., TOR_ALPHA])
    e_be = wrap_dihedral_diff_torch(be_pred, tgt_curr[..., TOR_BETA])

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
            (a1 - ar_c3).abs() / (sigma_a + eps),
            (a2 - ar_o3).abs() / (sigma_a + eps),
        ),
        (a3 - ar_po5).abs() / (sigma_a + eps),
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
        zf = torch.zeros((), device=dev, dtype=dtype)
        return {
            'closure_loss': z,
            'closure_bond_loss': z,
            'closure_angle_loss': z,
            'closure_torsion_loss': z,
            'closure_valid_bridge_fraction': zf,
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
    ang_mae = (a1 - ar_c3).abs() + (a2 - ar_o3).abs() + (a3 - ar_po5).abs()
    ang_mae = ang_mae / 3.0
    tor_mae = (e_eps.abs() + e_ze.abs() + e_al.abs() + e_be.abs()) / 4.0

    valid_frac = n_ok / bb_xyz_world.new_tensor(n_br)
    fail_rate = viol.float().sum() / (n_ok + eps)

    rad2deg = torch.tensor(180.0 / math.pi, device=dev, dtype=dtype)

    return {
        'closure_loss': closure,
        'closure_bond_loss': bond_mean,
        'closure_angle_loss': angle_mean,
        'closure_torsion_loss': torsion_mean,
        'closure_valid_bridge_fraction': valid_frac,
        'closure_fail_rate': fail_rate,
        'bridge_bond_mae': (bond_mae * m).sum() / (m.sum() + eps),
        'bridge_angle_mae_deg': ((ang_mae * m).sum() / (m.sum() + eps)) * rad2deg,
        'bridge_torsion_mae_deg': ((tor_mae * m).sum() / (m.sum() + eps)) * rad2deg,
    }
