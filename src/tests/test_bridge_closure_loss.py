"""Tests for phosphate bridge closure loss (bonds, angles, wrapped torsions)."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from bridge_closure import compute_bridge_closure_loss
from torsion_geometry import (
    N_TORSIONS,
    TOR_ALPHA,
    TOR_BETA,
    TOR_EPS,
    TOR_ZETA,
    _get_template,
    dihedral_rad_torch,
    wrap_dihedral_diff_torch,
)
from utils import backbone_atoms


def _ideal_bridge_bb_and_targets(dtype=torch.float64, device='cpu'):
    """Two stacked 'A' templates: residue 0 O3′ at origin; residue 1 rigid-translated so P matches."""
    tp = _get_template('A')
    tn = _get_template('A')
    o3_ref = tp["O3'"]
    prev = {k: tp[k] - o3_ref for k in backbone_atoms if k in tp}

    vp = tp['P'] - tp["O3'"]
    shift_next = vp - tn['P']
    nxt = {k: tn[k] + shift_next for k in backbone_atoms if k in tn}

    B, W = 1, 2
    n_bb = len(backbone_atoms)
    bb = torch.full((B, W, n_bb, 3), float('nan'), dtype=dtype, device=device)
    nj = {nm: j for j, nm in enumerate(backbone_atoms)}
    for nm in backbone_atoms:
        if nm in prev:
            bb[0, 0, nj[nm]] = torch.as_tensor(prev[nm], dtype=dtype, device=device)
        if nm in nxt:
            bb[0, 1, nj[nm]] = torch.as_tensor(nxt[nm], dtype=dtype, device=device)

    j_c4, j_c3, j_o3 = nj["C4'"], nj["C3'"], nj["O3'"]
    j_p, j_o5, j_c5 = nj['P'], nj["O5'"], nj["C5'"]

    c4_p = bb[:, 0, j_c4]
    c3_p = bb[:, 0, j_c3]
    o3_p = bb[:, 0, j_o3]
    p_n = bb[:, 1, j_p]
    o5_n = bb[:, 1, j_o5]
    c5_n = bb[:, 1, j_c5]
    c4_n = bb[:, 1, j_c4]

    eps = dihedral_rad_torch(c4_p, c3_p, o3_p, p_n)
    ze = dihedral_rad_torch(c3_p, o3_p, p_n, o5_n)
    al = dihedral_rad_torch(o3_p, p_n, o5_n, c5_n)
    be = dihedral_rad_torch(p_n, o5_n, c5_n, c4_n)

    tors = torch.zeros(B, W, N_TORSIONS, dtype=dtype, device=device)
    tors[:, 0, TOR_EPS] = eps
    tors[:, 0, TOR_ZETA] = ze
    tors[:, 1, TOR_ALPHA] = al
    tors[:, 1, TOR_BETA] = be

    mask = torch.zeros(B, W, N_TORSIONS, dtype=torch.bool, device=device)
    mask[:, 0, TOR_EPS] = True
    mask[:, 0, TOR_ZETA] = True
    mask[:, 1, TOR_ALPHA] = True
    mask[:, 1, TOR_BETA] = True

    ri = torch.zeros((B, W), dtype=torch.long, device=device)
    valid = torch.ones(B, W, dtype=torch.bool, device=device)
    return bb, tors, mask, valid, ri


def test_bridge_closure_loss_zero_or_low_on_reference_geometry():
    bb, tors, mask, valid, ri = _ideal_bridge_bb_and_targets()
    out = compute_bridge_closure_loss(bb, tors, mask, valid, ri, weights={'bond': 1, 'angle': 1, 'torsion': 1})
    assert torch.isfinite(out['closure_loss']).all()
    assert float(out['closure_loss']) < 1e-2


def test_bridge_closure_loss_penalizes_broken_bond():
    bb, tors, mask, valid, ri = _ideal_bridge_bb_and_targets()
    base = compute_bridge_closure_loss(bb, tors, mask, valid, ri)
    j_p = backbone_atoms.index('P')
    j_o3 = backbone_atoms.index("O3'")
    bb_bad = bb.clone()
    dv = bb_bad[:, 1, j_p] - bb_bad[:, 0, j_o3]
    bb_bad[:, 1, j_p] = bb_bad[:, 1, j_p] + 0.35 * dv / (dv.norm(dim=-1, keepdim=True) + 1e-12)
    bad = compute_bridge_closure_loss(bb_bad, tors, mask, valid, ri)
    assert float(bad['closure_bond_loss']) > float(base['closure_bond_loss']) + 5e-2


def test_bridge_closure_loss_penalizes_wrong_angle():
    bb, tors, mask, valid, ri = _ideal_bridge_bb_and_targets()
    base = compute_bridge_closure_loss(bb, tors, mask, valid, ri)
    j_o5 = backbone_atoms.index("O5'")
    bb_bad = bb.clone()
    bb_bad[:, 1, j_o5] = bb_bad[:, 1, j_o5] + torch.tensor([0.0, 0.55, 0.0], dtype=bb.dtype, device=bb.device)
    bad = compute_bridge_closure_loss(bb_bad, tors, mask, valid, ri)
    assert float(bad['closure_angle_loss']) > float(base['closure_angle_loss']) + 5e-2


def test_bridge_closure_loss_uses_wrapped_torsion_error():
    pred = torch.tensor([179.0 * math.pi / 180.0], dtype=torch.float64)
    tgt = torch.tensor([-179.0 * math.pi / 180.0], dtype=torch.float64)
    err = wrap_dihedral_diff_torch(pred, tgt)
    err_deg = abs(float(err)) * 180.0 / math.pi
    assert abs(err_deg - 2.0) < 0.1


def test_bridge_closure_loss_ignores_invalid_bridges():
    bb, tors, mask, valid, ri = _ideal_bridge_bb_and_targets()
    valid_z = torch.zeros_like(valid)
    out = compute_bridge_closure_loss(bb, tors, mask, valid_z, ri)
    assert float(out['closure_loss']) == 0.0
    assert float(out['closure_valid_bridge_fraction']) == 0.0
    assert torch.isfinite(out['closure_bond_loss']).all()


def test_delta_absent_from_closure_loss():
    text = Path(__file__).resolve().parents[1] / 'bridge_closure.py'
    low = text.read_text().lower()
    assert 'delta' not in low


def test_processed_pt_smoke_if_present():
    data_root = Path(__file__).resolve().parents[2] / 'data'
    pts = sorted(data_root.rglob('*.pt'))
    pts = [p for p in pts if not p.name.startswith('pre_')]
    if not pts:
        return
    sample = torch.load(pts[0], weights_only=False, map_location='cpu')
    assert hasattr(sample, 'torsions')
    td_last = sample.torsions.shape[-1]
    assert td_last in (N_TORSIONS, N_TORSIONS + 1), td_last

