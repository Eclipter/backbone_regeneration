"""Tests for phosphate bridge closure loss (bonds, angles, wrapped torsions)."""

import math
from pathlib import Path

import pytest
import torch

from base2backbone.bridge_closure import (
    canonical_two_residue_bridge_bb_tensor, compute_bridge_closure_loss)
from base2backbone.data import BACKBONE_ATOMS
from base2backbone.geometry.backbone import (
    N_TORSIONS,
    TOR_BRIDGE_PHASE,
    _get_template_tensors,
    wrap_dihedral_diff,
)
from base2backbone.geometry import bridge_phase_from_points_torch


def _ideal_bridge_bb_and_targets(dtype=torch.float64, device='cpu'):
    """Aligned with bridge closure refs: canonical A–A pair bb from ``bridge_closure`` helpers."""
    B, W = 1, 2
    nj = {nm: j for j, nm in enumerate(BACKBONE_ATOMS)}
    bb2 = canonical_two_residue_bridge_bb_tensor('A', 'A', dtype=dtype, device=device)
    bb = bb2.unsqueeze(0)

    j_c4, j_c3, j_o3 = nj["C4'"], nj["C3'"], nj["O3'"]
    j_p, j_o5 = nj['P'], nj["O5'"]

    c4_p = bb[:, 0, j_c4]
    c3_p = bb[:, 0, j_c3]
    o3_p = bb[:, 0, j_o3]
    p_n = bb[:, 1, j_p]
    o5_n = bb[:, 1, j_o5]
    tc = _get_template_tensors(str(bb.device))
    ri = torch.zeros((B, W), dtype=torch.long, device=device)
    phase = bridge_phase_from_points_torch(
        o3_p,
        o5_n,
        c3_p,
        p_n,
        tc['bond_p_o3_inter'][ri[:, 1]],
        tc['bond_p_o5'][ri[:, 1]],
        c4_p,
    )

    tors = torch.zeros(B, W, N_TORSIONS, dtype=dtype, device=device)
    tors[:, 1, TOR_BRIDGE_PHASE] = phase

    mask = torch.zeros(B, W, N_TORSIONS, dtype=torch.bool, device=device)
    mask[:, 1, TOR_BRIDGE_PHASE] = True
    valid = torch.ones(B, W, dtype=torch.bool, device=device)
    return bb, tors, mask, valid, ri


def test_bridge_closure_loss_zero_or_low_on_reference_geometry():
    bb, tors, mask, valid, ri = _ideal_bridge_bb_and_targets()
    out = compute_bridge_closure_loss(
        bb, tors, mask, ri,
        weights={'bond': 1, 'angle': 1, 'torsion': 1},
    )
    assert torch.isfinite(out['closure_loss']).all()
    assert float(out['closure_loss']) < 1e-2


def test_bridge_closure_loss_ignores_invalid_nan_bridge_backward():
    bb_valid, tors_valid, mask_valid, _valid, ri_valid = _ideal_bridge_bb_and_targets(dtype=torch.float32)
    bb = bb_valid.repeat(2, 1, 1, 1).detach()
    tors = tors_valid.repeat(2, 1, 1).detach()
    mask = mask_valid.repeat(2, 1, 1).detach()
    ri = ri_valid.repeat(2, 1).detach()

    # Invalid row mimics chain-edge phosphate atoms: NaN coordinates but bridge mask is false.
    j_p = BACKBONE_ATOMS.index('P')
    bb[1, 1, j_p] = float('nan')
    mask[1, 1, TOR_BRIDGE_PHASE] = False
    bb.requires_grad_(True)

    out = compute_bridge_closure_loss(bb, tors, mask, ri)
    assert torch.isfinite(out['closure_loss'])

    out['closure_loss'].backward()
    assert torch.isfinite(bb.grad).all()


def test_bridge_closure_loss_penalizes_broken_bond():
    bb, tors, mask, _, ri = _ideal_bridge_bb_and_targets()
    base = compute_bridge_closure_loss(bb, tors, mask, ri)
    j_p = BACKBONE_ATOMS.index('P')
    j_o3 = BACKBONE_ATOMS.index("O3'")
    bb_bad = bb.clone()
    dv = bb_bad[:, 1, j_p] - bb_bad[:, 0, j_o3]
    bb_bad[:, 1, j_p] = bb_bad[:, 1, j_p] + 0.35 * dv / (dv.norm(dim=-1, keepdim=True) + 1e-12)
    bad = compute_bridge_closure_loss(bb_bad, tors, mask, ri)
    assert float(bad['closure_bond_loss']) > float(base['closure_bond_loss']) + 5e-2


def test_bridge_closure_loss_penalizes_wrong_angle():
    bb, tors, mask, _, ri = _ideal_bridge_bb_and_targets()
    base = compute_bridge_closure_loss(bb, tors, mask, ri)
    j_o5 = BACKBONE_ATOMS.index("O5'")
    bb_bad = bb.clone()
    bb_bad[:, 1, j_o5] = bb_bad[:, 1, j_o5] + torch.tensor([0.0, 0.55, 0.0], dtype=bb.dtype, device=bb.device)
    bad = compute_bridge_closure_loss(bb_bad, tors, mask, ri)
    assert float(bad['closure_angle_loss']) > float(base['closure_angle_loss']) + 5e-2


def test_bridge_closure_loss_uses_wrapped_torsion_error():
    pred = torch.tensor([179.0 * math.pi / 180.0], dtype=torch.float64)
    tgt = torch.tensor([-179.0 * math.pi / 180.0], dtype=torch.float64)
    err = wrap_dihedral_diff(pred, tgt)
    err_deg = abs(float(err)) * 180.0 / math.pi
    assert abs(err_deg - 2.0) < 0.1


def test_bridge_closure_loss_ignores_invalid_bridges():
    bb, tors, mask, valid, ri = _ideal_bridge_bb_and_targets()
    pm = torch.zeros(1, 1, dtype=torch.bool, device=bb.device)
    out = compute_bridge_closure_loss(bb, tors, mask, ri, valid_pair_mask=pm)
    assert float(out['closure_loss']) == 0.0
    assert float(out['closure_valid_bridge_fraction']) == 0.0
    assert float(out['closure_num_valid_bridges']) == 0.0
    assert torch.isfinite(out['closure_bond_loss']).all()


def test_delta_absent_from_closure_loss():
    text = Path(__file__).resolve().parents[1] / 'src' / 'base2backbone' / 'bridge_closure.py'
    low = text.read_text().lower()
    assert 'delta' not in low


def test_bridge_closure_valid_pair_mask_wrong_shape_raises():
    bb, tors, mask, valid, ri = _ideal_bridge_bb_and_targets()
    bad = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match='valid_pair_mask'):
        compute_bridge_closure_loss(bb, tors, mask, ri, valid_pair_mask=bad)


def test_bridge_closure_ignores_pairs_masked_by_valid_pair_mask():
    bb, tors, mask, valid, ri = _ideal_bridge_bb_and_targets()
    pm_zero = torch.zeros(1, 1, dtype=torch.bool, device=bb.device)
    out = compute_bridge_closure_loss(bb, tors, mask, ri, valid_pair_mask=pm_zero)
    assert float(out['closure_loss']) == 0.0
    pm_one = torch.ones(1, 1, dtype=torch.bool, device=bb.device)
    out_one = compute_bridge_closure_loss(bb, tors, mask, ri, valid_pair_mask=pm_one)
    assert float(out_one['closure_valid_bridge_fraction']) > 0.0


def test_processed_pt_smoke_if_present():
    data_root = Path(__file__).resolve().parents[2] / 'data'
    pts = sorted(data_root.rglob('*.pt'))
    pts = [p for p in pts if not p.name.startswith('pre_')]
    if not pts:
        return
    sample = torch.load(pts[0], weights_only=False, map_location='cpu')
    assert hasattr(sample, 'torsions')


def test_bridge_o3p_bond_target_is_not_intraresidue_o3_p_separation():
    tc = _get_template_tensors('cpu')
    for idx in range(4):
        tgt = tc['bond_p_o3_inter'][idx].item()
        wrong = tc['tpl_o3_sep_p'][idx].item()
        po5 = tc['bond_p_o5'][idx].item()
        assert abs(tgt - po5) < 1e-5
        assert abs(tgt - wrong) > 0.4


def test_bridge_mask_allows_bridge_without_full_nucleotide_validity_flags():
    bb, tors, mask, _, ri = _ideal_bridge_bb_and_targets()
    vb = torch.ones(1, 1, dtype=torch.bool)
    out = compute_bridge_closure_loss(bb, tors, mask, ri, valid_bridge_mask=vb)
    assert float(out['closure_num_valid_bridges']) >= 1.0
    assert float(out['closure_loss']) < 1e-1


def test_bridge_mask_rejects_when_required_bridge_atom_is_nan():
    bb, tors, mask, _, ri = _ideal_bridge_bb_and_targets()
    j_o3 = BACKBONE_ATOMS.index("O3'")
    bb2 = bb.clone()
    bb2[0, 0, j_o3] = float('nan')
    vb = torch.ones(1, 1, dtype=torch.bool)
    out = compute_bridge_closure_loss(bb2, tors, mask, ri, valid_bridge_mask=vb)
    assert float(out['closure_num_valid_bridges']) == 0.0


def test_bridge_closure_valid_bridge_mask_wrong_shape_raises():
    bb, tors, mask, _, ri = _ideal_bridge_bb_and_targets()
    bad = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match='valid_bridge_mask'):
        compute_bridge_closure_loss(bb, tors, mask, ri, valid_bridge_mask=bad)
