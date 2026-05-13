"""Finite-branch cyclic sugar ring, builder invariants, wrapped score layout."""

import inspect
import math
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

import base2backbone.geometry.backbone as tg
from base2backbone.geometry.backbone import (
    _PSEUDOROTATION_OFFSETS, _build_sugar_ring_finite_branch_debug,
    _get_template_tensors, build_ring_for_c2_azimuth,
    build_sugar_ring_closed_form, dihedral_rad, phase_and_amplitude_to_nus,
    pseudorotation_phase_rad_from_nus, pucker_amplitude_rad,
    signed_tetra_volume, sugar_ring_from_xy_z, sugar_ring_torsions,
    wrap_dihedral_diff)
from base2backbone.model import BackboneLightningModule, TorsionScoreNetwork
from base2backbone.score_diffusion import (decode_torsions, encode_torsions,
                                           perturb_torsions,
                                           weighted_score_mse)
from base2backbone.torsion_constants import N_LATENT, N_TORSIONS, TORSION_NAMES


def test_sugar_ring_closed_form_no_grid():
    bb_src = inspect.getsource(tg.build_backbone_from_torsions)
    assert 'build_sugar_ring_closed_form' in bb_src
    assert 'build_sugar_ring_grid' not in bb_src
    assert 'build_sugar_ring_closed_torch' not in bb_src
    r = build_sugar_ring_closed_form
    assert r is not None and callable(r)


def test_sugar_pseudorotation_torsions_match(device='cpu'):
    dev = torch.device(device)
    P = torch.tensor(
        [
            -torch.pi,
            -2 * torch.pi / 3,
            -torch.pi / 3,
            0.0,
            torch.pi / 3,
            2 * torch.pi / 3,
            torch.pi,
        ],
        device=dev,
    )
    tau_vals = torch.tensor([0.3, 0.5, 0.7], device=dev)
    errs = []
    for tv in tau_vals:
        tau_m = tv.expand_as(P)
        chi = torch.zeros_like(P)
        ri = torch.arange(P.shape[0], device=dev, dtype=torch.long) % 4
        ring = build_sugar_ring_closed_form(chi, P, tau_m, ri)
        nu_act = sugar_ring_torsions(ring)
        nu_tgt = phase_and_amplitude_to_nus(P, tau_m)
        d = wrap_dihedral_diff(nu_act, nu_tgt).abs()
        errs.append(d.reshape(-1))
    all_e = torch.cat(errs)
    mean_all = float(all_e.mean())
    max_all = float(all_e.max())
    assert mean_all < 0.05, mean_all
    assert max_all < 0.13, max_all


def test_sugar_ring_bonds_reasonable(device='cpu'):
    dev = torch.device(device)
    tc = _get_template_tensors(str(dev))
    P = torch.tensor(
        [
            -torch.pi,
            -torch.pi / 2,
            0.0,
            torch.pi / 2,
            torch.pi,
        ],
        device=dev,
    )
    tau_m = torch.full_like(P, 0.45)
    chi = torch.zeros_like(P)
    ri = torch.tensor([0, 1, 2, 3, 0], device=dev, dtype=torch.long)
    ring = build_sugar_ring_closed_form(chi, P, tau_m, ri)
    bl_o4_c1 = (tc['c1'][ri] - tc['o4'][ri]).norm(dim=-1)
    bls = [
        ("O4'", "C1'", bl_o4_c1),
        ("C1'", "C2'", tc['bl_c2_c1'][ri]),
        ("C2'", "C3'", tc['bl_c3_c2'][ri]),
        ("C3'", "C4'", tc['bl_c4_c3'][ri]),
        ("C4'", "O4'", tc['bl_o4_c4'][ri]),
    ]
    diffs = []
    for a, b, tgt_len in bls:
        d = (ring[a] - ring[b]).norm(dim=-1)
        diffs.append((d - tgt_len).abs().reshape(-1))
    ad = torch.cat(diffs)
    assert float(ad.mean()) < 0.03, float(ad.mean())
    assert float(ad.max()) < 0.08, float(ad.max())


def test_sugar_ring_no_grid_no_branch_approx():
    body = inspect.getsource(build_sugar_ring_closed_form)
    banned_substrings = (
        'torch.linspace',
        'linspace',
        'two_phi_p3',
        'two_phi_p0',
        'c3_two_cone_sphere',
        'intersect_circles_3d_torch',
    )
    for s in banned_substrings:
        assert s not in body, s


def test_chi_changes_orientation_not_pucker(device='cpu'):
    dev = torch.device(device)
    ri = torch.tensor([1], device=dev, dtype=torch.long)
    P = torch.tensor([0.16], device=dev)
    tau_m = torch.tensor([0.32], device=dev)
    chi_a = torch.tensor([-0.6], device=dev)
    chi_b = torch.tensor([0.9], device=dev)
    ring_a = build_sugar_ring_closed_form(chi_a, P, tau_m, ri)
    ring_b = build_sugar_ring_closed_form(chi_b, P, tau_m, ri)
    nu_a = sugar_ring_torsions(ring_a)
    nu_b = sugar_ring_torsions(ring_b)
    assert wrap_dihedral_diff(nu_a, nu_b).abs().max().item() < 1e-2
    for nm in ("O4'", "C2'", "C3'", "C4'"):
        assert not torch.allclose(ring_a[nm], ring_b[nm], atol=1e-4, rtol=1e-4)
    tc = _get_template_tensors(str(dev))
    n_atom = tc['chi_n'][ri]
    c_atom = tc['chi_c'][ri]
    mchi = dihedral_rad(
        ring_b["O4'"].reshape(1, 3),
        ring_b["C1'"].reshape(1, 3),
        n_atom.reshape(1, 3),
        c_atom.reshape(1, 3),
    )
    assert wrap_dihedral_diff(mchi, chi_b).abs().item() < 0.02


def test_sugar_ring_chirality_matches_template(device='cpu'):
    dev = torch.device(device)
    tc = _get_template_tensors(str(dev))
    for ri_ in (0, 1, 2, 3):
        ri = torch.tensor([ri_], device=dev)
        P = torch.tensor([0.1 + 0.05 * ri_], device=dev)
        tau_m = torch.tensor([0.31], device=dev)
        chi = torch.zeros(1, device=dev)
        ring = build_sugar_ring_closed_form(chi, P, tau_m, ri)
        o4, c2, c3, c4 = ring["O4'"], ring["C2'"], ring["C3'"], ring["C4'"]
        trip = signed_tetra_volume(
            o4.reshape(1, 3),
            c2.reshape(1, 3),
            c3.reshape(1, 3),
            c4.reshape(1, 3),
        )
        ref = tc['ring_chiral_triple'][ri]
        assert torch.sign(trip) == torch.sign(ref)


def test_batch_shapes_and_finite(device='cpu'):
    dev = torch.device(device)
    B, W = 3, 5
    chi = torch.randn(B, W, device=dev)
    P = torch.randn(B, W, device=dev)
    tau_m = torch.rand(B, W, device=dev) * 0.08 + 0.28
    ri = torch.randint(0, 4, (B, W), device=dev)
    ring = build_sugar_ring_closed_form(chi, P, tau_m, ri)
    for nm in ("O4'", "C1'", "C2'", "C3'", "C4'"):
        assert ring[nm].shape == (B, W, 3)
        assert torch.isfinite(ring[nm]).all()


def test_sugar_ring_from_xy_z_contract(device='cpu'):
    dev = torch.device(device)
    sp = tg._planar_sugar_spec()
    xy = torch.as_tensor(sp['xy'], device=dev, dtype=torch.float32)
    z = torch.zeros(2, 5, device=dev)
    ctr = torch.as_tensor(sp['center'], device=dev)
    e1 = torch.as_tensor(sp['e1'], device=dev)
    e2 = torch.as_tensor(sp['e2'], device=dev)
    en = torch.as_tensor(sp['en'], device=dev)
    out = sugar_ring_from_xy_z(xy, z, ctr, e1, e2, en)
    assert out.shape == (2, 5, 3)
    assert torch.isfinite(out).all()


def test_closed_form_no_delta_no_sincos():
    assert N_TORSIONS == 7
    assert N_LATENT == 8
    assert TORSION_NAMES == (
        'alpha',
        'beta',
        'gamma',
        'epsilon',
        'zeta',
        'chi',
        'pseudorotation_phase',
    )
    src = inspect.getsource(build_sugar_ring_closed_form)
    assert 'delta' not in inspect.signature(build_sugar_ring_closed_form).parameters
    assert 'delta' not in src.lower()
    wsd = Path(__file__).resolve().parents[1] / 'src' / 'base2backbone' / 'score_diffusion.py'
    txt = wsd.read_text()
    assert re.search(r'N_TORSIONS\s*\*\s*2', txt) is None
    assert 'angular_score_target' in txt
    assert 'weighted_score_mse' in txt
    B = torch.zeros(2, 3, 8)
    pred = torch.randn(2, 3, 8)
    m = torch.ones(2, 3, 8, dtype=torch.bool)
    loss = weighted_score_mse(pred, B, m, None, weighting='none')
    assert torch.isfinite(loss)

    hp = dict(
        hidden_dim=16,
        num_heads=2,
        num_layers=1,
        num_timesteps=3,
        batch_size=1,
        lr=1e-3,
        lr_scheduler=None,
        lr_scheduler_patience=1,
        lr_scheduler_threshold=0.1,
        lr_scheduler_cooldown=0,
        angular_sigma_min=1e-3,
        angular_sigma_max=0.2,
        tau_sigma_min=1e-3,
        tau_sigma_max=0.2,
        tau_loss_weight=1.0,
        score_loss_weighting='sigma2',
        weight_decay=0.01,
        closure_loss_weight=0.0,
        closure_bond_weight=1.0,
        closure_angle_weight=1.0,
        closure_torsion_weight=1.0,
    )
    pl = BackboneLightningModule(**cast(Any, hp)).float()
    den = TorsionScoreNetwork(pl.node_dim, hp['hidden_dim'], hp['num_heads'], hp['num_layers'])
    x = torch.randn(2, 3, pl.node_dim)
    assert den(x).shape[-1] == N_LATENT

    theta = torch.randn(2, N_TORSIONS)
    tau = torch.exp(torch.randn(2) * 0.1 + 0.2)
    latent = encode_torsions(theta, tau)
    th2, t2 = decode_torsions(latent)
    assert th2.shape[-1] == N_TORSIONS
    out = perturb_torsions(
        theta,
        torch.log(tau.clamp(min=1e-6)).unsqueeze(-1),
        torch.tensor([0.5]),
        0.05,
        1.0,
        0.05,
        1.0,
    )
    assert out['angular_score_target'].shape == theta.shape


def test_pucker_amplitude_roundtrip():
    """tau = pucker_amplitude_rad(deg(tau*cos(P+offsets)), P) must hold exactly."""
    for P in np.linspace(-math.pi, math.pi, 13):
        for tau in [0.25, 0.35, 0.45, 0.55]:
            nu_rad = tau * np.cos(P + _PSEUDOROTATION_OFFSETS)
            nu_deg = np.degrees(nu_rad)
            P2 = pseudorotation_phase_rad_from_nus(nu_deg)
            tau2 = pucker_amplitude_rad(nu_deg, P2)
            assert abs(tau2 - tau) < 1e-6, (
                f'roundtrip failed: P={np.degrees(P):.1f} deg, tau={tau:.3f}, tau2={tau2:.6f}'
            )


def _sugar_grid_error_stats(builder, device='cpu'):
    dev = torch.device(device)
    errs = []
    for P_val in np.linspace(-math.pi, math.pi, 17):
        for tau_val in [0.25, 0.35, 0.45, 0.55]:
            for ri_ in range(4):
                P = torch.tensor([P_val], dtype=torch.float32, device=dev)
                tm = torch.tensor([tau_val], dtype=torch.float32, device=dev)
                chi = torch.zeros(1, device=dev)
                ri = torch.tensor([ri_], device=dev, dtype=torch.long)
                ring = builder(chi, P, tm, ri)
                nu_act = sugar_ring_torsions(ring)
                nu_tgt = phase_and_amplitude_to_nus(P, tm)
                d = wrap_dihedral_diff(nu_act, nu_tgt).abs()
                errs.append(d.reshape(-1))
    all_e = torch.cat(errs)
    return float(all_e.mean()), float(all_e.max())


def test_pseudorotation_roundtrip_diagnostic_bound(device='cpu'):
    """The continuous C2' azimuth solver must keep the global torsion error bounded."""
    mean_e, max_e = _sugar_grid_error_stats(build_sugar_ring_closed_form, device=device)
    assert mean_e < 0.04, f'mean torsion error {mean_e:.4f} rad >= 0.04'
    assert max_e < 0.13, f'max torsion error {max_e:.4f} rad >= 0.13'


def test_builder_enforces_nu0_nu4(device='cpu'):
    dev = torch.device(device)
    errs = []
    for P_val in np.linspace(-math.pi, math.pi, 11):
        for tau_val in [0.25, 0.35, 0.45, 0.55]:
            P = torch.tensor([P_val], dtype=torch.float32, device=dev)
            tm = torch.tensor([tau_val], dtype=torch.float32, device=dev)
            chi = torch.zeros(1, device=dev)
            ri = torch.tensor([1], device=dev, dtype=torch.long)
            ring = build_sugar_ring_closed_form(chi, P, tm, ri)
            nu_act = sugar_ring_torsions(ring)
            nu_tgt = phase_and_amplitude_to_nus(P, tm)
            errs.append(wrap_dihedral_diff(nu_act, nu_tgt).abs())
    all_err = torch.cat(errs, dim=0)
    assert all_err[:, 0].max().item() < 1e-4
    assert all_err[:, 4].max().item() < 1e-4


def test_builder_reduces_nu1_to_nu3_vs_current_finite_branch(device='cpu'):
    dev = torch.device(device)
    new_errs = []
    old_errs = []
    for P_val in np.linspace(-math.pi, math.pi, 17):
        for tau_val in [0.25, 0.35, 0.45, 0.55]:
            for ri_ in range(4):
                P = torch.tensor([P_val], dtype=torch.float32, device=dev)
                tm = torch.tensor([tau_val], dtype=torch.float32, device=dev)
                chi = torch.zeros(1, device=dev)
                ri = torch.tensor([ri_], device=dev, dtype=torch.long)
                nu_tgt = phase_and_amplitude_to_nus(P, tm)
                ring_new = build_sugar_ring_closed_form(chi, P, tm, ri)
                ring_old = _build_sugar_ring_finite_branch_debug(chi, P, tm, ri)
                new_errs.append(wrap_dihedral_diff(sugar_ring_torsions(ring_new), nu_tgt).abs()[:, 1:4])
                old_errs.append(wrap_dihedral_diff(sugar_ring_torsions(ring_old), nu_tgt).abs()[:, 1:4])
    new_mean = torch.cat(new_errs, dim=0).mean().item()
    old_mean = torch.cat(old_errs, dim=0).mean().item()
    assert new_mean < old_mean, (new_mean, old_mean)


def test_c2_azimuth_is_gauge_in_local_formulation(device='cpu'):
    dev = torch.device(device)
    chi = torch.zeros(1, device=dev)
    P = torch.tensor([0.41], dtype=torch.float32, device=dev)
    tau_m = torch.tensor([0.37], dtype=torch.float32, device=dev)
    ri = torch.tensor([2], device=dev, dtype=torch.long)
    ring_a = build_ring_for_c2_azimuth(torch.tensor([0.0], dtype=torch.float32, device=dev), chi, P, tau_m, ri)
    ring_b = build_ring_for_c2_azimuth(torch.tensor([0.9], dtype=torch.float32, device=dev), chi, P, tau_m, ri)
    nu_a = sugar_ring_torsions(ring_a)
    nu_b = sugar_ring_torsions(ring_b)
    d = wrap_dihedral_diff(nu_a, nu_b).abs()
    assert d.max().item() < 1e-5


def test_ring_bonds_exact(device='cpu'):
    """All five ring bond lengths must match template to within 1e-4 A."""
    dev = torch.device(device)
    tc = _get_template_tensors(str(dev))
    P = torch.linspace(-math.pi, math.pi, 9, device=dev)
    tau_m = torch.full_like(P, 0.4)
    chi = torch.zeros_like(P)
    ri = torch.arange(9, device=dev, dtype=torch.long) % 4
    ring = build_sugar_ring_closed_form(chi, P, tau_m, ri)
    bl_o4_c1_tgt = (tc['c1'][ri] - tc['o4'][ri]).norm(dim=-1)
    bls = [
        ("O4'", "C1'", bl_o4_c1_tgt),
        ("C1'", "C2'", tc['bl_c2_c1'][ri]),
        ("C2'", "C3'", tc['bl_c3_c2'][ri]),
        ("C3'", "C4'", tc['bl_c4_c3'][ri]),
        ("C4'", "O4'", tc['bl_o4_c4'][ri]),
    ]
    for a, b, tgt in bls:
        err = ((ring[a] - ring[b]).norm(dim=-1) - tgt).abs()
        assert err.max().item() < 1e-4, (
            f'Bond {a}-{b}: max err = {err.max().item():.2e} A'
        )


def test_no_pca_calibration_in_builder():
    """build_sugar_ring_closed_form must not call PCA/calibration helpers."""
    body = inspect.getsource(build_sugar_ring_closed_form)
    for banned in (
        '_planar_sugar_spec_tensors',
        '_planar_sugar_spec',
        'sugar_ring_from_xy_z',
        'z_scale',
        'phase_rad',
        'calibrate_planar',
    ):
        assert banned not in body, f'Found banned term in builder: {banned!r}'
