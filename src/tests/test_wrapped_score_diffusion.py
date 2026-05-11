"""Unit tests for wrapped-normal scores, perturbation, and VE sampler primitives."""

import math
from typing import Any, cast

import pytest
import torch

from model import PytorchLightningModule
from torsion_constants import N_LATENT, N_TORSIONS, TORSION_NAMES
from wrapped_score_diffusion import (
    decode_torsions,
    gaussian_score,
    perturb_torsions,
    reverse_ve_score_step,
    sigma_schedule,
    ve_sigma_grid,
    weighted_score_mse,
    wrap_angle,
    wrapped_angle_diff,
    wrapped_normal_score,
)


def test_wrapped_normal_score_small_sigma_matches_gaussian_near_zero():
    x = torch.tensor([0.05, -0.03])
    sigma = torch.tensor([0.1])
    s = wrapped_normal_score(x, sigma)
    approx = -x / (sigma ** 2)
    assert torch.allclose(s, approx, rtol=0.06, atol=1e-4)


def test_wrapped_normal_score_periodic_boundary():
    for xv in (math.pi - 1e-4, -math.pi + 1e-4):
        x = torch.tensor([xv])
        sigma = torch.tensor([0.5])
        s = wrapped_normal_score(x, sigma)
        assert torch.isfinite(s).all()


def test_wrapped_normal_score_shape_broadcast():
    B, W, C = 4, 5, 7
    x = torch.randn(B, W, C)
    sigma = torch.randn(B) * 0.05 + 0.2
    s = wrapped_normal_score(x, sigma)
    assert s.shape == x.shape


def test_perturb_torsions_shapes():
    B, W = 3, 4
    theta_0 = torch.randn(B, W, N_TORSIONS)
    log_tau_0 = torch.randn(B, W, 1)
    t = torch.rand(B)
    pert = perturb_torsions(theta_0, log_tau_0, t, 0.05, 1.2, 0.05, 1.2)
    assert pert['theta_t'].shape == (B, W, N_TORSIONS)
    assert pert['log_tau_t'].shape == (B, W, 1)
    assert pert['angular_score_target'].shape == (B, W, N_TORSIONS)
    assert pert['tau_score_target'].shape == (B, W, 1)


def test_perturb_torsions_angles_wrapped():
    torch.manual_seed(1)
    B = 8
    theta_0 = torch.randn(B, N_TORSIONS)
    log_tau_0 = torch.randn(B, 1)
    t = torch.rand(B)
    pert = perturb_torsions(theta_0, log_tau_0, t, 0.1, 2.0, 0.1, 2.0)
    ang = pert['theta_t']
    assert (ang > -math.pi - 1e-4).all() and (ang <= math.pi + 1e-4).all()


def test_angular_target_matches_wrapped_score_formula():
    torch.manual_seed(2)
    theta_0 = torch.randn(5, N_TORSIONS)
    log_tau_0 = torch.randn(5, 1)
    t = torch.ones(5) * 0.73
    pert = perturb_torsions(theta_0, log_tau_0, t, 0.2, 2.5, 0.2, 2.5)
    sig = pert['sigma_theta']
    ref = wrapped_normal_score(wrapped_angle_diff(pert['theta_t'], theta_0), sig)
    assert torch.allclose(pert['angular_score_target'], ref, rtol=1e-5, atol=1e-5)


def test_score_target_differs_from_naive_gaussian_score_when_wrapping_matters():
    torch.manual_seed(3)
    theta_0 = torch.zeros(10, N_TORSIONS)
    log_tau_0 = torch.zeros(10, 1)
    t = torch.ones(10) * 0.9
    pert = perturb_torsions(theta_0, log_tau_0, t, 1.5, 3.0, 0.05, 0.2)
    sig = pert['sigma_theta']
    x = wrapped_angle_diff(pert['theta_t'], theta_0)
    naive = -(x / (sig ** 2))
    diff = (pert['angular_score_target'] - naive).abs().max()
    assert float(diff) > 5e-3


def test_model_output_dim_is_8():
    hp = dict(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        num_timesteps=3,
        batch_size=1,
        lr=1e-3,
        lr_scheduler=None,
        lr_scheduler_patience=1,
        lr_scheduler_threshold=0.1,
        lr_scheduler_cooldown=0,
        angular_sigma_min=0.05,
        angular_sigma_max=1.0,
        tau_sigma_min=0.05,
        tau_sigma_max=1.0,
        tau_loss_weight=1.0,
        score_loss_weighting='sigma2',
        weight_decay=0.01,
        closure_loss_weight=0.0,
        closure_bond_weight=1.0,
        closure_angle_weight=1.0,
        closure_torsion_weight=1.0,
        log_closure_metrics_train=False,
        log_closure_metrics_val=True,
    )
    pl = PytorchLightningModule(**cast(Any, hp)).float()
    assert pl.denoiser.out.out_features == N_LATENT == 8


def test_no_sincos_latent_constants():
    assert N_LATENT == 8 and N_TORSIONS == 7


def test_delta_absent():
    assert 'delta' not in TORSION_NAMES


def test_decode_clamps_extreme_log_tau():
    from torsion_constants import TAU_M_MAX, TAU_M_MIN

    x = torch.zeros(2, N_LATENT)
    x[:, N_TORSIONS] = 10.0
    _, tau = decode_torsions(x)
    assert float(tau.max()) <= TAU_M_MAX + 1e-4
    x[:, N_TORSIONS] = -10.0
    _, tau2 = decode_torsions(x)
    assert float(tau2.min()) >= TAU_M_MIN - 1e-4


@pytest.mark.parametrize('weighting', ('none', 'sigma2'))
def test_weighted_score_mse_runs(weighting):
    pred = torch.randn(4, 7)
    tgt = torch.randn(4, 7)
    mask = torch.ones(4, 7)
    lam = (torch.ones(4, 1) * 0.25) if weighting == 'sigma2' else None
    weighted_score_mse(pred, tgt, mask, lam, weighting=weighting)


def test_gaussian_ve_step_reduces_magnitude():
    x_t = torch.tensor([1.5])
    sigma_cur = torch.tensor([2.0])
    sigma_next = torch.tensor([0.5])
    score = gaussian_score(x_t.unsqueeze(-1), torch.zeros(1, 1), sigma_cur).squeeze(-1)
    d2 = sigma_cur ** 2 - sigma_next ** 2
    x_next = x_t + d2 * score
    assert abs(float(x_next)) < abs(float(x_t))


def test_wrapped_small_angle_ve_step_reduces_magnitude():
    theta = torch.tensor([0.4])
    sigma_cur = torch.tensor([1.5])
    sigma_next = torch.tensor([0.3])
    score = wrapped_normal_score(theta, sigma_cur)
    d2 = sigma_cur ** 2 - sigma_next ** 2
    theta_next = wrap_angle(theta + d2 * score)
    assert abs(float(theta_next)) < abs(float(theta))


def test_score_sampler_reverse_step_shapes_and_finite():
    B = 6
    theta = torch.randn(B, N_TORSIONS)
    log_tau = torch.randn(B, 1)
    score = torch.randn(B, N_LATENT)
    dev = theta.device
    dtype = theta.dtype
    sc = torch.tensor(1.0, device=dev, dtype=dtype)
    sn = torch.tensor(0.55, device=dev, dtype=dtype)
    st = torch.tensor(1.2, device=dev, dtype=dtype)
    stn = torch.tensor(0.6, device=dev, dtype=dtype)
    th2, lt2 = reverse_ve_score_step(theta, log_tau, score, sc, sn, st, stn)
    assert th2.shape == theta.shape and lt2.shape == log_tau.shape
    assert torch.isfinite(th2).all() and torch.isfinite(lt2).all()


def test_ve_sigma_grid_monotonic():
    g = ve_sigma_grid(2.0, 0.05, 10, device=torch.device('cpu'), dtype=torch.float32)
    assert g[0] > g[-1]
    assert (torch.diff(g) < 0).all()


def test_sigma_schedule_endpoints():
    smin, smax = 0.01, 1.0
    a = sigma_schedule(torch.zeros(1), smin, smax)
    b = sigma_schedule(torch.ones(1), smin, smax)
    assert abs(float(a) - smin) < 1e-5
    assert abs(float(b) - smax) < 1e-5


def test_perturb_raises_on_wrong_torsion_width():
    with pytest.raises(ValueError, match='theta_0 last dimension'):
        perturb_torsions(
            torch.randn(2, 8),
            torch.randn(2, 1),
            torch.rand(2),
            0.1,
            1.0,
            0.1,
            1.0,
        )


def test_synthetic_perturb_decode_window_builder_closure_finite():
    from bridge_closure import compute_bridge_closure_loss
    from torsion_geometry import build_batch_window_backbone_from_torsions_torch

    torch.manual_seed(0)
    B, W = 2, 4
    theta0 = torch.randn(B, W, N_TORSIONS) * 0.05
    lt0 = torch.log(torch.full((B, W), 0.4)).unsqueeze(-1)
    t = torch.rand(B)
    pert = perturb_torsions(theta0, lt0, t, 0.05, 0.2, 0.05, 0.2)
    lat = torch.cat([pert['theta_t'], pert['log_tau_t']], dim=-1)
    th2, tau2 = decode_torsions(lat)
    assert torch.isfinite(th2).all() and torch.isfinite(tau2).all()
    ri = torch.randint(0, 4, (B, W))
    ori = torch.randn(B, W, 3)
    frm = torch.eye(3, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(B, W, 3, 3).contiguous()
    m = torch.ones(B, W, N_TORSIONS, dtype=torch.bool)
    bb = build_batch_window_backbone_from_torsions_torch(
        th2.float(), tau2.float(), ri.long(), ori.float(), frm.float(), m,
    )
    assert torch.isfinite(bb).all()
    pm = torch.ones(B, W - 1, dtype=torch.bool)
    clo = compute_bridge_closure_loss(bb, theta0, m, ri.long(), valid_pair_mask=pm)
    assert torch.isfinite(clo['closure_loss']).all()

