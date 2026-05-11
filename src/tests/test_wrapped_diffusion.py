"""Integration-style layout tests (torsions, χ geometry, closure conventions)."""

import math
import re
from pathlib import Path
from typing import Any, cast

import torch

from model import PytorchLightningModule, TorsionDenoiser
from torsion_constants import assert_torsion_layout
from torsion_geometry import (
    N_LATENT,
    N_TORSIONS,
    TORSION_NAMES,
    build_sugar_ring_grid_closed_torch,
    dihedral_rad_torch,
    wrap_dihedral_diff_torch,
)
from wrapped_score_diffusion import encode_torsions, decode_torsions, perturb_torsions, wrap_angle


def test_wrap_angle_range():
    x = torch.linspace(-25.0, 25.0, 500)
    w = wrap_angle(x)
    assert (w > -math.pi - 1e-5).all() and (w <= math.pi + 1e-5).all()


def test_wrapped_angle_diff_boundary():
    from wrapped_score_diffusion import wrapped_angle_diff

    a = torch.tensor([179.0 * math.pi / 180.0])
    b = torch.tensor([-179.0 * math.pi / 180.0])
    d = wrapped_angle_diff(a, b)
    assert abs(float(d) * 180.0 / math.pi) - 2.0 < 0.15


def test_encode_decode_torsions_shape():
    B, W = 2, 3
    theta = torch.randn(B, W, N_TORSIONS)
    tau_m = torch.exp(torch.randn(B, W) * 0.1 + 0.2)
    latent = encode_torsions(theta, tau_m)
    assert latent.shape == (B, W, N_LATENT)
    th2, t2 = decode_torsions(latent)
    assert th2.shape == (B, W, N_TORSIONS)
    assert t2.shape == (B, W)
    assert (t2 > 0).all()
    assert (wrap_angle(th2) - th2).abs().max() < 1e-5


def test_model_output_shape_wrapped():
    hp = dict(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_timesteps=5,
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
        log_closure_metrics_train=False,
        log_closure_metrics_val=True,
    )
    pl = PytorchLightningModule(**cast(Any, hp)).float()
    den = TorsionDenoiser(pl.node_dim, hp['hidden_dim'], hp['num_heads'], hp['num_layers'])
    x = torch.randn(2, 3, pl.node_dim)
    out = den(x)
    assert out.shape == (2, 3, N_LATENT)


def test_torsion_constants_consistent():
    assert_torsion_layout()


def test_delta_absent_everywhere():
    assert 'delta' not in TORSION_NAMES
    assert N_TORSIONS == 7
    assert N_LATENT == 8
    cl_text = Path(__file__).resolve().parents[1] / 'bridge_closure.py'
    assert 'delta' not in cl_text.read_text().lower()


def test_no_sincos_latent_in_model_source():
    text = Path(__file__).resolve().parents[1] / 'model.py'
    src = text.read_text()
    assert not re.search(r'N_TORSIONS\s*\*\s*2\s*\+\s*1', src)
    assert 'interleaved sin/cos' not in src.lower()


def test_chi_affects_coordinates():
    dev = torch.device('cpu')
    ri = torch.tensor([0])
    P = torch.tensor([0.15])
    tau = torch.tensor([0.33])
    chi_a = torch.tensor([-0.55])
    chi_b = torch.tensor([1.1])
    r_a = build_sugar_ring_grid_closed_torch(chi_a, P, tau, ri)
    r_b = build_sugar_ring_grid_closed_torch(chi_b, P, tau, ri)
    assert not torch.allclose(r_a["O4'"], r_b["O4'"], atol=1e-4)
    from torsion_geometry import _get_template_tensors

    tc = _get_template_tensors('cpu')
    n_atom = tc['chi_n'][ri].reshape(1, 3)
    c_atom = tc['chi_c'][ri].reshape(1, 3)
    meas = dihedral_rad_torch(
        r_a["O4'"].reshape(1, 3),
        r_a["C1'"].reshape(1, 3),
        n_atom,
        c_atom,
    )
    assert abs(float(meas.item()) - float(chi_a.item())) < 0.06


def test_chi_torch_pyrimidine_matches_measured_dihedral():
    ri = torch.tensor([1])
    P = torch.tensor([0.12])
    tau = torch.tensor([0.34])
    chi_tgt = torch.tensor([-0.4])
    r = build_sugar_ring_grid_closed_torch(chi_tgt, P, tau, ri)
    from torsion_geometry import _get_template_tensors

    tc = _get_template_tensors('cpu')
    n_atom = tc['chi_n'][ri].reshape(1, 3)
    c_atom = tc['chi_c'][ri].reshape(1, 3)
    meas = dihedral_rad_torch(
        r["O4'"].reshape(1, 3),
        r["C1'"].reshape(1, 3),
        n_atom,
        c_atom,
    )
    assert wrap_dihedral_diff_torch(meas, chi_tgt).abs().item() < 0.07


def test_world_local_roundtrip():
    torch.manual_seed(0)
    R = torch.randn(3, 3)
    q, _ = torch.linalg.qr(R)
    origin = torch.randn(3)
    local = torch.randn(5, 3)
    world = local @ q.T + origin
    local_rt = (world - origin) @ q
    assert torch.allclose(local, local_rt, atol=1e-5, rtol=1e-5)


def test_bridge_closure_wrapped_torsions():
    pred = torch.tensor([179.0 * math.pi / 180.0], dtype=torch.float64)
    tgt = torch.tensor([-179.0 * math.pi / 180.0], dtype=torch.float64)
    from torsion_geometry import wrap_dihedral_diff_torch

    err = wrap_dihedral_diff_torch(pred, tgt)
    err_deg = abs(float(err)) * 180.0 / math.pi
    assert abs(err_deg - 2.0) < 0.1


def test_smoke_perturb_processed_pt():
    root = Path(__file__).resolve().parents[2] / 'data'
    pts = sorted(root.rglob('*.pt'))
    pts = [p for p in pts if not p.name.startswith('pre_')]
    if not pts:
        return
    sample = torch.load(pts[0], weights_only=False, map_location='cpu')
    theta = sample.torsions[:1].float()
    tau = sample.tau_m[:1].float()
    if theta.shape[-1] != N_TORSIONS:
        return
    log0 = torch.log(tau.clamp(min=1e-6)).unsqueeze(-1)
    t = torch.tensor([0.5])
    pert = perturb_torsions(theta, log0, t, 0.05, 1.0, 0.05, 1.0)
    assert torch.isfinite(pert['theta_t']).all()
    assert torch.isfinite(pert['angular_score_target']).all()
