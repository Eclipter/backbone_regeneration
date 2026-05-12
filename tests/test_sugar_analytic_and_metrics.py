"""Analytic sugar ring closure (no φ grid), builder invariants, and wrapped score layout checks."""

import inspect
import re
from pathlib import Path
from typing import Any, cast

import torch

from bbregen.model import PytorchLightningModule, TorsionDenoiser
from bbregen.torsion_constants import N_LATENT, N_TORSIONS
from bbregen.torsion_geometry import (
    _get_template_tensors,
    build_sugar_ring_closed_torch,
    pseudorotation_to_nus_torch,
    sugar_ring_torsions_torch,
    wrap_dihedral_diff_torch,
)
from bbregen.wrapped_score_diffusion import (
    decode_torsions,
    encode_torsions,
    perturb_torsions,
    weighted_score_mse,
)


def test_no_sugar_grid_solver():
    import bbregen.torsion_geometry as tg

    src = Path(tg.__file__).read_text()
    assert '_PHI_SUGAR_GRID' not in src
    assert 'build_sugar_ring_grid' not in src
    body = inspect.getsource(tg.build_sugar_ring_closed_torch)
    assert 'linspace' not in body


def test_sugar_ring_analytic_lengths_angles(device='cpu'):
    dev = torch.device(device)
    tc = _get_template_tensors(str(dev))
    for ri_ in (0, 1, 2, 3):
        ri = torch.tensor([ri_], device=dev)
        P = torch.tensor([0.12 - 0.04 * ri_], device=dev)
        tau_m = torch.tensor([0.28 + 0.02 * ri_], device=dev)
        chi = torch.zeros(1, device=dev)
        ring = build_sugar_ring_closed_torch(chi, P, tau_m, ri)
        assert torch.isfinite(ring["O4'"]).all() and torch.isfinite(ring["C4'"]).all()

        bl_o4_c4 = tc['bl_o4_c4'][ri]
        assert ((ring["C4'"] - ring["O4'"]).norm(dim=-1) - bl_o4_c4).abs().item() < 0.5

        def _ang(a, b, c):
            ba = a - b
            bc = c - b
            x = (ba * bc).sum(dim=-1) / (ba.norm(dim=-1) * bc.norm(dim=-1)).clamp(min=1e-8)
            return torch.acos(x.clamp(-1.0, 1.0))

        tgt_o4 = _ang(ring["C2'"], ring["C1'"], ring["O4'"])
        cur = tc['ba_o4_c1_c2'][ri]
        assert wrap_dihedral_diff_torch(tgt_o4, cur).abs().item() < 0.35


def test_sugar_ring_analytic_torsions_match_pseudorotation(device='cpu'):
    dev = torch.device(device)
    restype_idx = torch.tensor([0, 1, 2, 3], device=dev)
    P = torch.linspace(-0.9, 0.9, 4, device=dev)
    tau = torch.full((4,), 0.34, device=dev)
    chi = torch.zeros(4, device=dev)
    ring = build_sugar_ring_closed_torch(chi, P, tau, restype_idx)
    nu_tgt = pseudorotation_to_nus_torch(P, tau)
    nu_act = sugar_ring_torsions_torch(ring)
    err = wrap_dihedral_diff_torch(nu_act, nu_tgt).abs().max().item()
    assert err < 0.55


def test_chi_affects_coordinates_without_changing_pucker(device='cpu'):
    dev = torch.device(device)
    ri = torch.tensor([1], device=dev, dtype=torch.long)
    P = torch.tensor([0.16], device=dev)
    tau_m = torch.tensor([0.32], device=dev)
    chi_a = torch.tensor([-0.6], device=dev)
    chi_b = torch.tensor([0.9], device=dev)
    ring_a = build_sugar_ring_closed_torch(chi_a, P, tau_m, ri)
    ring_b = build_sugar_ring_closed_torch(chi_b, P, tau_m, ri)
    nu_a = sugar_ring_torsions_torch(ring_a)
    nu_b = sugar_ring_torsions_torch(ring_b)
    assert wrap_dihedral_diff_torch(nu_a, nu_b).abs().max().item() < 1e-2
    for nm in ("O4'", "C2'", "C3'", "C4'"):
        assert not torch.allclose(ring_a[nm], ring_b[nm], atol=1e-4, rtol=1e-4)
    tc = _get_template_tensors(str(dev))
    n_atom = tc['chi_n'][ri]
    c_atom = tc['chi_c'][ri]
    from bbregen.torsion_geometry import dihedral_rad_torch

    mchi = dihedral_rad_torch(
        ring_b["O4'"].reshape(1, 3),
        ring_b["C1'"].reshape(1, 3),
        n_atom.reshape(1, 3),
        c_atom.reshape(1, 3),
    )
    assert wrap_dihedral_diff_torch(mchi, chi_b).abs().item() < 0.09


def test_builder_no_delta():
    src = inspect.getsource(build_sugar_ring_closed_torch)
    assert 'delta' not in src.lower()
    assert 'delta' not in inspect.signature(build_sugar_ring_closed_torch).parameters


def test_wrapped_score_diffusion_still_intact():
    assert N_TORSIONS == 7
    assert N_LATENT == 8
    wsd = Path(__file__).resolve().parents[1] / 'src' / 'bbregen' / 'wrapped_score_diffusion.py'
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
        log_closure_metrics_train=False,
        log_closure_metrics_val=False,
    )
    pl = PytorchLightningModule(**cast(Any, hp)).float()
    den = TorsionDenoiser(pl.node_dim, hp['hidden_dim'], hp['num_heads'], hp['num_layers'])
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
