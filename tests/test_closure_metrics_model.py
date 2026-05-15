"""Bridge closure diagnostics on the Lightning module (no NameError when logging enabled)."""

import inspect
from typing import Any, cast

import torch
from torch_geometric.data import Batch, Data

from base2backbone.data import BACKBONE_ATOMS
from base2backbone.model import BackboneLightningModule
from base2backbone.torsion_constants import N_LATENT, N_TORSIONS
from base2backbone.score_diffusion import encode_torsions


def _minimal_train_batch(ws: int = 3, *, device=torch.device('cpu')) -> Batch:
    n_bb = len(BACKBONE_ATOMS)
    nt_o = torch.zeros(ws, 3, dtype=torch.float32, device=device)
    nt_f = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(ws, 3, 3).contiguous()
    pair_o = torch.zeros(ws, 3, dtype=torch.float32, device=device)
    pair_f = torch.zeros(ws, 3, 3, dtype=torch.float32, device=device)
    tidx = ws // 2
    o_t = nt_o[tidx]
    R_t = nt_f[tidx]
    rel_o = (nt_o - o_t) @ R_t
    rel_f = torch.einsum('ji,njk->nik', R_t, nt_f)
    prel_o = (pair_o - o_t) @ R_t
    prel_f = torch.einsum('ji,njk->nik', R_t, pair_f)
    is_target = torch.zeros(ws, dtype=torch.float32, device=device)
    is_target[tidx] = 1.0
    d = Data(
        nt_origins_world=nt_o,
        nt_frames_world=nt_f,
        pair_origins_world=pair_o,
        pair_frames_world=pair_f,
        torsions=torch.randn(ws, N_TORSIONS, device=device) * 0.05,
        torsion_mask=torch.ones(ws, N_TORSIONS, dtype=torch.bool, device=device),
        tau_m=torch.full((ws,), 0.45, dtype=torch.float32, device=device),
        tau_m_mask=torch.ones(ws, dtype=torch.bool, device=device),
        base_types=torch.nn.functional.one_hot(torch.zeros(ws, dtype=torch.long, device=device), 4).float(),
        has_pair_nt=torch.zeros(ws, dtype=torch.bool, device=device),
        chain_end_class=torch.nn.functional.one_hot(torch.zeros(ws, dtype=torch.long, device=device), 3).float(),
        central_nt_mask=torch.zeros(ws, dtype=torch.bool, device=device),
        is_chain_edge_nt=torch.zeros(ws, dtype=torch.bool, device=device),
        touches_chain_edge=torch.tensor(False, device=device),
        bb_xyz_world=torch.randn(ws, n_bb, 3, device=device) * 0.1,
        o3_prev_local=torch.zeros(ws, 3, dtype=torch.float32, device=device),
        o3_prev_valid=torch.zeros(ws, dtype=torch.bool, device=device),
        target_nt_idx=torch.tensor(tidx, dtype=torch.long, device=device),
        rel_origins=rel_o,
        rel_frames=rel_f.reshape(ws, 9),
        pair_rel_origins=prel_o,
        pair_rel_frames=prel_f.reshape(ws, 9),
    )
    d.is_target_nt = is_target.unsqueeze(-1)
    return Batch.from_data_list([d])


def test_bridge_closure_metrics_returns_finite_dict():
    hp = dict(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_timesteps=2,
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
    )
    mod = BackboneLightningModule(**cast(Any, hp)).eval()
    batch = _minimal_train_batch()
    th = torch.randn(1, N_TORSIONS)
    tau = torch.ones(1) * 0.45
    pred_x0 = encode_torsions(th, tau)
    out = mod._bridge_closure_metrics(pred_x0, batch)
    assert isinstance(out, dict)
    for k in (
        'closure_loss',
        'closure_bond_loss',
        'closure_angle_loss',
        'closure_torsion_loss',
        'closure_fail_rate',
    ):
        assert k in out
        assert torch.isfinite(out[k].reshape(-1)).all()


def test_validation_step_source_always_logs_val_closure():
    src = inspect.getsource(BackboneLightningModule.validation_step)
    assert 'log_closure_metrics_val' not in src
    assert '_bridge_closure_metrics' in src
    assert "f'diagnostics/val/{key}'" in src.replace(' ', '')


def test_tl_latent_width_matches_torch_geometry():
    assert N_LATENT == 8


def test_p_sample_loop_finite_shapes_with_stub_scores(monkeypatch):
    hp = dict(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_timesteps=2,
        batch_size=1,
        lr=1e-3,
        lr_scheduler=None,
        lr_scheduler_patience=1,
        lr_scheduler_threshold=0.1,
        lr_scheduler_cooldown=0,
        angular_sigma_min=0.05,
        angular_sigma_max=0.15,
        tau_sigma_min=0.05,
        tau_sigma_max=0.15,
        tau_loss_weight=1.0,
        score_loss_weighting='sigma2',
        weight_decay=0.01,
        closure_loss_weight=0.0,
        closure_bond_weight=1.0,
        closure_angle_weight=1.0,
        closure_torsion_weight=1.0,
    )
    mod = BackboneLightningModule(**cast(Any, hp)).eval()

    def _stub_fwd(self, prefix, batch, x_tl, lg, sc):  # noqa: ARG001
        return torch.zeros(
            batch.num_graphs,
            N_LATENT,
            device=batch.torsions.device,
            dtype=torch.float32,
        )

    monkeypatch.setattr(BackboneLightningModule, 'forward_score_network_from_prefix', _stub_fwd)
    batch = _minimal_train_batch()
    theta0, (pred_theta, pred_tau_m) = mod.p_sample_loop(batch)
    b = batch.num_graphs
    assert theta0.shape == (b, N_TORSIONS)
    assert pred_theta.shape == (b, N_TORSIONS)
    assert pred_tau_m.shape == (b,)
    assert torch.isfinite(pred_theta).all() and torch.isfinite(pred_tau_m).all()
    assert torch.all(pred_tau_m > 0)


def test_p_sample_loop_reuses_cached_sigma_grids(monkeypatch):
    hp = dict(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_timesteps=4,
        batch_size=1,
        lr=1e-3,
        lr_scheduler=None,
        lr_scheduler_patience=1,
        lr_scheduler_threshold=0.1,
        lr_scheduler_cooldown=0,
        angular_sigma_min=0.05,
        angular_sigma_max=0.15,
        tau_sigma_min=0.05,
        tau_sigma_max=0.15,
        tau_loss_weight=1.0,
        score_loss_weighting='sigma2',
        weight_decay=0.01,
        closure_loss_weight=0.0,
        closure_bond_weight=1.0,
        closure_angle_weight=1.0,
        closure_torsion_weight=1.0,
    )
    mod = BackboneLightningModule(**cast(Any, hp)).eval()
    grid_calls: list[tuple[float, float, int]] = []
    orig_grid = __import__('base2backbone.model', fromlist=['ve_sigma_grid']).ve_sigma_grid

    def _counting_grid(sigma_max, sigma_min, num_steps, *, device, dtype):
        grid_calls.append((sigma_max, sigma_min, num_steps))
        return orig_grid(sigma_max, sigma_min, num_steps, device=device, dtype=dtype)

    def _stub_fwd(self, prefix, batch, x_tl, lg, sc):  # noqa: ARG001
        return torch.zeros(
            batch.num_graphs,
            N_LATENT,
            device=batch.torsions.device,
            dtype=torch.float32,
        )

    monkeypatch.setattr('base2backbone.model.ve_sigma_grid', _counting_grid)
    monkeypatch.setattr(BackboneLightningModule, 'forward_score_network_from_prefix', _stub_fwd)
    batch = _minimal_train_batch()

    mod.p_sample_loop(batch)
    mod.p_sample_loop(batch)

    assert len(grid_calls) == 2


def test_rmsd_ignores_invalid_non_op_atoms(monkeypatch):
    hp = dict(
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        num_timesteps=2,
        batch_size=1,
        lr=1e-3,
        lr_scheduler=None,
        lr_scheduler_patience=1,
        lr_scheduler_threshold=0.1,
        lr_scheduler_cooldown=0,
        angular_sigma_min=0.05,
        angular_sigma_max=0.15,
        tau_sigma_min=0.05,
        tau_sigma_max=0.15,
        tau_loss_weight=1.0,
        score_loss_weighting='sigma2',
        weight_decay=0.01,
        closure_loss_weight=0.0,
        closure_bond_weight=1.0,
        closure_angle_weight=1.0,
        closure_torsion_weight=1.0,
    )
    mod = BackboneLightningModule(**cast(Any, hp)).eval()
    batch = _minimal_train_batch()
    b, ws = batch.num_graphs, batch.torsions.size(0) // batch.num_graphs
    target_idx = int(batch.target_nt_idx.item())
    keep_atom = BACKBONE_ATOMS.index('P')

    def _stub_builder(theta_w, tau_w, restype, origins_w, frames_w, mask):  # noqa: ARG001
        coords = batch.bb_xyz_world.view(b, ws, len(BACKBONE_ATOMS), 3).clone()
        coords[0, target_idx, keep_atom] = torch.tensor(
            [float('nan'), 0.0, 0.0],
            dtype=coords.dtype,
            device=coords.device,
        )
        return coords

    monkeypatch.setattr('base2backbone.model.build_batch_window_backbone_from_torsions', _stub_builder)

    rmsd = mod._compute_rmsd_per_graph_local(
        torch.zeros((1, N_TORSIONS), dtype=torch.float32),
        torch.ones((1,), dtype=torch.float32) * 0.45,
        batch,
    )

    assert torch.isfinite(rmsd).all()
    assert torch.allclose(rmsd, torch.zeros_like(rmsd))


test_sampler_shapes_finite = test_p_sample_loop_finite_shapes_with_stub_scores
