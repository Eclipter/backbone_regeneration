import math
import os.path as osp

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data import BACKBONE_ATOMS, BASE_TO_INDEX, N_CHAIN_END_CLASSES
from .bridge_closure import compute_bridge_closure_loss
from .geometry import build_batch_window_backbone_from_torsions
from .torsion_constants import (LOG_TAU_M_MAX, LOG_TAU_M_MIN, TAU_M_MAX,
                                TAU_M_MIN, N_LATENT, N_TORSIONS)
from .score_diffusion import (decode_torsions, encode_torsions,
                                      estimate_latent_from_ve_score,
                                      perturb_torsions, reverse_ve_score_step,
                                      sigma_schedule, ve_sigma_grid,
                                      weighted_score_mse, wrap_angle)

# Same dimension constant name as in ONNX companion JSON (`N_TORSIONS_LATENT`).
N_TORSIONS_LATENT = N_LATENT


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TorsionScoreNetwork(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}) '
                'for nn.MultiheadAttention'
            )
        self.in_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True disables nested-tensor fast path; set explicitly to avoid the warning.
        self.tr = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.out = nn.Linear(hidden_dim, N_LATENT)

    def forward(self, x):
        h = self.in_mlp(x)
        h = self.tr(h)
        return self.out(h)


_N = ('torsions', 'tau_m', 'bb_xyz_world', 'nt_origins_world', 'nt_frames_world', 'base_types',
      'torsion_mask', 'target_nt_idx')


def _require_window_batch_fields(batch) -> None:
    for name in _N:
        if not hasattr(batch, name):
            raise ValueError(
                f'Batch missing `{name}` for window-level geometry (wrapped torsions layout).',
            )


def _zero_bridge_closure_metrics(pred_x0: torch.Tensor) -> dict[str, torch.Tensor]:
    z = pred_x0.sum() * 0.0
    zf = pred_x0.new_zeros(())
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


class BackboneLightningModule(pl.LightningModule):
    def __init__(
        self, hidden_dim, num_heads, num_layers, num_timesteps, batch_size, lr,
        lr_scheduler, lr_scheduler_patience, lr_scheduler_threshold, lr_scheduler_cooldown,
        angular_sigma_min, angular_sigma_max,
        tau_sigma_min, tau_sigma_max,
        tau_loss_weight, weight_decay,
        score_loss_weighting: str = 'sigma2',
        closure_loss_weight: float = 0.0,
        closure_bond_weight: float = 1.0,
        closure_angle_weight: float = 1.0,
        closure_torsion_weight: float = 1.0,
        log_closure_metrics_train: bool = False,
        log_closure_metrics_val: bool = True,
        log_tau_init_noise_scale: float | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.time_emb_dim = 32
        self.time_mlp = SinusoidalPositionalEmbeddings(self.time_emb_dim)
        # +1 for per-atom is_target_nt flag
        self.node_dim = (
            3
            + 9
            + 3   # pair_rel_origins
            + 9   # pair_rel_frames (flattened 3x3)
            + len(BASE_TO_INDEX)
            + 1
            + N_CHAIN_END_CLASSES
            + 1
            + N_LATENT
            + self.time_emb_dim
            + N_TORSIONS
            + N_LATENT
        )

        self.score_network = TorsionScoreNetwork(
            node_dim=self.node_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[no-untyped-def]
        # Support checkpoints saved before the `score_network` rename and before torch.compile wrapping.
        renamed_state_dict = {}
        for key, value in state_dict.items():
            key_str = str(key)
            if key_str.startswith('denoiser._orig_mod.'):
                key_str = 'score_network.' + key_str[len('denoiser._orig_mod.'):]
            elif key_str.startswith('denoiser.'):
                key_str = 'score_network.' + key_str[len('denoiser.'):]
            elif key_str.startswith('score_network._orig_mod.'):
                key_str = 'score_network.' + key_str[len('score_network._orig_mod.'):]
            renamed_state_dict[key_str] = value
        state_dict = renamed_state_dict
        return super().load_state_dict(state_dict, strict=strict)

    def on_test_start(self):
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        logger = self.trainer.logger
        dm = self.trainer.datamodule  # type: ignore[attr-defined]
        log_dir = getattr(logger, 'log_dir', None)
        if log_dir is None or dm is None:
            return
        torch.save(dm.test_dataset, osp.join(log_dir, 'test_dataset.pt'))

    def _b_ws(self, batch):
        b = batch.num_graphs
        n = batch.torsions.size(0)
        ws = n // b
        return b, ws

    def _theta_mask_target(self, batch):
        b, ws = self._b_ws(batch)
        tors = batch.torsions.view(b, ws, N_TORSIONS)
        m = batch.torsion_mask.view(b, ws, N_TORSIONS)
        ti = batch.target_nt_idx.long()
        bi = torch.arange(b, device=tors.device)
        theta0 = tors[bi, ti]
        mk = m[bi, ti]
        tau_flat = batch.tau_m.view(b, ws)
        tmk_flat = batch.tau_m_mask.view(b, ws)
        tau0 = tau_flat[bi, ti]
        tau_mk = tmk_flat[bi, ti]
        return theta0, mk, tau0, tau_mk, ti

    def _bridge_closure_metrics(self, pred_x0: torch.Tensor, batch) -> dict:
        """World-window backbone + phosphate bridges; closure vs GT torsions (neighbor residues GT)."""
        b, ws = self._b_ws(batch)
        ti = batch.target_nt_idx.long()
        bi = torch.arange(b, device=pred_x0.device)

        pred_theta, pred_tau_m = decode_torsions(pred_x0)

        tors_m = batch.torsions.view(b, ws, N_TORSIONS).clone()
        tau_m = batch.tau_m.view(b, ws).clone()
        tors_m[bi, ti] = pred_theta
        tau_m[bi, ti] = pred_tau_m.clamp(min=TAU_M_MIN, max=TAU_M_MAX)

        restype = batch.base_types.view(b, ws, len(BASE_TO_INDEX)).argmax(-1)
        mask = batch.torsion_mask.view(b, ws, N_TORSIONS)
        origins = batch.nt_origins_world.view(b, ws, 3)
        frames = batch.nt_frames_world.view(b, ws, 3, 3)

        bb = build_batch_window_backbone_from_torsions(
            tors_m.float(),
            tau_m.float(),
            restype.long(),
            origins.float(),
            frames.float(),
            mask,
        )

        pair_mask = torch.zeros(b, ws - 1, dtype=torch.bool, device=pred_x0.device)
        mk_l = ti > 0
        pair_mask[bi[mk_l], ti[mk_l] - 1] = True
        mk_r = ti < ws - 1
        pair_mask[bi[mk_r], ti[mk_r]] = True

        same_chain_mask = None

        weights = {
            'bond': float(self.hparams.get('closure_bond_weight', 1.0)),
            'angle': float(self.hparams.get('closure_angle_weight', 1.0)),
            'torsion': float(self.hparams.get('closure_torsion_weight', 1.0)),
        }

        return compute_bridge_closure_loss(
            bb,
            batch.torsions.view(b, ws, N_TORSIONS),
            mask,
            restype.long(),
            same_chain_mask=same_chain_mask,
            valid_pair_mask=pair_mask,
            weights=weights,
            grad_prop_tensor=pred_theta,
        )

    def _closure_loss(self, pred_x0: torch.Tensor, batch) -> torch.Tensor:
        return self._bridge_closure_metrics(pred_x0, batch)['closure_loss']

    def _build_x(self, batch, x_t_latent, log_sigma_per_graph, sc):
        b, ws = self._b_ws(batch)
        rel_o = batch.rel_origins.view(b, ws, 3)
        rel_R = batch.rel_frames.view(b, ws, 9)
        pair_o = batch.pair_rel_origins.view(b, ws, 3)
        pair_R = batch.pair_rel_frames.view(b, ws, 3, 3).reshape(b, ws, 9)
        base = batch.base_types.view(b, ws, len(BASE_TO_INDEX))
        hp = batch.has_pair_nt.view(b, ws, 1).float()
        ce = batch.chain_end_class.view(b, ws, N_CHAIN_END_CLASSES)
        it = batch.is_target_nt.view(b, ws, 1)
        tidx = batch.target_nt_idx.long()
        bi = torch.arange(b, device=rel_o.device)
        pad = torch.zeros(
            b, ws,
            N_LATENT + self.time_emb_dim + N_TORSIONS + N_LATENT,
            device=rel_o.device, dtype=rel_o.dtype,
        )
        te_all = self.time_mlp(log_sigma_per_graph.float())
        o = 0
        pad[bi, tidx, o:o + N_LATENT] = x_t_latent
        o += N_LATENT
        pad[bi, tidx, o:o + self.time_emb_dim] = te_all
        o += self.time_emb_dim
        pad[bi, tidx, o:o + N_TORSIONS] = (
            batch.torsion_mask.view(b, ws, N_TORSIONS)[bi, tidx].float()
        )
        o += N_TORSIONS
        pad[bi, tidx, o:o + N_LATENT] = sc
        return torch.cat([rel_o, rel_R, pair_o, pair_R, base, hp, ce, it, pad], dim=-1)

    def forward_score_network(self, batch, x_t_latent, log_sigma_per_graph, sc):
        b, _ = self._b_ws(batch)
        x = self._build_x(batch, x_t_latent, log_sigma_per_graph, sc)
        score_all = self.score_network(x)
        bi = torch.arange(b, device=score_all.device)
        return score_all[bi, batch.target_nt_idx.long()]

    def _should_compute_closure_train(self) -> bool:
        return (
            float(self.hparams.get('closure_loss_weight', 0.0)) > 0.0
            or bool(self.hparams.get('log_closure_metrics_train', False))
        )

    def training_step(self, batch, batch_idx):
        theta0, m, tau0, tau_mk, _ = self._theta_mask_target(batch)
        if theta0.shape[-1] != N_TORSIONS:
            raise ValueError(
                f'Expected torsions last dim {N_TORSIONS}, got {theta0.shape[-1]}',
            )
        b = batch.num_graphs
        tau_safe = torch.where(
            tau_mk,
            tau0.clamp(min=TAU_M_MIN, max=TAU_M_MAX),
            torch.full_like(tau0, 0.611),
        )
        log_tau_0 = torch.log(tau_safe.clamp(min=TAU_M_MIN, max=TAU_M_MAX)).unsqueeze(-1)
        t_unif = torch.rand((b,), device=self.device, dtype=torch.float32)
        pert = perturb_torsions(
            theta0,
            log_tau_0,
            t_unif,
            float(self.hparams['angular_sigma_min']),
            float(self.hparams['angular_sigma_max']),
            float(self.hparams['tau_sigma_min']),
            float(self.hparams['tau_sigma_max']),
        )
        log_tau_t = pert['log_tau_t']
        x_t = torch.cat([pert['theta_t'], log_tau_t], dim=-1)
        mask_theta = m.float()
        mask_tau = tau_mk.float().unsqueeze(-1)

        sc = torch.zeros_like(x_t)
        sigma_theta_b = sigma_schedule(
            t_unif,
            float(self.hparams['angular_sigma_min']),
            float(self.hparams['angular_sigma_max']),
        )
        log_sigma_cond = torch.log(sigma_theta_b.clamp(min=1e-8))
        pred = self.forward_score_network(batch, x_t, log_sigma_cond, sc)
        sw = str(self.hparams.get('score_loss_weighting', 'sigma2'))
        lam_theta = (pert['sigma_theta'] ** 2) if sw == 'sigma2' else None
        lam_tau = (pert['sigma_tau'] ** 2) if sw == 'sigma2' else None
        mse_theta = weighted_score_mse(
            pred[..., :N_TORSIONS],
            pert['angular_score_target'],
            mask_theta,
            lam_theta,
            weighting='sigma2' if sw == 'sigma2' else 'none',
        )
        mse_tau = weighted_score_mse(
            pred[..., N_TORSIONS:N_LATENT],
            pert['tau_score_target'],
            mask_tau,
            lam_tau,
            weighting='sigma2' if sw == 'sigma2' else 'none',
        )
        tw = float(self.hparams.get('tau_loss_weight', 1.0))
        mse = mse_theta + tw * mse_tau
        pred_x0_cl = estimate_latent_from_ve_score(
            pert['theta_t'],
            log_tau_t,
            pred,
            pert['sigma_theta'],
            pert['sigma_tau'],
        )
        if self._should_compute_closure_train():
            clo_metrics = self._bridge_closure_metrics(pred_x0_cl, batch)
        else:
            clo_metrics = _zero_bridge_closure_metrics(pred_x0_cl)
        cl = clo_metrics['closure_loss']
        loss = mse + float(self.hparams.get('closure_loss_weight', 0.0)) * cl
        self.log(
            'train_loss', mse,
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'train/angular_score_loss', mse_theta,
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'train/tau_score_loss', mse_tau,
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'train/score_norm', pred.square().mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        tgt = torch.cat(
            [pert['angular_score_target'], pert['tau_score_target'].reshape(b, N_LATENT - N_TORSIONS)],
            dim=-1,
        )
        self.log(
            'train/target_score_norm', tgt.square().mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'train/sigma_theta_mean',
            pert['sigma_theta'].mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'train/sigma_tau_mean',
            pert['sigma_tau'].mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'train_closure', cl,
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        for key in (
            'closure_loss',
            'closure_bond_loss',
            'closure_angle_loss',
            'closure_torsion_loss',
            'closure_valid_bridge_fraction',
            'closure_num_valid_bridges',
            'closure_fail_rate',
            'bridge_bond_mae',
            'bridge_angle_mae_deg',
            'bridge_torsion_mae_deg',
        ):
            self.log(
                key, clo_metrics[key],
                on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
            )
        return loss

    def on_train_epoch_end(self):
        self._write_epoch_scalars([
            'train_loss',
            'train/angular_score_loss',
            'train/tau_score_loss',
            'train/score_norm',
            'train/target_score_norm',
            'train/sigma_theta_mean',
            'train/sigma_tau_mean',
            'train_closure',
            'closure_loss',
            'closure_bond_loss',
            'closure_angle_loss',
            'closure_torsion_loss',
            'closure_valid_bridge_fraction',
            'closure_num_valid_bridges',
            'closure_fail_rate',
            'bridge_bond_mae',
            'bridge_angle_mae_deg',
            'bridge_torsion_mae_deg',
        ])

    def _is_edge_target(self, batch):
        """Bool mask [b]: True if the target nucleotide is a chain edge."""
        b, ws = self._b_ws(batch)
        edge_mask = batch.is_chain_edge_nt.view(b, ws)
        bi = torch.arange(b, device=batch.target_nt_idx.device)
        return edge_mask[bi, batch.target_nt_idx.long()]

    def _write_epoch_scalars(self, keys):
        """Read accumulated callback_metrics and write to TensorBoard via add_scalar."""
        if (
            self.trainer is None
            or not self.trainer.is_global_zero
            or not isinstance(self.logger, TensorBoardLogger)
        ):
            return
        metrics = self.trainer.callback_metrics
        for key in keys:
            if key in metrics:
                self.logger.experiment.add_scalar(key, metrics[key], self.current_epoch)

    def _write_rmsd_scalars(self, prefix):
        if prefix == 'val':
            keys = [
                'val_rmsd',
                'val_rmsd_central',
                'val_rmsd_edge',
            ]
        else:
            keys = [
                'test_rmsd',
                'test_rmsd_central',
                'test_rmsd_edge',
            ]
        self._write_epoch_scalars(keys)

    def _val_closure_tensorboard_keys(self):
        """TensorBoard extras when validation logs bridge closure diagnostics."""
        if not bool(self.hparams.get('log_closure_metrics_val', False)):
            return []
        return [
            'val/closure_loss',
            'val/closure_bond_loss',
            'val/closure_angle_loss',
            'val/closure_torsion_loss',
            'val/closure_valid_bridge_fraction',
            'val/closure_num_valid_bridges',
            'val/closure_fail_rate',
            'val/bridge_bond_mae',
            'val/bridge_angle_mae_deg',
            'val/bridge_torsion_mae_deg',
        ]

    def on_validation_epoch_end(self):
        self._write_rmsd_scalars('val')
        self._write_epoch_scalars(self._val_closure_tensorboard_keys())

    def on_test_epoch_end(self):
        self._write_rmsd_scalars('test')

    @torch.no_grad()
    def p_sample_loop(self, batch):
        theta0, _, _, _, _ = self._theta_mask_target(batch)
        b = batch.num_graphs
        dev = self.device
        dtype = torch.float32
        num_steps = int(self.hparams['num_timesteps'])
        sig_theta = ve_sigma_grid(
            float(self.hparams['angular_sigma_max']),
            float(self.hparams['angular_sigma_min']),
            num_steps,
            device=torch.device(dev),
            dtype=dtype,
        )
        sig_tau = ve_sigma_grid(
            float(self.hparams['tau_sigma_max']),
            float(self.hparams['tau_sigma_min']),
            num_steps,
            device=torch.device(dev),
            dtype=dtype,
        )
        theta = wrap_angle(
            torch.rand(b, N_TORSIONS, device=dev, dtype=dtype) * (2.0 * math.pi) - math.pi,
        )
        lt_scale = self.hparams.get('log_tau_init_noise_scale')
        if lt_scale is None:
            lt_scale = float(self.hparams['tau_sigma_max'])
        logt = torch.randn(b, 1, device=dev, dtype=dtype) * float(lt_scale)
        logt = logt.clamp(LOG_TAU_M_MIN, LOG_TAU_M_MAX)
        x_t = torch.cat([theta, logt], dim=-1)
        sc = torch.zeros_like(x_t)
        zero = torch.tensor(0.0, device=dev, dtype=dtype)
        n_sg = sig_theta.shape[0]
        for i in range(n_sg):
            sigma_cur_th = sig_theta[i]
            sigma_next_th = sig_theta[i + 1] if i + 1 < n_sg else zero
            sigma_cur_tau = sig_tau[i]
            sigma_next_tau = sig_tau[i + 1] if i + 1 < n_sg else zero
            log_s = torch.full(
                (b,),
                math.log(float(sigma_cur_th.clamp(min=1e-8))),
                device=dev,
                dtype=dtype,
            )
            pred = self.forward_score_network(batch, x_t, log_s, sc)
            theta, logt = reverse_ve_score_step(
                x_t[..., :N_TORSIONS],
                x_t[..., N_TORSIONS:N_LATENT],
                pred,
                sigma_cur_th,
                sigma_next_th,
                sigma_cur_tau,
                sigma_next_tau,
            )
            x_t = torch.cat([theta, logt], dim=-1)
            sc = torch.zeros_like(x_t)
            if not torch.isfinite(x_t).all():
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=math.pi, neginf=-math.pi)
                theta = wrap_angle(x_t[..., :N_TORSIONS])
                logt = x_t[..., N_TORSIONS:N_LATENT]
                x_t = torch.cat([theta, logt], dim=-1)
        pred_theta, pred_tau_m = decode_torsions(x_t)
        return theta0, (pred_theta, pred_tau_m)

    def _compute_rmsd_per_graph_local(
        self,
        pred_torsions: torch.Tensor,
        pred_tau_m: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        _require_window_batch_fields(batch)
        if batch.torsions.shape[-1] != N_TORSIONS:
            raise ValueError(
                f'Expected batch.torsions last dim {N_TORSIONS}; '
                f'got {batch.torsions.shape[-1]}.',
            )

        b, ws = self._b_ws(batch)
        ti = batch.target_nt_idx.long()
        bi = torch.arange(b, device=ti.device)
        dev = pred_torsions.device

        n_bb = len(BACKBONE_ATOMS)
        bb_world_all = batch.bb_xyz_world.view(b, ws, n_bb, 3)
        gt_bb_world = bb_world_all[bi, ti]
        origins = batch.nt_origins_world.view(b, ws, 3)[bi, ti]
        frames = batch.nt_frames_world.view(b, ws, 3, 3)[bi, ti]
        gt_local = (gt_bb_world - origins.unsqueeze(1)) @ frames

        theta_w = batch.torsions.view(b, ws, N_TORSIONS).clone()
        tau_w = batch.tau_m.view(b, ws).clone()
        theta_w[bi, ti] = pred_torsions.float()
        tau_w[bi, ti] = pred_tau_m.clamp(min=TAU_M_MIN, max=TAU_M_MAX).float()

        restype = batch.base_types.view(b, ws, len(BASE_TO_INDEX)).argmax(-1)
        mask = batch.torsion_mask.view(b, ws, N_TORSIONS)
        origins_w = batch.nt_origins_world.view(b, ws, 3)
        frames_w = batch.nt_frames_world.view(b, ws, 3, 3)

        coords_w = build_batch_window_backbone_from_torsions(
            theta_w,
            tau_w,
            restype.long(),
            origins_w.float(),
            frames_w.float(),
            mask,
        )
        pred_bb_world = coords_w[bi, ti]
        pred_local = (pred_bb_world - origins.unsqueeze(1)) @ frames

        j1 = BACKBONE_ATOMS.index('OP1')
        j2 = BACKBONE_ATOMS.index('OP2')
        contrib = torch.zeros(b, device=dev, dtype=torch.float64)
        count = torch.zeros(b, device=dev, dtype=torch.float64)

        for j, nm in enumerate(BACKBONE_ATOMS):
            if nm in ('OP1', 'OP2'):
                continue
            pred_xyz = pred_local[:, j, :].to(dtype=torch.float64)
            gt_xyz = gt_local[:, j, :].to(dtype=torch.float64)
            valid = torch.isfinite(pred_xyz).all(dim=-1) & torch.isfinite(gt_xyz).all(
                dim=-1,
            )
            diff = pred_xyz - gt_xyz
            sq = torch.einsum('bi,bi->b', diff, diff)
            contrib = contrib + sq * valid.to(dtype=torch.float64)
            count = count + valid.to(dtype=torch.float64)

        p1 = pred_local[:, j1, :].to(dtype=torch.float64)
        p2 = pred_local[:, j2, :].to(dtype=torch.float64)
        g1 = gt_local[:, j1, :].to(dtype=torch.float64)
        g2 = gt_local[:, j2, :].to(dtype=torch.float64)
        v_op = (
            torch.isfinite(p1).all(dim=-1)
            & torch.isfinite(p2).all(dim=-1)
            & torch.isfinite(g1).all(dim=-1)
            & torch.isfinite(g2).all(dim=-1)
        )
        d1s, d2s = p1 - g1, p2 - g2
        d1w, d2w = p1 - g2, p2 - g1
        d_str = torch.einsum('bi,bi->b', d1s, d1s) + torch.einsum('bi,bi->b', d2s, d2s)
        d_swp = torch.einsum('bi,bi->b', d1w, d1w) + torch.einsum('bi,bi->b', d2w, d2w)
        op_sq = torch.minimum(d_str, d_swp) * 0.5
        contrib = contrib + op_sq * v_op.to(dtype=torch.float64)
        count = count + v_op.to(dtype=torch.float64)

        out = torch.full((b,), float('nan'), device=dev, dtype=torch.float64)
        ok = count > 0
        out = torch.where(ok, torch.sqrt(contrib / count), out)
        return out.to(pred_torsions.dtype)

    def _log_rmsd(
        self, prefix: str, pred_theta: torch.Tensor, pred_tau_m: torch.Tensor, batch,
    ):
        """Oracle-context window RMSD (GT neighbor torsions, predicted target); TensorBoard step log."""
        per_graph_rmsd = self._compute_rmsd_per_graph_local(
            pred_theta, pred_tau_m, batch,
        )
        is_edge = self._is_edge_target(batch)
        finite = torch.isfinite(per_graph_rmsd)

        for name, mask in [
            (f'{prefix}_rmsd', finite),
            (f'{prefix}_rmsd_central', (~is_edge) & finite),
            (f'{prefix}_rmsd_edge', is_edge & finite),
        ]:
            if mask.any():
                self.log(
                    name, per_graph_rmsd[mask].mean(),
                    on_epoch=True, on_step=False, sync_dist=True,
                    batch_size=max(int(mask.sum().item()), 1), logger=False,
                )

    def validation_step(self, batch, batch_idx):
        _, (pred_theta, pred_tau_m) = self.p_sample_loop(batch)
        self._log_rmsd('val', pred_theta, pred_tau_m, batch)
        if bool(self.hparams.get('log_closure_metrics_val', False)):
            pred_x0 = encode_torsions(pred_theta, pred_tau_m)
            clo = self._bridge_closure_metrics(pred_x0, batch)
            for key, val in clo.items():
                self.log(
                    f'val/{key}',
                    val,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    batch_size=batch.num_graphs,
                    logger=False,
                )

    def test_step(self, batch, _):
        _, (pred_theta, pred_tau_m) = self.p_sample_loop(batch)
        self._log_rmsd('test', pred_theta, pred_tau_m, batch)

    @torch.no_grad()
    def sample(self, batch):
        _, (pred_theta, pred_tau_m) = self.p_sample_loop(batch)
        return pred_theta, pred_tau_m

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )
        if self.hparams['lr_scheduler'] is None:
            return optimizer
        ls = self.hparams['lr_scheduler']
        if ls != 'ReduceLROnPlateau':
            raise NotImplementedError(f'unknown lr scheduler: {ls}')
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=self.hparams['lr_scheduler_patience'],
            threshold=self.hparams['lr_scheduler_threshold'],
            cooldown=self.hparams['lr_scheduler_cooldown'],
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_rmsd'},
        }
