import math
import os.path as osp

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch

from .bridge_closure import (CLOSURE_SIGMA_ANGLE_RAD, CLOSURE_SIGMA_BOND_A,
                             CLOSURE_SIGMA_TORSION_RAD,
                             compute_bridge_closure_loss)
from .data import (BACKBONE_ATOMS, BASE_TO_INDEX, CHAIN_END_CLASS_3_PRIME,
                   CHAIN_END_CLASS_5_PRIME, N_CHAIN_END_CLASSES)
from .geometry import build_batch_window_backbone_from_torsions
from .score_diffusion import (decode_torsions, encode_torsions,
                              estimate_latent_from_ve_score, perturb_torsions,
                              reverse_ve_score_ode_step, sigma_schedule,
                              ve_sigma_grid, weighted_score_mse, wrap_angle)
from .torsion_constants import (LOG_TAU_M_MAX, LOG_TAU_M_MIN, N_LATENT,
                                N_TORSIONS, TAU_M_MAX, TAU_M_MIN, TOR_ALPHA,
                                TOR_EPS, TOR_ZETA)

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
        # Per-component σ for closure-loss normalization (squared deviations are divided by σ²).
        # Defaults mirror the module constants in bridge_closure.py for backwards compatibility.
        closure_sigma_bond_a: float = CLOSURE_SIGMA_BOND_A,
        closure_sigma_angle_rad: float = CLOSURE_SIGMA_ANGLE_RAD,
        closure_sigma_torsion_rad: float = CLOSURE_SIGMA_TORSION_RAD,
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
        self._sampling_sigma_cache: dict[
            tuple[str, int | None, torch.dtype, int, float, float, float, float],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._rmsd_keep_cache: dict[tuple[str, int | None], torch.Tensor] = {}

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[no-untyped-def]
        # Align checkpoint keys with eager vs torch.compile-wrapped `score_network` (OptimizedModule uses `_orig_mod`).
        compiled_score = getattr(self.score_network, '_orig_mod', None) is not None
        renamed_state_dict = {}
        for key, value in state_dict.items():
            key_str = str(key)
            if key_str.startswith('denoiser._orig_mod.'):
                key_str = 'score_network.' + key_str[len('denoiser._orig_mod.'):]
            elif key_str.startswith('denoiser.'):
                key_str = 'score_network.' + key_str[len('denoiser.'):]
            elif key_str.startswith('score_network._orig_mod.') and not compiled_score:
                key_str = 'score_network.' + key_str[len('score_network._orig_mod.'):]
            elif (
                compiled_score
                and key_str.startswith('score_network.')
                and not key_str[len('score_network.'):].startswith('_orig_mod.')
            ):
                key_str = 'score_network._orig_mod.' + key_str[len('score_network.'):]
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
        geometry = {
            'sigma_bond': float(self.hparams.get('closure_sigma_bond_a', CLOSURE_SIGMA_BOND_A)),
            'sigma_angle_rad': float(self.hparams.get('closure_sigma_angle_rad', CLOSURE_SIGMA_ANGLE_RAD)),
            'sigma_torsion_rad': float(self.hparams.get('closure_sigma_torsion_rad', CLOSURE_SIGMA_TORSION_RAD)),
        }

        return compute_bridge_closure_loss(
            bb,
            batch.torsions.view(b, ws, N_TORSIONS),
            mask,
            restype.long(),
            same_chain_mask=same_chain_mask,
            valid_pair_mask=pair_mask,
            geometry=geometry,
            weights=weights,
            grad_prop_tensor=pred_theta,
        )

    def _closure_loss(self, pred_x0: torch.Tensor, batch) -> torch.Tensor:
        return self._bridge_closure_metrics(pred_x0, batch)['closure_loss']

    def _build_static_x_prefix(self, batch):
        b, ws = self._b_ws(batch)
        rel_o = batch.rel_origins.view(b, ws, 3)
        rel_R = batch.rel_frames.view(b, ws, 9)
        pair_o = batch.pair_rel_origins.view(b, ws, 3)
        pair_R = batch.pair_rel_frames.view(b, ws, 3, 3).reshape(b, ws, 9)
        base = batch.base_types.view(b, ws, len(BASE_TO_INDEX))
        hp = batch.has_pair_nt.view(b, ws, 1).float()
        ce = batch.chain_end_class.view(b, ws, N_CHAIN_END_CLASSES)
        it = batch.is_target_nt.view(b, ws, 1)
        return torch.cat([rel_o, rel_R, pair_o, pair_R, base, hp, ce, it], dim=-1)

    def _build_x_from_prefix(self, prefix, batch, x_t_latent, log_sigma_per_graph, sc):
        b, ws = self._b_ws(batch)
        tidx = batch.target_nt_idx.long()
        bi = torch.arange(b, device=prefix.device)
        pad = torch.zeros(
            b, ws,
            N_LATENT + self.time_emb_dim + N_TORSIONS + N_LATENT,
            device=prefix.device, dtype=prefix.dtype,
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
        return torch.cat([prefix, pad], dim=-1)

    def _build_x(self, batch, x_t_latent, log_sigma_per_graph, sc):
        prefix = self._build_static_x_prefix(batch)
        return self._build_x_from_prefix(prefix, batch, x_t_latent, log_sigma_per_graph, sc)

    @staticmethod
    def _prepare_target_view(sample_data, target_idx: int):
        ws = int(sample_data.nt_origins_world.size(0))
        dc = sample_data.clone()
        dev = dc.nt_origins_world.device
        dc.target_nt_idx = torch.tensor(target_idx, dtype=torch.long, device=dev)
        is_target = torch.zeros(ws, dtype=torch.float32, device=dev)
        is_target[target_idx] = 1.0
        dc.is_target_nt = is_target.unsqueeze(-1)
        origin_t = dc.nt_origins_world[target_idx]
        frame_t = dc.nt_frames_world[target_idx]
        dc.rel_origins = ((dc.nt_origins_world - origin_t) @ frame_t).float()
        dc.rel_frames = torch.einsum('ji,njk->nik', frame_t, dc.nt_frames_world).float()
        dc.pair_rel_origins = ((dc.pair_origins_world - origin_t) @ frame_t).float()
        dc.pair_rel_frames = torch.einsum('ji,njk->nik', frame_t, dc.pair_frames_world).float()
        return dc

    def _inference_chain_end_mask(self, batch) -> torch.Tensor:
        b, ws = self._b_ws(batch)
        mask = torch.ones((b, ws, N_TORSIONS), dtype=torch.bool, device=batch.torsions.device)
        ce = batch.chain_end_class.view(b, ws, N_CHAIN_END_CLASSES)
        is_5prime = ce[..., CHAIN_END_CLASS_5_PRIME] > 0
        is_3prime = ce[..., CHAIN_END_CLASS_3_PRIME] > 0
        mask[..., TOR_ALPHA] &= ~is_5prime
        mask[..., TOR_EPS] &= ~is_3prime
        mask[..., TOR_ZETA] &= ~is_3prime
        return mask

    def _build_window_backbone(
        self,
        theta_w: torch.Tensor,
        tau_w: torch.Tensor,
        batch,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, ws = self._b_ws(batch)
        restype = batch.base_types.view(b, ws, len(BASE_TO_INDEX)).argmax(-1)
        origins_w = batch.nt_origins_world.view(b, ws, 3)
        frames_w = batch.nt_frames_world.view(b, ws, 3, 3)
        return build_batch_window_backbone_from_torsions(
            theta_w,
            tau_w,
            restype.long(),
            origins_w.float(),
            frames_w.float(),
            mask,
        )

    @torch.no_grad()
    def _predict_eval_window(self, batch):
        sample_data_list = batch.to_data_list()
        target_samples = []
        sample_keys: list[tuple[int, int]] = []
        for graph_idx, data in enumerate(sample_data_list):
            ws = int(data.nt_origins_world.size(0))
            for target_idx in range(ws):
                target_samples.append(self._prepare_target_view(data, target_idx))
                sample_keys.append((graph_idx, target_idx))

        if not target_samples:
            raise ValueError('Cannot evaluate an empty batch.')

        pred_batch = Batch.from_data_list(target_samples).to(self.device)
        pred_theta_all, pred_tau_all = self.sample(pred_batch)
        eval_batch = batch.to(pred_theta_all.device)
        theta_w = torch.stack([data.torsions.clone() for data in sample_data_list], dim=0).to(
            device=pred_theta_all.device,
            dtype=pred_theta_all.dtype,
        )
        tau_w = torch.stack([data.tau_m.clone() for data in sample_data_list], dim=0).to(
            device=pred_tau_all.device,
            dtype=pred_tau_all.dtype,
        )
        for sample_idx, (graph_idx, target_idx) in enumerate(sample_keys):
            theta_w[graph_idx, target_idx] = pred_theta_all[sample_idx]
            tau_w[graph_idx, target_idx] = pred_tau_all[sample_idx]
        tau_w = tau_w.clamp(min=TAU_M_MIN, max=TAU_M_MAX)
        mask = self._inference_chain_end_mask(eval_batch)
        coords_w = self._build_window_backbone(
            theta_w.float(),
            tau_w.float(),
            eval_batch,
            mask,
        )
        return eval_batch, theta_w, tau_w, coords_w

    def forward_score_network_from_prefix(self, prefix, batch, x_t_latent, log_sigma_per_graph, sc):
        b, _ = self._b_ws(batch)
        x = self._build_x_from_prefix(prefix, batch, x_t_latent, log_sigma_per_graph, sc)
        score_all = self.score_network(x)
        bi = torch.arange(b, device=score_all.device)
        return score_all[bi, batch.target_nt_idx.long()]

    def forward_score_network(self, batch, x_t_latent, log_sigma_per_graph, sc):
        prefix = self._build_static_x_prefix(batch)
        return self.forward_score_network_from_prefix(prefix, batch, x_t_latent, log_sigma_per_graph, sc)

    def _get_sampling_sigmas(
        self,
        device: torch.device,
        dtype: torch.dtype,
        num_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (
            device.type,
            device.index,
            dtype,
            num_steps,
            float(self.hparams['angular_sigma_max']),
            float(self.hparams['angular_sigma_min']),
            float(self.hparams['tau_sigma_max']),
            float(self.hparams['tau_sigma_min']),
        )
        if key not in self._sampling_sigma_cache:
            self._sampling_sigma_cache[key] = (
                ve_sigma_grid(
                    float(self.hparams['angular_sigma_max']),
                    float(self.hparams['angular_sigma_min']),
                    num_steps,
                    device=device,
                    dtype=dtype,
                ),
                ve_sigma_grid(
                    float(self.hparams['tau_sigma_max']),
                    float(self.hparams['tau_sigma_min']),
                    num_steps,
                    device=device,
                    dtype=dtype,
                ),
            )
        return self._sampling_sigma_cache[key]

    def _get_rmsd_keep_indices(self, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index)
        if key not in self._rmsd_keep_cache:
            self._rmsd_keep_cache[key] = torch.tensor(
                [j for j, nm in enumerate(BACKBONE_ATOMS) if nm not in ('OP1', 'OP2')],
                device=device,
                dtype=torch.long,
            )
        return self._rmsd_keep_cache[key]

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
        if torch.rand((), device=x_t.device) < 0.5:
            with torch.no_grad():
                pred_sc = self.forward_score_network(batch, x_t, log_sigma_cond, sc)
                sc = estimate_latent_from_ve_score(
                    pert['theta_t'],
                    log_tau_t,
                    pred_sc,
                    pert['sigma_theta'],
                    pert['sigma_tau'],
                ).detach()
                sc = torch.nan_to_num(sc)
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
        clo_metrics = self._bridge_closure_metrics(pred_x0_cl, batch)
        cl = clo_metrics['closure_loss']
        cw = float(self.hparams.get('closure_loss_weight', 0.0))
        loss = mse if cw == 0.0 else mse + cw * cl
        self.log(
            'train/loss', mse,
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'diagnostics/train/angular_score_loss', mse_theta,
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'diagnostics/train/tau_score_loss', mse_tau,
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'diagnostics/train/score_norm', pred.square().mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        tgt = torch.cat(
            [pert['angular_score_target'], pert['tau_score_target'].reshape(b, N_LATENT - N_TORSIONS)],
            dim=-1,
        )
        self.log(
            'diagnostics/train/target_score_norm', tgt.square().mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'diagnostics/train/sigma_theta_mean',
            pert['sigma_theta'].mean(),
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        self.log(
            'diagnostics/train/sigma_tau_mean',
            pert['sigma_tau'].mean(),
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
                f'diagnostics/train/{key}', clo_metrics[key],
                on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
            )
        return loss

    def on_train_epoch_end(self):
        self._write_epoch_scalars([
            'train/loss',
            'diagnostics/train/angular_score_loss',
            'diagnostics/train/tau_score_loss',
            'diagnostics/train/score_norm',
            'diagnostics/train/target_score_norm',
            'diagnostics/train/sigma_theta_mean',
            'diagnostics/train/sigma_tau_mean',
            'diagnostics/train/closure_loss',
            'diagnostics/train/closure_bond_loss',
            'diagnostics/train/closure_angle_loss',
            'diagnostics/train/closure_torsion_loss',
            'diagnostics/train/closure_valid_bridge_fraction',
            'diagnostics/train/closure_num_valid_bridges',
            'diagnostics/train/closure_fail_rate',
            'diagnostics/train/bridge_bond_mae',
            'diagnostics/train/bridge_angle_mae_deg',
            'diagnostics/train/bridge_torsion_mae_deg',
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
                'val/rmsd/avg',
                'val/rmsd/central',
                'val/rmsd/edge',
            ]
        else:
            keys = [
                'test/rmsd/avg',
                'test/rmsd/central',
                'test/rmsd/edge',
            ]
        self._write_epoch_scalars(keys)

    def _val_closure_tensorboard_keys(self):
        """TensorBoard scalars for bridge closure on validation samples."""
        return [
            'diagnostics/val/closure_loss',
            'diagnostics/val/closure_bond_loss',
            'diagnostics/val/closure_angle_loss',
            'diagnostics/val/closure_torsion_loss',
            'diagnostics/val/closure_valid_bridge_fraction',
            'diagnostics/val/closure_num_valid_bridges',
            'diagnostics/val/closure_fail_rate',
            'diagnostics/val/bridge_bond_mae',
            'diagnostics/val/bridge_angle_mae_deg',
            'diagnostics/val/bridge_torsion_mae_deg',
        ]

    def on_validation_epoch_end(self):
        self._write_rmsd_scalars('val')
        self._write_epoch_scalars(self._val_closure_tensorboard_keys())

    def on_test_epoch_end(self):
        self._write_rmsd_scalars('test')

    @torch.no_grad()
    def p_sample_loop(self, batch, num_timesteps: int | None = None):
        theta0, _, _, _, _ = self._theta_mask_target(batch)
        b = batch.num_graphs
        dev = self.device
        dtype = torch.float32
        if num_timesteps is None:
            num_timesteps = int(self.hparams['num_timesteps'])
        step_fn = reverse_ve_score_ode_step
        device = torch.device(dev)
        sig_theta, sig_tau = self._get_sampling_sigmas(device, dtype, num_timesteps)
        prefix = self._build_static_x_prefix(batch)
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
            pred = self.forward_score_network_from_prefix(prefix, batch, x_t, log_s, sc)
            sc = estimate_latent_from_ve_score(
                x_t[..., :N_TORSIONS],
                x_t[..., N_TORSIONS:N_LATENT],
                pred,
                sigma_cur_th,
                sigma_cur_tau,
            ).detach()
            sc = torch.nan_to_num(sc)
            theta, logt = step_fn(
                x_t[..., :N_TORSIONS],
                x_t[..., N_TORSIONS:N_LATENT],
                pred,
                sigma_cur_th,
                sigma_next_th,
                sigma_cur_tau,
                sigma_next_tau,
            )
            x_t = torch.cat([theta, logt], dim=-1)
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
        coords_w: torch.Tensor | None = None,
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

        if coords_w is None:
            theta_w = batch.torsions.view(b, ws, N_TORSIONS).clone()
            tau_w = batch.tau_m.view(b, ws).clone()
            theta_w[bi, ti] = pred_torsions.float()
            tau_w[bi, ti] = pred_tau_m.clamp(min=TAU_M_MIN, max=TAU_M_MAX).float()
            coords_w = self._build_window_backbone(
                theta_w,
                tau_w,
                batch,
                batch.torsion_mask.view(b, ws, N_TORSIONS),
            )
        pred_bb_world = coords_w[bi, ti]
        pred_local = (pred_bb_world - origins.unsqueeze(1)) @ frames

        j1 = BACKBONE_ATOMS.index('OP1')
        j2 = BACKBONE_ATOMS.index('OP2')
        contrib = torch.zeros(b, device=dev, dtype=torch.float64)
        count = torch.zeros(b, device=dev, dtype=torch.float64)
        keep = self._get_rmsd_keep_indices(torch.device(dev))
        pred_keep = pred_local.index_select(1, keep).to(dtype=torch.float64)
        gt_keep = gt_local.index_select(1, keep).to(dtype=torch.float64)
        valid_keep = torch.isfinite(pred_keep).all(dim=-1) & torch.isfinite(gt_keep).all(dim=-1)
        sq_keep = ((pred_keep - gt_keep) ** 2).sum(dim=-1)
        sq_keep = torch.where(valid_keep, sq_keep, torch.zeros_like(sq_keep))
        contrib = contrib + sq_keep.sum(dim=-1)
        count = count + valid_keep.to(dtype=torch.float64).sum(dim=-1)

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
        op_sq = torch.where(v_op, op_sq, torch.zeros_like(op_sq))
        contrib = contrib + op_sq
        count = count + v_op.to(dtype=torch.float64)

        out = torch.full((b,), float('nan'), device=dev, dtype=torch.float64)
        ok = count > 0
        out = torch.where(ok, torch.sqrt(contrib / count), out)
        return out.to(pred_torsions.dtype)

    def _bridge_closure_metrics_full_window(
        self,
        theta_w: torch.Tensor,
        tau_w: torch.Tensor,
        batch,
        bb_world: torch.Tensor | None = None,
    ) -> dict:
        b, ws = self._b_ws(batch)
        ti = batch.target_nt_idx.long()
        bi = torch.arange(b, device=tau_w.device)
        if bb_world is None:
            bb_world = self._build_window_backbone(
                theta_w.float(),
                tau_w.float(),
                batch,
                self._inference_chain_end_mask(batch),
            )
        pair_mask = torch.zeros(b, ws - 1, dtype=torch.bool, device=tau_w.device)
        mk_l = ti > 0
        pair_mask[bi[mk_l], ti[mk_l] - 1] = True
        mk_r = ti < ws - 1
        pair_mask[bi[mk_r], ti[mk_r]] = True
        restype = batch.base_types.view(b, ws, len(BASE_TO_INDEX)).argmax(-1)
        weights = {
            'bond': float(self.hparams.get('closure_bond_weight', 1.0)),
            'angle': float(self.hparams.get('closure_angle_weight', 1.0)),
            'torsion': float(self.hparams.get('closure_torsion_weight', 1.0)),
        }
        geometry = {
            'sigma_bond': float(self.hparams.get('closure_sigma_bond_a', CLOSURE_SIGMA_BOND_A)),
            'sigma_angle_rad': float(self.hparams.get('closure_sigma_angle_rad', CLOSURE_SIGMA_ANGLE_RAD)),
            'sigma_torsion_rad': float(self.hparams.get('closure_sigma_torsion_rad', CLOSURE_SIGMA_TORSION_RAD)),
        }
        return compute_bridge_closure_loss(
            bb_world,
            batch.torsions.view(b, ws, N_TORSIONS),
            self._inference_chain_end_mask(batch),
            restype.long(),
            same_chain_mask=None,
            valid_pair_mask=pair_mask,
            geometry=geometry,
            weights=weights,
            grad_prop_tensor=theta_w[bi, ti],
        )

    def _log_rmsd(
        self,
        prefix: str,
        pred_theta: torch.Tensor,
        pred_tau_m: torch.Tensor,
        batch,
        coords_w: torch.Tensor | None = None,
    ):
        """TensorBoard RMSD for the target nucleotide in each window."""
        per_graph_rmsd = self._compute_rmsd_per_graph_local(
            pred_theta,
            pred_tau_m,
            batch,
            coords_w=coords_w,
        )
        is_edge = self._is_edge_target(batch)
        finite = torch.isfinite(per_graph_rmsd)

        for name, mask in [
            (f'{prefix}/rmsd/avg', finite),
            (f'{prefix}/rmsd/central', (~is_edge) & finite),
            (f'{prefix}/rmsd/edge', is_edge & finite),
        ]:
            if mask.any():
                self.log(
                    name, per_graph_rmsd[mask].mean(),
                    on_epoch=True, on_step=False, sync_dist=True,
                    batch_size=max(int(mask.sum().item()), 1), logger=False,
                )

    def validation_step(self, batch, batch_idx):
        eval_batch, theta_w, tau_w, coords_w = self._predict_eval_window(batch)
        bi = torch.arange(eval_batch.num_graphs, device=coords_w.device)
        ti = eval_batch.target_nt_idx.long()
        pred_theta = theta_w[bi, ti]
        pred_tau_m = tau_w[bi, ti]
        self._log_rmsd('val', pred_theta, pred_tau_m, eval_batch, coords_w=coords_w)
        clo = self._bridge_closure_metrics_full_window(theta_w, tau_w, eval_batch, bb_world=coords_w)
        for key, val in clo.items():
            self.log(
                f'diagnostics/val/{key}',
                val,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
                batch_size=batch.num_graphs,
                logger=False,
            )

    def test_step(self, batch, _):
        eval_batch, theta_w, tau_w, coords_w = self._predict_eval_window(batch)
        bi = torch.arange(eval_batch.num_graphs, device=coords_w.device)
        ti = eval_batch.target_nt_idx.long()
        self._log_rmsd(
            'test',
            theta_w[bi, ti],
            tau_w[bi, ti],
            eval_batch,
            coords_w=coords_w,
        )

    @torch.no_grad()
    def sample(self, batch, num_timesteps: int | None = None):
        _, (pred_theta, pred_tau_m) = self.p_sample_loop(
            batch,
            num_timesteps=num_timesteps,
        )
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
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/rmsd/avg'},
        }
