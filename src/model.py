import math
import os
import os.path as osp
import sys
import time

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bridge_closure import compute_bridge_closure_loss
from torsion_geometry import (
    N_TORSIONS,
    build_batch_window_backbone_from_torsions_torch,
)
from utils import N_CHAIN_END_CLASSES, backbone_atoms, base_to_idx

# Sin/cos latent per torsion angle plus log τ_m (must match denoiser output width).
N_TORSIONS_LATENT = N_TORSIONS * 2 + 1


def _fitdbg(msg: str) -> None:
    """Stdout probe for hangs under DDP / heavy steps. Off: FITDBG=0 (or false/off/no).

    Extra GPU sync before each line: FITDBG_SYNC_CUDA=1.
    """
    flag = os.environ.get('FITDBG', '1').strip().lower()
    if flag in ('0', 'false', 'no', 'off'):
        return
    if (
        os.environ.get('FITDBG_SYNC_CUDA', '').lower() in ('1', 'true', 'yes')
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()
    lr = os.environ.get('LOCAL_RANK', '?')
    rk = os.environ.get('RANK', '?')
    sys.stderr.write(
        f'[fitdbg t={time.monotonic():.3f}s pid={os.getpid()} '
        f'RANK={rk} LOCAL_RANK={lr}] {msg}\n',
    )
    sys.stderr.flush()


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


def get_beta_schedule(beta_schedule, num_timesteps):
    def linear_beta_schedule(num_timesteps):
        scale = 1000 / num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

    def cosine_beta_schedule(num_timesteps, s=0.008):
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    if beta_schedule == 'linear':
        return linear_beta_schedule(num_timesteps)
    if beta_schedule == 'cosine':
        return cosine_beta_schedule(num_timesteps)
    raise NotImplementedError(f'unknown beta schedule: {beta_schedule}')


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class TorsionDenoiser(nn.Module):
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
        self.out = nn.Linear(hidden_dim, N_TORSIONS_LATENT)

    def forward(self, x):
        h = self.in_mlp(x)
        h = self.tr(h)
        return self.out(h)


class PytorchLightningModule(pl.LightningModule):
    def __init__(
        self, hidden_dim, num_heads, num_layers, num_timesteps, batch_size, lr,
        lr_scheduler, lr_scheduler_patience, lr_scheduler_threshold, lr_scheduler_cooldown,
        beta_schedule, weight_decay,
        closure_loss_weight: float = 0.0,
        closure_bond_weight: float = 1.0,
        closure_angle_weight: float = 1.0,
        closure_torsion_weight: float = 1.0,
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
            + len(base_to_idx)
            + 1
            + N_CHAIN_END_CLASSES
            + 1
            + N_TORSIONS_LATENT
            + self.time_emb_dim
            + N_TORSIONS
            + N_TORSIONS_LATENT
        )

        self.denoiser = TorsionDenoiser(
            node_dim=self.node_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        betas = get_beta_schedule(beta_schedule, num_timesteps=num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def load_state_dict(self, state_dict, strict=True):  # type: ignore[no-untyped-def]
        _out_w = next(
            (v for k, v in state_dict.items() if str(k).endswith('denoiser.out.weight')),
            None,
        )
        if _out_w is not None and _out_w.shape[0] != N_TORSIONS_LATENT:
            raise RuntimeError(
                f'Checkpoint denoiser output dim is {_out_w.shape[0]}, '
                f'expected {N_TORSIONS_LATENT} (interleaved sin/cos torsions + log τ_m). '
                'Use a checkpoint trained with matching latent layout.',
            )
        # Checkpoints saved with torch.compile(denoiser) use denoiser._orig_mod.*; eager module expects denoiser.*.
        _orig = 'denoiser._orig_mod.'
        if any(str(k).startswith(_orig) for k in state_dict):
            state_dict = {
                ('denoiser.' + str(k)[len(_orig):] if str(k).startswith(_orig) else k): v
                for k, v in state_dict.items()
            }
        return super().load_state_dict(state_dict, strict=strict)

    @staticmethod
    def encode_torsions(theta: torch.Tensor, tau_m: torch.Tensor) -> torch.Tensor:
        """[..., N_TORSIONS], [...] → [..., N_TORSIONS_LATENT]: interleaved sin/cos + log τ_m."""
        sc = torch.stack([theta.sin(), theta.cos()], dim=-1).flatten(-2)
        log_tau = torch.log(tau_m.clamp(min=1e-3)).unsqueeze(-1)
        return torch.cat([sc, log_tau], dim=-1)

    @staticmethod
    def decode_torsions(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[..., N_TORSIONS_LATENT] → (theta [..., N_TORSIONS], τ_m [...])."""
        theta = torch.atan2(
            latent[..., 0::2][..., :N_TORSIONS],
            latent[..., 1::2][..., :N_TORSIONS],
        )
        tau_m = torch.exp(latent[..., -1].clamp(max=2.0))
        return theta, tau_m

    def _expand_latent_mask(self, torsion_mask: torch.Tensor, tau_m_mask: torch.Tensor) -> torch.Tensor:
        """[b, N_TORSIONS], [b] → [b, N_TORSIONS_LATENT]."""
        b = torsion_mask.shape[0]
        pair = torsion_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(b, N_TORSIONS * 2)
        return torch.cat([pair, tau_m_mask.unsqueeze(-1)], dim=-1)

    def on_test_start(self):
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        logger = self.trainer.logger
        dm = self.trainer.datamodule  # type: ignore[attr-defined]
        log_dir = getattr(logger, 'log_dir', None)
        if log_dir is None or dm is None:
            return
        torch.save(dm.test_dataset, osp.join(log_dir, 'test_dataset.pt'))

    def q_sample(self, theta_start: torch.Tensor, tau_m_start: torch.Tensor, t, noise=None):
        x_start = self.encode_torsions(theta_start, tau_m_start)
        if noise is None:
            noise = torch.randn_like(x_start)
        sa = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        s1 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sa * x_start + s1 * noise, noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _decode_epsilon_to_x0(self, x_t, pred_noise, t):
        sa = extract(self.sqrt_alphas_cumprod, t, x_t.shape).clamp(min=1e-8)
        s1 = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape).clamp(min=1e-8)
        return (x_t - s1 * pred_noise) / sa

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

        pred_theta, pred_tau_m = self.decode_torsions(pred_x0)

        tors_m = batch.torsions.view(b, ws, N_TORSIONS).clone()
        tau_m = batch.tau_m.view(b, ws).clone()
        tors_m[bi, ti] = pred_theta
        tau_m[bi, ti] = pred_tau_m.clamp(min=1e-3)

        restype = batch.base_types.view(b, ws, len(base_to_idx)).argmax(-1)
        mask = batch.torsion_mask.view(b, ws, N_TORSIONS)
        origins = batch.nt_origins_world.view(b, ws, 3)
        frames = batch.nt_frames_world.view(b, ws, 3, 3)

        _fitdbg(
            f'_bridge_closure_metrics: build_batch_window_backbone B={b} W={ws}',
        )
        t_bb = time.perf_counter()
        bb = build_batch_window_backbone_from_torsions_torch(
            tors_m.float(),
            tau_m.float(),
            restype.long(),
            origins.float(),
            frames.float(),
            mask,
        )
        _fitdbg(
            f'_bridge_closure_metrics: backbone done in {time.perf_counter() - t_bb:.3f}s',
        )

        n_bb = len(backbone_atoms)
        bb_gt = batch.bb_xyz_world.view(b, ws, n_bb, 3)
        j_c4 = backbone_atoms.index("C4'")
        j_c3 = backbone_atoms.index("C3'")
        j_o3 = backbone_atoms.index("O3'")
        j_p = backbone_atoms.index('P')
        j_o5 = backbone_atoms.index("O5'")
        j_c5 = backbone_atoms.index("C5'")
        fin_prev = (
            torch.isfinite(bb_gt[..., j_c4]).all(dim=-1)
            & torch.isfinite(bb_gt[..., j_c3]).all(dim=-1)
            & torch.isfinite(bb_gt[..., j_o3]).all(dim=-1)
        )
        fin_next = (
            torch.isfinite(bb_gt[..., j_p]).all(dim=-1)
            & torch.isfinite(bb_gt[..., j_o5]).all(dim=-1)
            & torch.isfinite(bb_gt[..., j_c5]).all(dim=-1)
            & torch.isfinite(bb_gt[..., j_c4]).all(dim=-1)
        )
        is_edge = batch.is_chain_edge_nt.view(b, ws)
        valid_nt_mask = (~is_edge) & fin_prev & fin_next

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
            valid_nt_mask,
            restype.long(),
            same_chain_mask,
            weights=weights,
            grad_prop_tensor=pred_theta,
        )

    def _closure_loss(self, pred_x0: torch.Tensor, batch) -> torch.Tensor:
        return self._bridge_closure_metrics(pred_x0, batch)['closure_loss']

    def _build_x(self, batch, x_t_latent, t_per_graph, sc):
        b, ws = self._b_ws(batch)
        rel_o = batch.rel_origins.view(b, ws, 3)
        rel_R = batch.rel_frames.view(b, ws, 9)
        pair_o = batch.pair_rel_origins.view(b, ws, 3)
        pair_R = batch.pair_rel_frames.view(b, ws, 3, 3).reshape(b, ws, 9)
        base = batch.base_types.view(b, ws, len(base_to_idx))
        hp = batch.has_pair_nt.view(b, ws, 1).float()
        ce = batch.chain_end_class.view(b, ws, N_CHAIN_END_CLASSES)
        it = batch.is_target_nt.view(b, ws, 1)
        tidx = batch.target_nt_idx.long()
        bi = torch.arange(b, device=rel_o.device)
        pad = torch.zeros(
            b, ws,
            N_TORSIONS_LATENT + self.time_emb_dim + N_TORSIONS + N_TORSIONS_LATENT,
            device=rel_o.device, dtype=rel_o.dtype,
        )
        te_all = self.time_mlp(t_per_graph.float())
        o = 0
        pad[bi, tidx, o:o + N_TORSIONS_LATENT] = x_t_latent
        o += N_TORSIONS_LATENT
        pad[bi, tidx, o:o + self.time_emb_dim] = te_all
        o += self.time_emb_dim
        pad[bi, tidx, o:o + N_TORSIONS] = (
            batch.torsion_mask.view(b, ws, N_TORSIONS)[bi, tidx].float()
        )
        o += N_TORSIONS
        pad[bi, tidx, o:o + N_TORSIONS_LATENT] = sc
        return torch.cat([rel_o, rel_R, pair_o, pair_R, base, hp, ce, it, pad], dim=-1)

    def on_fit_start(self):
        ws = int(self.trainer.world_size) if self.trainer else -1
        _fitdbg(
            f'on_fit_start world_size={ws} '
            f'num_nodes={getattr(self.trainer, "num_nodes", "?")}',
        )

    def on_train_epoch_start(self):
        ep = int(getattr(self, 'current_epoch', -1))
        _fitdbg(f'on_train_epoch_start epoch={ep}')

    def on_validation_epoch_start(self):
        ep = int(getattr(self, 'current_epoch', -1))
        _fitdbg(
            f'on_validation_epoch_start epoch={ep} '
            f'(sampling+RMSD может занять много времени — не путать с зависанием)',
        )

    def forward_denoiser(self, batch, x_t_latent, t_per_graph, sc):
        b, _ = self._b_ws(batch)
        x = self._build_x(batch, x_t_latent, t_per_graph, sc)
        eps_all = self.denoiser(x)
        bi = torch.arange(b, device=eps_all.device)
        return eps_all[bi, batch.target_nt_idx.long()]

    def training_step(self, batch, batch_idx):
        theta0, m, tau0, tau_mk, _ = self._theta_mask_target(batch)
        b = batch.num_graphs
        _, ws = self._b_ws(batch)
        _fitdbg(
            f'train_step batch_idx={batch_idx} num_graphs={b} ws={ws} '
            f'device={batch.torsions.device}',
        )
        tau_safe = torch.where(
            tau_mk,
            tau0.clamp(min=1e-3),
            torch.full_like(tau0, 0.611),
        )
        t = torch.randint(0, self.hparams['num_timesteps'], (b,), device=self.device).long()
        x_t, noise = self.q_sample(theta0, tau_safe, t)
        mask_latent = self._expand_latent_mask(m, tau_mk)

        sc = torch.zeros_like(x_t)
        do_self_cond = torch.rand(1).item() < 0.5
        if do_self_cond:
            _fitdbg(f'train_step batch_idx={batch_idx}: self-conditioning forward (no_grad)')
            with torch.no_grad():
                pred_first = self.forward_denoiser(batch, x_t, t, sc)
                sc = self._decode_epsilon_to_x0(x_t, pred_first, t).detach()
            _fitdbg(f'train_step batch_idx={batch_idx}: self-conditioning done')

        _fitdbg(f'train_step batch_idx={batch_idx}: forward_denoiser')
        pred = self.forward_denoiser(batch, x_t, t, sc)
        _fitdbg(f'train_step batch_idx={batch_idx}: forward_denoiser OK')
        mse = (
            (pred - noise) ** 2 * mask_latent.float()
        ).sum() / mask_latent.float().sum().clamp(min=1.0)
        pred_x0_cl = self._decode_epsilon_to_x0(x_t, pred, t)
        _fitdbg(f'train_step batch_idx={batch_idx}: MSE ok, start closure_loss')
        clo_metrics = self._bridge_closure_metrics(pred_x0_cl, batch)
        cl = clo_metrics['closure_loss']
        loss = mse + float(self.hparams.get('closure_loss_weight', 0.0)) * cl
        _fitdbg(f'train_step batch_idx={batch_idx}: closure_loss OK')
        self.log(
            'train_loss', mse,
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
            'closure_fail_rate',
            'bridge_bond_mae',
            'bridge_angle_mae_deg',
            'bridge_torsion_mae_deg',
        ):
            self.log(
                key, clo_metrics[key],
                on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
            )
        _fitdbg(
            f'train_step batch_idx={batch_idx}: logged metrics; return loss={float(loss.detach())}',
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        _fitdbg(
            f'on_train_batch_end batch_idx={batch_idx} '
            f'(шаг Lightning после backward; см. следующий batch или val)',
        )

    def on_train_epoch_end(self):
        self._write_epoch_scalars([
            'train_loss',
            'train_closure',
            'closure_loss',
            'closure_bond_loss',
            'closure_angle_loss',
            'closure_torsion_loss',
            'closure_valid_bridge_fraction',
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
        from lightning.pytorch.loggers import TensorBoardLogger
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
        self._write_epoch_scalars([f'{prefix}_{k}' for k in ('rmsd', 'rmsd_central', 'rmsd_edge')])

    def on_validation_epoch_end(self):
        self._write_rmsd_scalars('val')

    def on_test_epoch_end(self):
        self._write_rmsd_scalars('test')

    @torch.no_grad()
    def p_sample_loop(self, batch):
        theta0, _, _, _, _ = self._theta_mask_target(batch)
        b = batch.num_graphs
        x_t = torch.randn(b, N_TORSIONS_LATENT, device=self.device)
        sc = torch.zeros_like(x_t)
        for step in reversed(range(self.hparams['num_timesteps'])):
            t = torch.full((b,), step, device=x_t.device, dtype=torch.long)
            x_t, sc = self.p_sample(x_t, t, batch, sc)
            if not torch.isfinite(x_t).all():
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
            if not torch.isfinite(sc).all():
                sc = torch.nan_to_num(sc, nan=0.0, posinf=1.0, neginf=-1.0)
        pred_theta, pred_tau_m = self.decode_torsions(x_t)
        return theta0, (pred_theta, pred_tau_m)

    @torch.no_grad()
    def p_sample(self, x_t, t, batch, sc):
        pred_noise = self.forward_denoiser(batch, x_t, t, sc)
        pred_x0 = self._decode_epsilon_to_x0(x_t, pred_noise, t)
        model_mean, _, model_log_var = self.q_posterior_mean_variance(pred_x0, x_t, t)
        noise = torch.randn_like(x_t) * (t > 0).float().unsqueeze(-1)
        x_prev = model_mean + (0.5 * model_log_var).exp() * noise
        return x_prev, pred_x0

    def _compute_rmsd_per_graph_local(
        self,
        pred_torsions: torch.Tensor,
        pred_tau_m: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        from torsion_geometry import build_backbone_from_torsions_torch

        b, ws = self._b_ws(batch)
        ti = batch.target_nt_idx.long()
        bi = torch.arange(b, device=ti.device)
        dev = pred_torsions.device

        n_bb = len(backbone_atoms)
        bb_world_all = batch.bb_xyz_world.view(b, ws, n_bb, 3)
        gt_bb_world = bb_world_all[bi, ti]
        origins = batch.nt_origins_world.view(b, ws, 3)[bi, ti]
        frames = batch.nt_frames_world.view(b, ws, 3, 3)[bi, ti]
        gt_local = (gt_bb_world - origins.unsqueeze(1)) @ frames

        o3_prev = batch.o3_prev_local.view(b, ws, 3)[bi, ti]
        o3_valid = batch.o3_prev_valid.view(b, ws)[bi, ti]

        base_onehot = batch.base_types.view(b, ws, len(base_to_idx))[bi, ti]
        restype_idx = base_onehot.argmax(dim=-1)

        o3_prev_input = torch.where(
            o3_valid.unsqueeze(-1), o3_prev, torch.zeros_like(o3_prev),
        )

        with torch.no_grad():
            pred_bb = build_backbone_from_torsions_torch(
                pred_torsions.float(),
                pred_tau_m.clamp(min=1e-3).float(),
                restype_idx,
                o3_prev_local=o3_prev_input,
            )

        j1 = backbone_atoms.index('OP1')
        j2 = backbone_atoms.index('OP2')
        contrib = torch.zeros(b, device=dev, dtype=torch.float64)
        count = torch.zeros(b, device=dev, dtype=torch.float64)

        for j, nm in enumerate(backbone_atoms):
            if nm in ('OP1', 'OP2'):
                continue
            if nm not in pred_bb:
                continue
            pred_xyz = pred_bb[nm].to(dtype=torch.float64)
            gt_xyz = gt_local[:, j, :].to(dtype=torch.float64)
            valid = torch.isfinite(pred_xyz).all(dim=-1) & torch.isfinite(gt_xyz).all(
                dim=-1,
            )
            diff = pred_xyz - gt_xyz
            sq = torch.einsum('bi,bi->b', diff, diff)
            contrib = contrib + sq * valid.to(dtype=torch.float64)
            count = count + valid.to(dtype=torch.float64)

        if 'OP1' in pred_bb and 'OP2' in pred_bb:
            p1 = pred_bb['OP1'].to(dtype=torch.float64)
            p2 = pred_bb['OP2'].to(dtype=torch.float64)
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
        """Accumulate per-step RMSD (all / central / edge) for later TensorBoard write."""
        _fitdbg(f'_log_rmsd `{prefix}`: start CPU geometry over B={pred_theta.shape[0]}')
        t0 = time.perf_counter()
        per_graph_rmsd = self._compute_rmsd_per_graph_local(
            pred_theta, pred_tau_m, batch,
        )
        _fitdbg(f'_log_rmsd `{prefix}`: RMSD compute {time.perf_counter() - t0:.3f}s')
        is_edge = self._is_edge_target(batch)
        finite = torch.isfinite(per_graph_rmsd)

        for name, mask in [
            (f'{prefix}_rmsd',         finite),
            (f'{prefix}_rmsd_central', (~is_edge) & finite),
            (f'{prefix}_rmsd_edge',    is_edge & finite),
        ]:
            if mask.any():
                self.log(
                    name, per_graph_rmsd[mask].mean(),
                    on_epoch=True, on_step=False, sync_dist=True,
                    batch_size=max(int(mask.sum().item()), 1), logger=False,
                )

    def validation_step(self, batch, batch_idx):
        b = batch.num_graphs
        nt = int(self.hparams['num_timesteps'])
        _fitdbg(
            f'validation_step batch_idx={batch_idx} num_graphs={b} '
            f'sampling_steps={nt} (тяжёлый проход)',
        )
        _fitdbg(f'validation_step batch_idx={batch_idx}: p_sample_loop …')
        _, (pred_theta, pred_tau_m) = self.p_sample_loop(batch)
        _fitdbg(f'validation_step batch_idx={batch_idx}: p_sample_loop OK → RMSD')
        self._log_rmsd('val', pred_theta, pred_tau_m, batch)
        _fitdbg(f'validation_step batch_idx={batch_idx}: готово')

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
