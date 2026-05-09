import math
import os.path as osp

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torsion_geometry import N_TORSIONS, build_backbone_from_torsions
from utils import N_CHAIN_END_CLASSES, backbone_atoms, base_to_idx

N_TORSIONS_LATENT = N_TORSIONS * 2 + 1  # 17: 16 sin/cos + log(τ_m)


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

    @staticmethod
    def encode_torsions(theta: torch.Tensor, tau_m: torch.Tensor) -> torch.Tensor:
        """[..., N_TORSIONS], [...] → [..., 17]: interleaved sin/cos + log τ_m."""
        sc = torch.stack([theta.sin(), theta.cos()], dim=-1).flatten(-2)
        log_tau = torch.log(tau_m.clamp(min=1e-3)).unsqueeze(-1)
        return torch.cat([sc, log_tau], dim=-1)

    @staticmethod
    def decode_torsions(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[..., 17] → (theta [..., N_TORSIONS], τ_m [...])."""
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

    def forward_denoiser(self, batch, x_t_latent, t_per_graph, sc):
        b, _ = self._b_ws(batch)
        x = self._build_x(batch, x_t_latent, t_per_graph, sc)
        eps_all = self.denoiser(x)
        bi = torch.arange(b, device=eps_all.device)
        return eps_all[bi, batch.target_nt_idx.long()]

    def training_step(self, batch, _):
        theta0, m, tau0, tau_mk, _ = self._theta_mask_target(batch)
        b = batch.num_graphs
        tau_safe = torch.where(
            tau_mk,
            tau0.clamp(min=1e-3),
            torch.full_like(tau0, 0.611),
        )
        t = torch.randint(0, self.hparams['num_timesteps'], (b,), device=self.device).long()
        x_t, noise = self.q_sample(theta0, tau_safe, t)
        mask_latent = self._expand_latent_mask(m, tau_mk)

        sc = torch.zeros_like(x_t)
        if torch.rand(1).item() < 0.5:
            with torch.no_grad():
                pred_first = self.forward_denoiser(batch, x_t, t, sc)
                sc = self._decode_epsilon_to_x0(x_t, pred_first, t).detach()

        pred = self.forward_denoiser(batch, x_t, t, sc)
        mse = (
            (pred - noise) ** 2 * mask_latent.float()
        ).sum() / mask_latent.float().sum().clamp(min=1.0)
        self.log(
            'train_loss', mse,
            on_step=False, on_epoch=True, sync_dist=True, batch_size=b, logger=False,
        )
        return mse

    def on_train_epoch_end(self):
        self._write_epoch_scalars(['train_loss'])

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
    ) -> np.ndarray:
        b, ws = self._b_ws(batch)
        ti = batch.target_nt_idx.long()               # [b]
        bi = torch.arange(b, device=ti.device)

        n_bb = len(backbone_atoms)
        bb_world_all = batch.bb_xyz_world.view(b, ws, n_bb, 3)            # [b, ws, n_bb, 3]
        gt_bb_world = bb_world_all[bi, ti]                                 # [b, n_bb, 3]
        origins = batch.nt_origins_world.view(b, ws, 3)[bi, ti]           # [b, 3]
        frames = batch.nt_frames_world.view(b, ws, 3, 3)[bi, ti]         # [b, 3, 3]

        # Convert GT world → local frame: local = (world - origin) @ R
        gt_local = (gt_bb_world - origins.unsqueeze(1)) @ frames          # [b, n_bb, 3]

        j_O3 = backbone_atoms.index("O3'")
        ti_np = ti.cpu().numpy()
        origins_np = origins.cpu().numpy()           # [b, 3]
        frames_np = frames.cpu().numpy()             # [b, 3, 3]
        o3_prev_locals: list = []
        for i in range(b):
            if ti_np[i] > 0:
                o3w = bb_world_all[i, ti_np[i] - 1, j_O3].cpu().numpy()  # (3,)
                if not np.isnan(o3w).any():
                    o3_prev_locals.append((o3w - origins_np[i]) @ frames_np[i])
                else:
                    o3_prev_locals.append(None)
            else:
                o3_prev_locals.append(None)

        base_onehot = batch.base_types.view(b, ws, len(base_to_idx))[bi, ti]  # [b, 4]
        base_idx_np = base_onehot.argmax(dim=-1).cpu().numpy()            # [b]
        idx_to_base = {v: k for k, v in base_to_idx.items()}

        gt_local_np = gt_local.cpu().numpy()
        per_graph_rmsd = np.full((b,), np.nan, dtype=np.float64)
        j1 = backbone_atoms.index('OP1')
        j2 = backbone_atoms.index('OP2')
        tau_np = pred_tau_m.detach().cpu().numpy()
        for i in range(b):
            restype = idx_to_base[int(base_idx_np[i])]
            tors_np = pred_torsions[i].float().cpu().numpy()
            tm = float(np.clip(tau_np[i], 1e-3, None))
            pred_local = build_backbone_from_torsions(
                tors_np, restype, o3_prev_local=o3_prev_locals[i], tau_m=tm,
            )

            sq: list = []
            for j, nm in enumerate(backbone_atoms):
                if nm in ('OP1', 'OP2'):
                    continue
                pred_xyz = pred_local.get(nm)
                if pred_xyz is None or np.isnan(pred_xyz).any():
                    continue
                gt_xyz = gt_local_np[i, j]
                if np.isnan(gt_xyz).any():
                    continue
                sq.append(np.sum((pred_xyz - gt_xyz) ** 2))

            gt1, gt2 = gt_local_np[i, j1], gt_local_np[i, j2]
            p1 = pred_local.get('OP1')
            p2 = pred_local.get('OP2')
            if p1 is not None and p2 is not None:
                if not (
                    np.isnan(p1).any() or np.isnan(p2).any()
                    or np.isnan(gt1).any() or np.isnan(gt2).any()
                ):
                    d_straight = np.sum((p1 - gt1) ** 2) + np.sum((p2 - gt2) ** 2)
                    d_swapped = np.sum((p1 - gt2) ** 2) + np.sum((p2 - gt1) ** 2)
                    sq.append(min(d_straight, d_swapped) / 2)

            if sq:
                per_graph_rmsd[i] = float(np.sqrt(np.mean(sq)))

        return per_graph_rmsd

    def _log_rmsd(
        self, prefix: str, pred_theta: torch.Tensor, pred_tau_m: torch.Tensor, batch,
    ):
        """Accumulate per-step RMSD (all / central / edge) for later TensorBoard write."""
        per_graph_rmsd_np = self._compute_rmsd_per_graph_local(
            pred_theta, pred_tau_m, batch,
        )
        per_graph_rmsd = torch.from_numpy(per_graph_rmsd_np).to(
            device=pred_theta.device, dtype=torch.float32
        )
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

    def validation_step(self, batch, _):
        _, (pred_theta, pred_tau_m) = self.p_sample_loop(batch)
        self._log_rmsd('val', pred_theta, pred_tau_m, batch)

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
        if self.hparams['lr_scheduler'] != 'ReduceLROnPlateau':
            raise NotImplementedError(f'unknown lr scheduler: {self.hparams['lr_scheduler']}')
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
