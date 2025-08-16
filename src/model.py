import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import atom_to_idx


class EquivariantConv(nn.Module):
    def __init__(self, hidden_nf, act_fn=nn.SiLU()):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + 1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        )
        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )

    def forward(self, h, x, edge_index, edge_attr=None):
        # h: [N, hidden_nf], x: [N, 3]
        row, col = edge_index

        # Message passing
        rel_coords = x[row] - x[col]
        dist = torch.norm(rel_coords, p=2, dim=-1, keepdim=True)
        message_input = torch.cat([h[row], h[col], dist], dim=-1)
        messages = self.message_mlp(message_input)

        # Aggregate messages
        agg_messages = torch.zeros_like(h)
        index = col.unsqueeze(1).expand_as(messages)
        agg_messages.scatter_add_(0, index, messages)

        # Update coordinates
        # Normalize relative coordinates to unit vectors
        rel_dir = rel_coords / (dist + 1e-8)
        coord_mult = self.coord_mlp(messages)
        coord_update_src = coord_mult * rel_dir
        coord_update = torch.zeros_like(x)
        index = col.unsqueeze(1).expand_as(coord_update_src)
        coord_update.scatter_add_(0, index, coord_update_src)
        x_new = x + coord_update

        # Update features
        node_update_input = torch.cat([h, agg_messages], dim=-1)
        h_new = self.node_update_mlp(node_update_input)

        return h_new, x_new


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def linear_beta_schedule(timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    if beta_schedule == 'linear':
        return linear_beta_schedule(num_diffusion_timesteps)
    elif beta_schedule == 'cosine':
        return cosine_beta_schedule(num_diffusion_timesteps)
    else:
        raise NotImplementedError(f'unknown beta schedule: {beta_schedule}')


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class EGNNDiff(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers, in_edge_nf=0, act_fn=nn.SiLU(), normalize=False, attention=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.embedding_out = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            act_fn,
            nn.Linear(self.hidden_nf, out_node_nf)
        )

        for i in range(0, n_layers):
            self.add_module(f'gcl_{i}', EquivariantConv(self.hidden_nf, act_fn=act_fn))

    def forward(self, h, x, edge_index, edge_attr):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x = self._modules[f'gcl_{i}'](h, x, edge_index, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x


class PytorchLightningModule(pl.LightningModule):
    def __init__(self, hidden_dim, num_timesteps, batch_size, patience):
        super().__init__()
        self.save_hyperparameters()

        self.gnn = EGNNDiff(
            in_node_nf=len(atom_to_idx) + 1,  # +1 for time embedding
            hidden_nf=hidden_dim,
            out_node_nf=len(atom_to_idx) + 3,  # atom types and positions
            n_layers=7
        )

        betas = get_beta_schedule('linear', beta_start=1e-6, beta_end=1e-2, num_diffusion_timesteps=num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # For q_sample
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))

        # For p_sample
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', torch.log(self.posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_sample(self, x_start, pos_start, t, noise_x=None, noise_pos=None):
        if noise_x is None:
            noise_x = torch.randn_like(x_start)
        if noise_pos is None:
            noise_pos = torch.randn_like(pos_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        noisy_x = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise_x
        noisy_pos = sqrt_alphas_cumprod_t * pos_start + sqrt_one_minus_alphas_cumprod_t * noise_pos
        return noisy_x, noisy_pos

    def p_mean_variance(self, x, t, model_output):
        # The model predicts noise, so we need to compute x_start from it
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        pred_xstart = sqrt_recip_alphas_cumprod_t * (x - sqrt_one_minus_alphas_cumprod_t * model_output)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x_t, pos_t, t, batch):
        target_mask = batch.central_mask & batch.backbone_mask

        # Create complete graph with noisy target
        h = batch.x.clone()
        pos = batch.pos.clone()
        h[target_mask] = x_t
        pos[target_mask] = pos_t

        # Add time embedding
        time_emb = t.repeat(h.size(0), 1)
        h = torch.cat([h, time_emb], dim=1)

        h_out, x_out = self.gnn(h, pos, batch.edge_index, None)

        pred_x_noise_features, _ = torch.split(h_out, [len(atom_to_idx), 3], dim=-1)
        pred_x_noise = pred_x_noise_features[target_mask]

        model_mean, _, model_log_variance = self.p_mean_variance(x=x_t, t=t, model_output=pred_x_noise)

        noise = torch.randn_like(x_t) if t[0] > 0 else 0.
        pred_x = model_mean + (0.5 * model_log_variance).exp() * noise

        # The new positions are in x_out, but only for the target atoms
        pred_pos = x_out[target_mask]

        return pred_x, pred_pos

    def p_losses(self, batch, t):
        target_mask = batch.central_mask & batch.backbone_mask
        x_start = batch.x[target_mask]
        pos_start = batch.pos[target_mask]

        noise_x = torch.randn_like(x_start)
        noise_pos = torch.randn_like(pos_start)

        x_noisy, pos_noisy = self.q_sample(x_start, pos_start, t, noise_x=noise_x, noise_pos=noise_pos)

        # Create complete graph with noisy target
        h = batch.x.clone()
        pos = batch.pos.clone()
        h[target_mask] = x_noisy
        pos[target_mask] = pos_noisy

        # Add time embedding
        time_emb = t.repeat(h.size(0), 1)
        h = torch.cat([h, time_emb], dim=1)

        h_out, pos_out = self.gnn(h, pos, batch.edge_index, None)
        pred_x_features, _ = torch.split(h_out, [len(atom_to_idx), 3], dim=-1)
        pred_x_noise = pred_x_features[target_mask]
        pred_pos_noise = pos_out[target_mask] - pos[target_mask]

        loss_x = F.l1_loss(pred_x_noise, noise_x)
        loss_pos = F.mse_loss(pred_pos_noise, noise_pos)

        return loss_x, loss_pos

    def _common_step(self, batch):
        t = torch.randint(0, self.hparams.num_timesteps, (1,), device=self.device).long()
        loss_x, loss_pos = self.p_losses(batch, t)
        return loss_x + loss_pos

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size, sync_dist=True)

    def test_step(self, batch, _):
        loss = self._common_step(batch)
        self.log('test_loss', loss, logger=True, batch_size=self.hparams.batch_size, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, patience=int(self.hparams.patience*0.5)),
                'monitor': 'val_loss'
            },
        }

    @torch.no_grad()
    def sample(self, batch):
        target_mask = batch.central_mask & batch.backbone_mask
        n_samples = target_mask.sum().item()
        shape_x = (n_samples, len(atom_to_idx))
        shape_pos = (n_samples, 3)

        device = self.betas.device
        x = torch.randn(shape_x, device=device)
        pos = torch.randn(shape_pos, device=device)

        for i in reversed(range(0, self.hparams.num_timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            x, pos = self.p_sample(x, pos, t, batch)
        return x, pos
