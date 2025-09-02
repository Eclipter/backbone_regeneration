import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torchmetrics.classification import CohenKappa

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

        # Normalize coordinate updates by the number of incoming messages to prevent explosion
        num_nodes = x.size(0)
        in_degree = degree(col, num_nodes=num_nodes, dtype=x.dtype).to(x.device).unsqueeze(1)
        x_new = x + coord_update / (in_degree + 1.0)

        # Update features
        node_update_input = torch.cat([h, agg_messages], dim=-1)
        h_new = h + self.node_update_mlp(node_update_input)

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
    def __init__(self, hidden_dim, num_timesteps, batch_size, lr):
        super().__init__()
        self.save_hyperparameters()

        self.n_atom_types = len(atom_to_idx)

        self.gnn = EGNNDiff(
            in_node_nf=self.n_atom_types + 1,  # +1 for time embedding
            hidden_nf=hidden_dim,
            out_node_nf=self.n_atom_types,  # atom types and positions
            n_layers=7
        )

        self.train_kappa = CohenKappa(task='multiclass', num_classes=self.n_atom_types)
        self.val_kappa = CohenKappa(task='multiclass', num_classes=self.n_atom_types)
        self.test_kappa = CohenKappa(task='multiclass', num_classes=self.n_atom_types)

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

    def q_sample(self, atom_names_start, pos_start, t, noise_atom_names=None, noise_pos=None):
        if noise_atom_names is None:
            noise_atom_names = torch.randn_like(atom_names_start)
        if noise_pos is None:
            noise_pos = torch.randn_like(pos_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, atom_names_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, atom_names_start.shape)

        noisy_atom_names = sqrt_alphas_cumprod_t * atom_names_start + sqrt_one_minus_alphas_cumprod_t * noise_atom_names
        noisy_pos = sqrt_alphas_cumprod_t * pos_start + sqrt_one_minus_alphas_cumprod_t * noise_pos
        return noisy_atom_names, noisy_pos

    def p_mean_variance(self, x, t, model_output):
        # The model predicts noise, so we need to compute x_start from it
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        pred_x_start = sqrt_recip_alphas_cumprod_t * (x - sqrt_one_minus_alphas_cumprod_t * model_output)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_start=pred_x_start, x_t=x, t=t)
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
    def p_sample(self, atom_names_t, pos_t, t, batch):
        target_mask = batch.central_mask & batch.backbone_mask

        # Create complete graph with noisy target
        h = batch.x.clone()
        pos = batch.pos.clone()
        h[target_mask] = atom_names_t
        pos[target_mask] = pos_t

        # Add time embedding
        time_emb = t.repeat(h.size(0), 1)
        h = torch.cat([h, time_emb], dim=1)

        h_out, x_out = self.gnn(h, pos, batch.edge_index, None)

        # Process atom names (model predicts noise)
        pred_atom_names_noise = h_out[target_mask]
        model_mean_atom_names, _, _ = self.p_mean_variance(x=atom_names_t, t=t, model_output=pred_atom_names_noise)

        # ODE sampler is deterministic
        pred_atom_names = model_mean_atom_names

        # Process positions (model predicts x_0)
        pred_pos_start = x_out[target_mask]
        model_mean_pos, _, _ = self.q_posterior_mean_variance(x_start=pred_pos_start, x_t=pos_t, t=t)

        # ODE sampler is deterministic
        pred_pos = model_mean_pos

        return pred_atom_names, pred_pos

    def p_losses(self, batch, t):
        target_mask = batch.central_mask & batch.backbone_mask
        atom_names_start = batch.x[target_mask]
        pos_start = batch.pos[target_mask]

        noise_atom_names = torch.randn_like(atom_names_start)
        noise_pos = torch.randn_like(pos_start)

        atom_names_noisy, pos_noisy = self.q_sample(atom_names_start, pos_start, t, noise_atom_names=noise_atom_names, noise_pos=noise_pos)

        # Create complete graph with noisy target
        h = batch.x.clone()
        pos = batch.pos.clone()
        h[target_mask] = atom_names_noisy
        pos[target_mask] = pos_noisy

        # Add time embedding
        time_emb = t.repeat(h.size(0), 1)
        h = torch.cat([h, time_emb], dim=1)

        h_out, pos_out = self.gnn(h, pos, batch.edge_index, None)

        # Atom names loss (model predicts noise)
        pred_atom_names_noise = h_out[target_mask]

        # Denoise the atom names to calculate the loss
        sqrt_recip_alphas_cumprod_t = extract(self.sqrt_recip_alphas_cumprod, t, atom_names_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, atom_names_noisy.shape)
        pred_atom_names_start = sqrt_recip_alphas_cumprod_t * (atom_names_noisy - sqrt_one_minus_alphas_cumprod_t * pred_atom_names_noise)

        true_atom_names_idx = torch.argmax(atom_names_start, dim=1)

        atom_names_loss = F.cross_entropy(pred_atom_names_start, true_atom_names_idx)

        # Position loss (model predicts x_0)
        pred_pos_start = pos_out[target_mask]
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, pos_noisy.shape)
        sqrt_one_minus_alphas_cumprod_t_pos = extract(self.sqrt_one_minus_alphas_cumprod, t, pos_noisy.shape)
        implied_noise_pos = (pos_noisy - sqrt_alphas_cumprod_t * pred_pos_start) / sqrt_one_minus_alphas_cumprod_t_pos
        pos_loss = torch.sqrt(F.mse_loss(implied_noise_pos, noise_pos))

        # Calculate accuracy for atom names
        pred_atom_names_idx = torch.argmax(pred_atom_names_start, dim=1)

        return atom_names_loss, pos_loss, pred_atom_names_idx, true_atom_names_idx

    def training_step(self, batch, _):
        t = torch.randint(0, self.hparams.num_timesteps, (1,), device=self.device).long()
        atom_name_loss, pos_loss, pred_atom_names_idx, true_atom_names_idx = self.p_losses(batch, t)

        self.train_kappa(pred_atom_names_idx, true_atom_names_idx)

        self.log_dict({
            'train_loss': atom_name_loss+pos_loss,
            'train_rmse': pos_loss,
            'train_kappa': self.train_kappa
        }, on_step=False, on_epoch=True, sync_dist=True)
        return atom_name_loss + pos_loss

    def _get_generations(self, batch):
        target_mask = batch.central_mask & batch.backbone_mask
        true_pos = batch.pos[target_mask]
        true_atom_names_idx = torch.argmax(batch.x[target_mask], dim=1)

        pred_atom_names, pred_pos = self.sample(batch)
        pred_atom_names_idx = torch.argmax(pred_atom_names, dim=1)

        return true_pos, true_atom_names_idx, pred_pos, pred_atom_names_idx

    def validation_step(self, batch, _):
        true_pos, true_atom_names_idx, pred_pos, pred_atom_names_idx = self._get_generations(batch)

        rmse = torch.sqrt(F.mse_loss(pred_pos, true_pos))

        # Normalize RMSE by the standard deviation of true positions' norms for better scaling
        std_dev_true_pos = torch.std(torch.norm(true_pos, dim=1))
        # Avoid division by zero
        if std_dev_true_pos > 1e-6:
            nrmse = rmse / std_dev_true_pos
        else:
            nrmse = rmse

        self.val_kappa(pred_atom_names_idx, true_atom_names_idx)

        # We want to minimize nrmse and maximize kappa. So we minimize (nrmse - kappa).
        combined_score = nrmse - self.val_kappa.compute()

        # # Debugging prints, only on rank 0
        # if self.global_rank == 0:
        #     print(f'--- Validation Step Stats ---')
        #     print(f'RMSE: {rmse.item():.4f}')
        #     print(f'Kappa: {self.val_kappa.compute().item():.4f}')
        #     print(f'Combined Score: {combined_score.item():.4f}')
        #     print(f'-----------------------------')

        self.log_dict({
            'val_rmse': rmse,
            'val_kappa': self.val_kappa,
            'val_combined_score': combined_score
        }, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, _):
        true_pos, true_atom_names_idx, pred_pos, pred_atom_names_idx = self._get_generations(batch)

        rmse = torch.sqrt(F.mse_loss(pred_pos, true_pos))

        # Normalize RMSE by the standard deviation of true positions' norms for better scaling
        std_dev_true_pos = torch.std(torch.norm(true_pos, dim=1))
        # Avoid division by zero
        if std_dev_true_pos > 1e-6:
            nrmse = rmse / std_dev_true_pos
        else:
            nrmse = rmse

        self.test_kappa(pred_atom_names_idx, true_atom_names_idx)
        combined_score = nrmse - self.test_kappa.compute()

        self.log_dict({
            'test_rmse': rmse,
            'test_kappa': self.test_kappa,
            'test_combined_score': combined_score
        }, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def sample(self, batch):
        target_mask = batch.central_mask & batch.backbone_mask
        n_samples = target_mask.sum().item()
        shape_atom_names = (n_samples, self.n_atom_types)
        shape_pos = (n_samples, 3)

        device = self.betas.device
        atom_names = torch.randn(shape_atom_names, device=device)
        pos = torch.randn(shape_pos, device=device)

        for i in reversed(range(0, self.hparams.num_timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            atom_names, pos = self.p_sample(atom_names, pos, t, batch)

        #     # Debugging prints, only on rank 0
        #     if self.global_rank == 0:
        #         if torch.isnan(pos).any() or torch.isinf(pos).any():
        #             print(f'!!! NaN or Inf detected in "pos" at timestep t={i}')
        #             break
        #         if i % 100 == 0:
        #             print(f'Sample step t={i}, pos mean: {pos.mean().item():.4f}, pos std: {pos.std().item():.4f}')

        # if self.global_rank == 0:
        #     print(f'--- Final sampled pos stats ---')
        #     print(f'Has NaN: {torch.isnan(pos).any().item()}')
        #     print(f'Has Inf: {torch.isinf(pos).any().item()}')
        #     if not (torch.isnan(pos).any() or torch.isinf(pos).any()):
        #         print(f'pos mean: {pos.mean().item():.4f}, pos std: {pos.std().item():.4f}, pos min: {pos.min().item():.4f}, pos max: {pos.max().item():.4f}')
        #     print(f'-----------------------------')

        return atom_names, pos
