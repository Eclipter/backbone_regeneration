import math
import os.path as osp

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree

from utils import atom_to_idx, base_to_idx


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
            nn.Linear(hidden_nf, 1, bias=False),
            nn.Tanh()
        )
        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf)
        )
        self.ln = nn.LayerNorm(hidden_nf)

    def forward(self, h, x, edge_index, edge_attr=None):
        # h: [N, hidden_nf], x: [N, 3]
        row, col = edge_index

        # Message passing
        rel_coords = x[row] - x[col]
        dist = torch.norm(rel_coords, p=2, dim=-1, keepdim=True)
        message_input = torch.cat([h[row], h[col], dist], dim=-1)
        messages = self.message_mlp(message_input)

        # Aggregate messages
        agg_messages = torch.zeros_like(h, dtype=messages.dtype)
        index = col.unsqueeze(1).expand_as(messages)
        agg_messages.scatter_add_(0, index, messages)

        # Update & normalize coordinates
        rel_dir = rel_coords / (dist + 1e-8)
        coord_mult = self.coord_mlp(messages)
        coord_update_src = coord_mult * rel_dir
        coord_update = torch.zeros_like(x, dtype=coord_update_src.dtype)
        index = col.unsqueeze(1).expand_as(coord_update_src)
        coord_update.scatter_add_(0, index, coord_update_src)

        # Normalize coordinate updates by the number of incoming messages to prevent explosion
        num_nodes = x.size(0)
        in_degree = degree(col, num_nodes=num_nodes, dtype=x.dtype).to(x.device).unsqueeze(1)
        x_new = x + coord_update / (in_degree + 1.0)

        # Update features
        node_update_input = torch.cat([h, agg_messages], dim=-1)
        h_new = h + self.node_update_mlp(node_update_input)
        h_new = self.ln(h_new)

        return h_new, x_new


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
    elif beta_schedule == 'cosine':
        return cosine_beta_schedule(num_timesteps)
    else:
        raise NotImplementedError(f'unknown beta schedule: {beta_schedule}')


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class EGNNDiff(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        num_layers,
        eps_directional_head=False,
        eps_use_local_head=True,
        eps_normalize_agg=False,
        act_fn=nn.SiLU()
    ):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.num_layers = num_layers
        self.eps_directional_head = eps_directional_head
        self.eps_use_local_head = eps_use_local_head
        self.eps_normalize_agg = eps_normalize_agg

        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.eps_message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + 4, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        self.eps_coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 3)
        )
        self.eps_coord_scalar_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False),
            nn.Tanh()
        )
        self.eps_head = nn.Sequential(
            nn.Linear(hidden_nf + 3, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 3)
        )

        for i in range(0, num_layers):
            self.add_module(f'gcl_{i}', EquivariantConv(self.hidden_nf, act_fn=act_fn))

    def predict_epsilon(self, h, x, edge_index):
        row, col = edge_index
        rel_coords = x[row] - x[col]
        dist = torch.norm(rel_coords, p=2, dim=-1, keepdim=True)

        message_input = torch.cat([h[row], h[col], rel_coords, dist], dim=-1)
        eps_messages = self.eps_message_mlp(message_input)
        if self.eps_directional_head:
            rel_dir = rel_coords / (dist + 1e-8)
            eps_update_src = self.eps_coord_scalar_mlp(eps_messages) * rel_dir
        else:
            eps_update_src = self.eps_coord_mlp(eps_messages)

        eps = torch.zeros_like(x, dtype=eps_update_src.dtype)
        index = col.unsqueeze(1).expand_as(eps_update_src)
        eps.scatter_add_(0, index, eps_update_src)
        if self.eps_normalize_agg:
            in_degree = degree(col, num_nodes=x.size(0), dtype=x.dtype).to(x.device).unsqueeze(1)
            eps = eps / (in_degree + 1.0)
        if self.eps_use_local_head:
            eps = eps + self.eps_head(torch.cat([h, x], dim=-1))
        return eps

    def forward(self, h, x, edge_index, edge_attr=None):
        h = self.embedding_in(h)
        for i in range(0, self.num_layers):
            h, x = getattr(self, f'gcl_{i}')(h, x, edge_index, edge_attr=edge_attr)
        eps = self.predict_epsilon(h, x, edge_index)
        return h, x, eps


class PytorchLightningModule(pl.LightningModule):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        num_timesteps,
        batch_size,
        lr,
        beta_schedule='linear',
        train_loss_type='mse',
        prediction_target='epsilon',
        eps_directional_head=False,
        eps_use_local_head=True,
        eps_normalize_agg=False,
        train_t_max=None,
        debug_fixed_t=None,
        debug_fixed_noise=False,
        debug_eval_t=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_atom_types = len(atom_to_idx)
        self.time_emb_dim = 32
        self.time_mlp = SinusoidalPositionalEmbeddings(self.time_emb_dim)

        self.gnn = EGNNDiff(
            in_node_nf=self.n_atom_types + self.time_emb_dim + len(base_to_idx) + 1,  # + time_emb_dim for time embedding, +4 for base types, +1 for has_pair
            hidden_nf=hidden_dim,
            num_layers=num_layers,
            eps_directional_head=eps_directional_head,
            eps_use_local_head=eps_use_local_head,
            eps_normalize_agg=eps_normalize_agg
        )

        betas = get_beta_schedule(beta_schedule, num_timesteps=num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
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
        self.register_buffer('posterior_log_variance_clipped', torch.log(getattr(self, 'posterior_variance').clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def _get_target_graph_index(self, batch, target_mask):
        if hasattr(batch, 'batch') and batch.batch is not None and batch.batch.numel() == batch.x.size(0):
            return batch.batch[target_mask].long()
        return torch.zeros(target_mask.sum(), device=target_mask.device, dtype=torch.long)

    def _center_by_graph(self, pos, graph_idx):
        if pos.numel() == 0:
            return pos
        num_graphs = int(graph_idx.max().item()) + 1
        graph_sum = torch.zeros(num_graphs, pos.size(-1), device=pos.device, dtype=pos.dtype)
        graph_sum.index_add_(0, graph_idx, pos)
        graph_count = torch.zeros(num_graphs, 1, device=pos.device, dtype=pos.dtype)
        graph_count.index_add_(0, graph_idx, torch.ones(pos.size(0), 1, device=pos.device, dtype=pos.dtype))
        # Keep coordinates in the zero center-of-gravity subspace to prevent global translation drift.
        graph_mean = graph_sum / graph_count.clamp(min=1.0)
        return pos - graph_mean[graph_idx]

    def on_test_start(self):
        if self.trainer.is_global_zero:
            test_dataset_path = osp.join(getattr(self.trainer.logger, 'log_dir'), 'test_dataset.pt')
            torch.save(getattr(self.trainer, 'datamodule').test_dataset, test_dataset_path)

    def q_sample(self, pos_start, t, noise_pos=None):
        if noise_pos is None:
            noise_pos = torch.randn_like(pos_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, pos_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, pos_start.shape)

        noisy_pos = sqrt_alphas_cumprod_t * pos_start + sqrt_one_minus_alphas_cumprod_t * noise_pos
        return noisy_pos

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _expand_node_t(self, batch, t):
        num_nodes = batch.x.size(0)
        num_graphs = getattr(batch, 'num_graphs', None)
        if t.numel() == 1:
            node_t = t.expand(num_nodes)
        elif hasattr(batch, 'batch') and num_graphs is not None and t.numel() == num_graphs:
            node_t = t[batch.batch]
        elif t.numel() == num_nodes:
            node_t = t
        else:
            raise ValueError('Unsupported t shape.')
        return node_t.long()

    def _build_node_features(self, batch, node_t):
        t_emb = self.time_mlp(node_t.float())
        has_pair_expanded = batch.has_pair.float().unsqueeze(1)
        return torch.cat([batch.x, t_emb, batch.base_types, has_pair_expanded], dim=1)

    def _predict_noise(self, batch, target_mask, pos_target, node_t):
        h = self._build_node_features(batch, node_t)
        pos = batch.pos.clone()
        pos[target_mask] = pos_target
        _, _, eps_out = self.gnn(h, pos, batch.edge_index)
        return eps_out[target_mask]

    def _get_fixed_eval_noise(self, pos_start):
        # Use a fixed CPU generator so validation and test metrics stay deterministic.
        generator = torch.Generator(device='cpu')
        generator.manual_seed(0)
        return torch.randn(pos_start.shape, generator=generator, dtype=pos_start.dtype).to(pos_start.device)

    def _get_fixed_eval_t(self, device):
        eval_t = getattr(self.hparams, 'debug_eval_t')
        if eval_t is None:
            eval_t = getattr(self.hparams, 'debug_fixed_t')
        if eval_t is None:
            eval_t = max(1, getattr(self.hparams, 'num_timesteps') // 2)
        return torch.tensor([eval_t], device=device, dtype=torch.long)

    def _get_training_t(self, device):
        fixed_t = getattr(self.hparams, 'debug_fixed_t')
        if fixed_t is not None:
            return torch.tensor([fixed_t], device=device, dtype=torch.long)
        train_t_max = getattr(self.hparams, 'train_t_max')
        if train_t_max is None:
            train_t_max = getattr(self.hparams, 'num_timesteps') - 1
        train_t_max = min(train_t_max, getattr(self.hparams, 'num_timesteps') - 1)
        return torch.randint(0, train_t_max + 1, (1,), device=device).long()

    def _get_training_noise(self, pos_start):
        if getattr(self.hparams, 'debug_fixed_noise'):
            # Reuse the same noise tensor to make one-batch overfit checks deterministic.
            return self._get_fixed_eval_noise(pos_start)
        return torch.randn_like(pos_start)

    def _reconstruct_pos(self, pos_noisy, pred_noise_pos, target_t):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, target_t, pos_noisy.shape).clamp(min=1e-8)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, target_t, pos_noisy.shape).clamp(min=1e-8)
        return (pos_noisy - sqrt_one_minus_alphas_cumprod_t * pred_noise_pos) / sqrt_alphas_cumprod_t

    def _compute_v_target(self, pos_start, noise_pos, target_t):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, target_t, pos_start.shape).clamp(min=1e-8)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, target_t, pos_start.shape).clamp(min=1e-8)
        return sqrt_alphas_cumprod_t * noise_pos - sqrt_one_minus_alphas_cumprod_t * pos_start

    def _get_prediction_target_tensor(self, pos_start, noise_pos, target_t):
        prediction_target = getattr(self.hparams, 'prediction_target')
        if prediction_target == 'epsilon':
            return noise_pos
        if prediction_target == 'v':
            return self._compute_v_target(pos_start, noise_pos, target_t)
        raise ValueError(f'Unsupported prediction_target: {prediction_target}')

    def _decode_model_output(self, pos_noisy, model_output, target_t):
        prediction_target = getattr(self.hparams, 'prediction_target')
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, target_t, pos_noisy.shape).clamp(min=1e-8)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, target_t, pos_noisy.shape).clamp(min=1e-8)
        if prediction_target == 'epsilon':
            pred_noise_pos = model_output
            pred_pos_start = (pos_noisy - sqrt_one_minus_alphas_cumprod_t * pred_noise_pos) / sqrt_alphas_cumprod_t
            return pred_pos_start, pred_noise_pos
        if prediction_target == 'v':
            pred_pos_start = sqrt_alphas_cumprod_t * pos_noisy - sqrt_one_minus_alphas_cumprod_t * model_output
            pred_noise_pos = sqrt_one_minus_alphas_cumprod_t * pos_noisy + sqrt_alphas_cumprod_t * model_output
            return pred_pos_start, pred_noise_pos
        raise ValueError(f'Unsupported prediction_target: {prediction_target}')

    def _compute_eps_losses(self, pred_noise_pos, noise_pos):
        eps_mse = F.mse_loss(pred_noise_pos, noise_pos)
        eps_rmse = torch.sqrt(eps_mse)
        return eps_mse, eps_rmse

    def _select_training_loss(self, eps_mse, eps_rmse):
        train_loss_type = getattr(self.hparams, 'train_loss_type')
        if train_loss_type == 'mse':
            return eps_mse
        if train_loss_type == 'rmse':
            return eps_rmse
        raise ValueError(f'Unsupported train_loss_type: {train_loss_type}')

    def p_losses(self, batch, t):
        target_mask = batch.central_mask & batch.backbone_mask
        pos_start = batch.pos[target_mask]
        node_t = self._expand_node_t(batch, t)
        target_t = node_t[target_mask]

        noise_pos = self._get_training_noise(pos_start)
        pos_noisy = self.q_sample(pos_start, target_t, noise_pos=noise_pos)
        prediction_target = self._get_prediction_target_tensor(pos_start, noise_pos, target_t)
        model_output = self._predict_noise(batch, target_mask, pos_noisy, node_t)
        _, pred_noise_pos = self._decode_model_output(pos_noisy, model_output, target_t)
        eps_mse, eps_rmse = self._compute_eps_losses(pred_noise_pos, noise_pos)

        target_mse = F.mse_loss(model_output, prediction_target)
        target_rmse = torch.sqrt(target_mse)
        return self._select_training_loss(target_mse, target_rmse), eps_mse, eps_rmse

    @torch.no_grad()
    def p_sample(self, pos_t, t, batch):
        target_mask = batch.central_mask & batch.backbone_mask
        node_t = self._expand_node_t(batch, t)
        target_t = node_t[target_mask]

        model_output = self._predict_noise(batch, target_mask, pos_t, node_t)
        pred_pos_start, _ = self._decode_model_output(pos_t, model_output, target_t)
        model_mean_pos, _, model_log_variance_pos = self.q_posterior_mean_variance(x_start=pred_pos_start, x_t=pos_t, t=target_t)

        noise_pos = torch.randn_like(pos_t)
        nonzero_mask = (target_t > 0).float().unsqueeze(1)
        pred_pos = model_mean_pos + (0.5 * model_log_variance_pos).exp() * noise_pos * nonzero_mask

        return pred_pos

    def training_step(self, batch, _):
        target_mask = batch.central_mask & batch.backbone_mask
        pos_start = batch.pos[target_mask]
        t = self._get_training_t(self.device)
        node_t = self._expand_node_t(batch, t)
        target_t = node_t[target_mask]
        noise_pos = self._get_training_noise(pos_start)
        pos_noisy = self.q_sample(pos_start, target_t, noise_pos=noise_pos)
        prediction_target = self._get_prediction_target_tensor(pos_start, noise_pos, target_t)
        model_output = self._predict_noise(batch, target_mask, pos_noisy, node_t)
        pred_pos_start, pred_noise_pos = self._decode_model_output(pos_noisy, model_output, target_t)
        eps_mse, eps_rmse = self._compute_eps_losses(pred_noise_pos, noise_pos)
        target_mse = F.mse_loss(model_output, prediction_target)
        target_rmse = torch.sqrt(target_mse)
        loss = self._select_training_loss(target_mse, target_rmse)
        train_x0_rmse = torch.sqrt(F.mse_loss(pred_pos_start, pos_start))

        self.log('train_loss', loss, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_target_rmse', target_rmse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_eps_mse', eps_mse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_rmse', eps_rmse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_x0_rmse', train_x0_rmse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def _get_generations(self, batch):
        target_mask = batch.central_mask & batch.backbone_mask
        true_pos = batch.pos[target_mask]
        eval_t = self._get_fixed_eval_t(batch.pos.device)
        node_t = self._expand_node_t(batch, eval_t)
        target_t = node_t[target_mask]
        noise_pos = self._get_fixed_eval_noise(true_pos)
        pos_noisy = self.q_sample(true_pos, target_t, noise_pos=noise_pos)
        model_output = self._predict_noise(batch, target_mask, pos_noisy, node_t)
        pred_pos, pred_noise_pos = self._decode_model_output(pos_noisy, model_output, target_t)
        eps_mse, eps_rmse = self._compute_eps_losses(pred_noise_pos, noise_pos)

        return true_pos, pred_pos, eps_mse, eps_rmse

    def validation_step(self, batch, _):
        true_pos, pred_pos, eps_mse, eps_rmse = self._get_generations(batch)

        rmse = torch.sqrt(F.mse_loss(pred_pos, true_pos))
        self.log('val_eps_mse', eps_mse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_eps_rmse', eps_rmse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_rmse', rmse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, _):
        true_pos, pred_pos, eps_mse, eps_rmse = self._get_generations(batch)

        rmse = torch.sqrt(F.mse_loss(pred_pos, true_pos))
        self.log('test_eps_mse', eps_mse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_eps_rmse', eps_rmse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_rmse', rmse, batch_size=len(batch.pos), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(self.parameters(), lr=getattr(self.hparams, 'lr'), betas=(0.9, 0.999), weight_decay=0)

        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=400,
            cooldown=50,
            factor=0.5,
            mode='min'
        )

        return optimizer
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_rmse'
        #     },
        # }
