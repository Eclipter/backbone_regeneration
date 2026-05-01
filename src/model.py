import math
import os.path as osp

import lightning.pytorch as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import N_CHAIN_END_CLASSES, atom_to_idx, base_to_idx


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


class GraphConv(nn.Module):
    def __init__(self, hidden_nf, act_fn_cls=nn.SiLU):
        super().__init__()
        self.hidden_nf = hidden_nf
        act_fn = act_fn_cls
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + 6, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn()
        )
        self.coord_update_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, 3),
            nn.Tanh()
        )
        self.node_update_mlp = nn.Sequential(
            nn.Linear(hidden_nf + hidden_nf, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, hidden_nf)
        )

    def forward(self, h, x, edge_index):
        # h: [N, hidden_nf], x: [N, 3]
        row, col = edge_index[0], edge_index[1]

        # Message passing
        message_input = torch.cat([h[row], h[col], x[row], x[col]], dim=-1)
        messages = self.message_mlp(message_input)

        # Aggregate messages
        agg_messages = torch.zeros_like(h, dtype=messages.dtype)
        index = col.unsqueeze(1).expand_as(messages)
        agg_messages.scatter_add_(0, index, messages)

        # Update & normalize coordinates
        coord_update_src = self.coord_update_mlp(messages)
        coord_update = torch.zeros_like(x, dtype=coord_update_src.dtype)
        index = col.unsqueeze(1).expand_as(coord_update_src)
        coord_update.scatter_add_(0, index, coord_update_src)

        # Normalize coordinate updates by the number of incoming messages to prevent explosion
        ones = torch.ones(col.size(0), device=x.device, dtype=x.dtype)
        in_degree = torch.zeros(x.size(0), device=x.device, dtype=x.dtype).scatter_add(0, col, ones).unsqueeze(1)
        x_new = x + coord_update / (in_degree + 1.0)

        # Update features
        node_update_input = torch.cat([h, agg_messages], dim=-1)
        h_new = h + self.node_update_mlp(node_update_input)

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


class GraphDiffusionDenoiser(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, num_layers, act_fn_cls=nn.SiLU):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.num_layers = num_layers
        act_fn = act_fn_cls

        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.eps_message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_nf + 6, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn()
        )
        self.eps_vector_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, 3),
            nn.Tanh()
        )
        self.eps_head = nn.Sequential(
            nn.Linear(hidden_nf + 3, hidden_nf),
            act_fn(),
            nn.Linear(hidden_nf, 3)
        )

        self.layers = nn.ModuleList([
            GraphConv(self.hidden_nf, act_fn_cls=act_fn)
            for _ in range(num_layers)
        ])

    def predict_epsilon(self, h, x, edge_index):
        row, col = edge_index[0], edge_index[1]

        message_input = torch.cat([h[row], h[col], x[row], x[col]], dim=-1)
        eps_messages = self.eps_message_mlp(message_input)
        eps_update_src = self.eps_vector_mlp(eps_messages)

        eps = torch.zeros_like(x, dtype=eps_update_src.dtype)
        index = col.unsqueeze(1).expand_as(eps_update_src)
        eps.scatter_add_(0, index, eps_update_src)
        eps = eps + self.eps_head(torch.cat([h, x], dim=-1))
        return eps

    def forward(self, h, x, edge_index):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        eps = self.predict_epsilon(h, x, edge_index)
        return h, x, eps


class PytorchLightningModule(pl.LightningModule):
    def __init__(self, hidden_dim, num_layers, num_timesteps, batch_size, lr,
                 lr_scheduler, lr_scheduler_patience, lr_scheduler_threshold, lr_scheduler_cooldown,
                 beta_schedule, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        self.n_atom_types = len(atom_to_idx)
        self.time_emb_dim = 32
        self.time_mlp = SinusoidalPositionalEmbeddings(self.time_emb_dim)

        self.gnn = GraphDiffusionDenoiser(
            # +1 for has_pair, +N_CHAIN_END_CLASSES for chain_end_class one-hot, +1 for is_target
            in_node_nf=self.n_atom_types + self.time_emb_dim + len(base_to_idx) + 2 + N_CHAIN_END_CLASSES,
            hidden_nf=hidden_dim,
            num_layers=num_layers,
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

        # For p_sample
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped', torch.log(getattr(self, 'posterior_variance').clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))
        self._epoch_window_rmsds = {}

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

    def _target_mask(self, batch):
        return batch.is_target & batch.backbone_mask & batch.present_mask

    def _central_context_hidden_mask(self, batch):
        central_target_nodes = batch.is_target & batch.central_mask
        if not central_target_nodes.any():
            return torch.zeros_like(batch.backbone_mask)

        graph_ids = getattr(batch, 'batch', None)
        if graph_ids is not None:
            num_graphs = getattr(batch, 'num_graphs', None)
            if num_graphs is None:
                num_graphs = int(graph_ids.max().item()) + 1
            graph_has_central_target = torch.zeros(
                num_graphs,
                device=batch.backbone_mask.device,
                dtype=torch.bool
            )
            graph_has_central_target[graph_ids[central_target_nodes]] = True
            central_target_graph_nodes = graph_has_central_target[graph_ids]
        else:
            central_target_graph_nodes = torch.ones_like(batch.backbone_mask)

        return central_target_graph_nodes & ~batch.central_mask & batch.backbone_mask

    def _build_node_features(self, batch, node_t):
        t_emb = self.time_mlp(node_t.float())
        has_pair_expanded = batch.has_pair.float().unsqueeze(1)
        is_target_expanded = batch.is_target.float().unsqueeze(1)
        chain_end = getattr(batch, 'chain_end_class')
        return torch.cat([
            batch.x, t_emb, batch.base_types,
            has_pair_expanded, chain_end, is_target_expanded,
        ], dim=1)

    def _predict_noise(self, batch, target_mask, pos_target, node_t):
        h = self._build_node_features(batch, node_t)
        pos = batch.pos.clone()
        pos[target_mask] = pos_target
        pos[self._central_context_hidden_mask(batch)] = 0.0
        _, _, eps_out = self.gnn(h, pos, batch.edge_index)
        return eps_out[target_mask]

    def _decode_epsilon_to_x0(self, pos_noisy, pred_noise_pos, target_t):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, target_t, pos_noisy.shape).clamp(min=1e-8)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, target_t, pos_noisy.shape).clamp(min=1e-8)
        return (pos_noisy - sqrt_one_minus_alphas_cumprod_t * pred_noise_pos) / sqrt_alphas_cumprod_t

    @torch.no_grad()
    def p_sample(self, pos_t, t, batch):
        target_mask = self._target_mask(batch)
        node_t = self._expand_node_t(batch, t)
        target_t = node_t[target_mask]

        model_output = self._predict_noise(batch, target_mask, pos_t, node_t)
        pred_pos_start = self._decode_epsilon_to_x0(pos_t, model_output, target_t)
        model_mean_pos, _, model_log_variance_pos = self.q_posterior_mean_variance(x_start=pred_pos_start, x_t=pos_t, t=target_t)

        noise_pos = torch.randn_like(pos_t)
        nonzero_mask = (target_t > 0).float().unsqueeze(1)
        pred_pos = model_mean_pos + (0.5 * model_log_variance_pos).exp() * noise_pos * nonzero_mask

        return pred_pos

    @torch.no_grad()
    def p_sample_loop(self, batch):
        """Full T-step reverse diffusion from Gaussian noise to predicted clean positions."""
        target_mask = self._target_mask(batch)
        true_pos = batch.pos[target_mask]
        pos_t = torch.randn_like(true_pos)
        T = getattr(self.hparams, 'num_timesteps')
        for t_idx in reversed(range(T)):
            t = torch.tensor([t_idx], device=true_pos.device, dtype=torch.long)
            pos_t = self.p_sample(pos_t, t, batch)
        return true_pos, pos_t

    def _collect_epoch_window_rmsd(self, metric_name, pred_pos, true_pos, graph_ids):
        if pred_pos.numel() == 0:
            return

        squared_dist = torch.sum((pred_pos - true_pos) ** 2, dim=1)
        num_graphs = int(graph_ids.max().item()) + 1
        msd_sum = squared_dist.new_zeros(num_graphs).scatter_add_(0, graph_ids, squared_dist)
        counts = squared_dist.new_zeros(num_graphs).scatter_add_(0, graph_ids, torch.ones_like(squared_dist))
        window_rmsd = torch.sqrt((msd_sum[counts > 0] / counts[counts > 0]).clamp(min=0.0))
        self._epoch_window_rmsds.setdefault(metric_name, []).append(window_rmsd.detach())

    def _log_eval_rmsd_components(self, stage, batch, pred_pos, true_pos, atom_names=None):
        target_mask = self._target_mask(batch)
        if atom_names is not None:
            atom_names = atom_names.to(device=pred_pos.device)
            op1_indices = (atom_names == atom_to_idx['OP1']).nonzero(as_tuple=False).flatten()
            op2_indices = (atom_names == atom_to_idx['OP2']).nonzero(as_tuple=False).flatten()
            if op1_indices.numel() == op2_indices.numel() and op1_indices.numel() > 0:
                same_msd = (
                    torch.sum((pred_pos[op1_indices] - true_pos[op1_indices]) ** 2, dim=1)
                    + torch.sum((pred_pos[op2_indices] - true_pos[op2_indices]) ** 2, dim=1)
                )
                swapped_msd = (
                    torch.sum((pred_pos[op1_indices] - true_pos[op2_indices]) ** 2, dim=1)
                    + torch.sum((pred_pos[op2_indices] - true_pos[op1_indices]) ** 2, dim=1)
                )
                swap_mask = swapped_msd < same_msd
                if swap_mask.any():
                    swap_op1_indices = op1_indices[swap_mask]
                    swap_op2_indices = op2_indices[swap_mask]
                    pred_pos = pred_pos.clone()
                    pred_op1_pos = pred_pos[swap_op1_indices].clone()
                    pred_pos[swap_op1_indices] = pred_pos[swap_op2_indices]
                    pred_pos[swap_op2_indices] = pred_op1_pos
        central_target_mask = batch.central_mask[target_mask]
        target_graph_ids = batch.batch[target_mask]
        self._collect_epoch_window_rmsd(stage, pred_pos, true_pos, target_graph_ids)
        self._collect_epoch_window_rmsd(
            f'central_{stage}',
            pred_pos[central_target_mask],
            true_pos[central_target_mask],
            target_graph_ids[central_target_mask]
        )
        self._collect_epoch_window_rmsd(
            f'edge_{stage}',
            pred_pos[~central_target_mask],
            true_pos[~central_target_mask],
            target_graph_ids[~central_target_mask]
        )

    def _gather_variable_length(self, values):
        values = values.to(self.device)
        if not (dist.is_available() and dist.is_initialized()):
            return values

        local_size = torch.tensor([values.numel()], device=values.device, dtype=torch.long)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_sizes, local_size)

        max_size = int(torch.stack(gathered_sizes).max().item())
        if max_size == 0:
            return values

        padded = F.pad(values, (0, max_size - values.numel()))
        gathered_values = [torch.empty(max_size, device=values.device, dtype=values.dtype) for _ in gathered_sizes]
        dist.all_gather(gathered_values, padded)
        return torch.cat([
            gathered[:int(size.item())]
            for gathered, size in zip(gathered_values, gathered_sizes)
        ])

    def _finalize_epoch_rmsd(self, metric_name):
        local_values = self._epoch_window_rmsds.pop(metric_name, [])
        local_window_rmsds = torch.cat(local_values) if local_values else torch.empty(0, device=self.device)
        window_rmsds = self._gather_variable_length(local_window_rmsds)
        if window_rmsds.numel() == 0:
            return

        rmsd = torch.quantile(window_rmsds.float(), 0.5)
        self.log(
            f'{metric_name}_rmsd',
            rmsd,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
            logger=False
        )
        if self.trainer.is_global_zero:
            self.logger.experiment.add_scalar(f'{metric_name}_rmsd', rmsd, self.current_epoch)  # type: ignore

    def training_step(self, batch, _):
        target_mask = self._target_mask(batch)
        pos_start = batch.pos[target_mask]
        num_graphs = getattr(batch, 'num_graphs', 1)
        t = torch.randint(0, getattr(self.hparams, 'num_timesteps'), (num_graphs,), device=self.device).long()
        node_t = self._expand_node_t(batch, t)
        target_t = node_t[target_mask]
        noise_pos = torch.randn_like(pos_start)
        pos_noisy = self.q_sample(pos_start, target_t, noise_pos=noise_pos)
        model_output = self._predict_noise(batch, target_mask, pos_noisy, node_t)
        rmse = torch.sqrt(F.mse_loss(model_output, noise_pos))
        self.log(
            'train_rmse_step',
            rmse,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch.x.size(0),
            logger=False
        )

        return rmse

    def on_train_epoch_end(self):
        metric = self.trainer.callback_metrics['train_rmse_step']
        self.logger.experiment.add_scalar('train_rmse', metric, self.current_epoch)  # type: ignore

    def validation_step(self, batch, _):
        true_pos, gen_pos = self.p_sample_loop(batch)
        atom_names = batch.x[self._target_mask(batch)].argmax(dim=1)
        self._log_eval_rmsd_components('val', batch, gen_pos, true_pos, atom_names)

    def on_validation_epoch_end(self):
        self._finalize_epoch_rmsd('val')
        self._finalize_epoch_rmsd('central_val')
        self._finalize_epoch_rmsd('edge_val')

    def test_step(self, batch, _):
        true_pos, gen_pos = self.p_sample_loop(batch)
        atom_names = batch.x[self._target_mask(batch)].argmax(dim=1)
        self._log_eval_rmsd_components('test', batch, gen_pos, true_pos, atom_names)

    def on_test_epoch_end(self):
        self._finalize_epoch_rmsd('test')
        self._finalize_epoch_rmsd('central_test')
        self._finalize_epoch_rmsd('edge_test')

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(self.parameters(), lr=getattr(self.hparams, 'lr'), weight_decay=getattr(self.hparams, 'weight_decay'))

        lr_scheduler = getattr(self.hparams, 'lr_scheduler')
        if lr_scheduler is None:
            return {'optimizer': optimizer}
        if lr_scheduler != 'ReduceLROnPlateau':
            raise NotImplementedError(f'unknown lr scheduler: {lr_scheduler}')

        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=getattr(self.hparams, 'lr_scheduler_patience'),
            threshold=getattr(self.hparams, 'lr_scheduler_threshold'),
            cooldown=getattr(self.hparams, 'lr_scheduler_cooldown')
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_rmsd'
            },
        }

    @torch.no_grad()
    def sample(self, batch):
        _, gen_pos = self.p_sample_loop(batch)
        return gen_pos
