import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.nn import BatchNorm, Gate
from torch_geometric.utils import softmax as tg_softmax


# Basic sinusoidal timestep embedding
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()

        self.dim = dim
        self.max_period = max_period

    def forward(self, timestep):
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(timestep.device)
        args = timestep[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


# An SE(3) equivariant GNN for the denoising model
class DenoisingGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_atom_types=4):
        super().__init__()
        # In E3NN, we define representations by o3.Irreps.
        # 'o3.Irreps("1x0e")' is 1 scalar (l=0, even parity).
        # 'o3.Irreps("1x1o")' is 1 vector (l=1, odd parity).
        self.node_ireps_in = o3.Irreps(f'{num_atom_types}x0e')  # One-hot atom types
        self.node_ireps_hidden = o3.Irreps(f'{hidden_dim}x0e + {hidden_dim//4}x1o')  # scalars and vectors
        self.node_ireps_out = o3.Irreps(f'{num_atom_types}x0e + 1x1o')  # atom type noise + position noise

        self.time_emb = TimestepEmbedding(hidden_dim)
        self.node_emb = o3.Linear(self.node_ireps_in, self.node_ireps_hidden)

        self.condition_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        scalar_dim = sum(mul * ir.dim for mul, ir in self.node_ireps_hidden if ir.l == 0)

        self.conv1 = EquivariantConv(self.node_ireps_hidden, scalar_dim)
        self.conv2 = EquivariantConv(self.conv1.irreps_out, self.conv1.scalar_out_dim)

        self.node_out = o3.Linear(self.conv2.irreps_out, self.node_ireps_out)
        self.edge_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, atom_types, pos, edge_index, timestep, batch_map, batch_obj, edge_candidates):
        time_emb = self.time_emb(timestep)

        # Separate atom features (scalars) and positions (vectors)
        # Note: In e3nn, positions are handled by relative vectors in convolutions
        # The `atom_types` here are the invariant features (atom types)
        node_h = self.node_emb(atom_types)

        # Condition on the nitrogenous bases, without any backbone information
        is_base = ~batch_obj.backbone_mask
        is_condition = is_base

        condition_nodes_atom_types = batch_obj.atom_types[is_condition]
        condition_nodes_batch_map = batch_obj.batch[is_condition]

        num_graphs = batch_obj.num_graphs if hasattr(batch_obj, 'num_graphs') else 1
        scalar_dim = sum(mul * ir.dim for mul, ir in self.node_ireps_hidden if ir.l == 0)

        if condition_nodes_atom_types.size(0) == 0:
            pooled_condition = torch.zeros(num_graphs, scalar_dim, device=atom_types.device)
        else:
            # For conditioning, we only use the scalar features
            condition_emb = self.node_emb(condition_nodes_atom_types)[:, :scalar_dim]
            # Manual global_mean_pool
            pooled_condition = torch.zeros(num_graphs, scalar_dim, device=atom_types.device)
            pooled_condition.index_add_(0, condition_nodes_batch_map, condition_emb)
            graph_sizes = torch.bincount(condition_nodes_batch_map, minlength=num_graphs).float().unsqueeze(1).clamp(min=1)
            pooled_condition = pooled_condition / graph_sizes

        time_h = self.time_proj(time_emb)
        condition_h = self.condition_proj(pooled_condition)

        # Add time and condition embeddings to the scalar part of node embeddings
        node_h[:, :scalar_dim] = node_h[:, :scalar_dim] + time_h[batch_map] + condition_h[batch_map]

        h = self.conv1(node_h, pos, edge_index)
        h = self.conv2(h, pos, edge_index)

        node_noise_pred = self.node_out(h)

        if edge_candidates.shape[1] > 0:
            row, col = edge_candidates[0], edge_candidates[1]
            # Use scalar features for edge prediction
            edge_h = torch.cat([h[row, :scalar_dim], h[col, :scalar_dim]], dim=1)
            edge_logits = self.edge_out(edge_h).squeeze(-1)
        else:
            edge_logits = torch.empty(0, device=h.device)

        return node_noise_pred, edge_logits


class EquivariantConv(nn.Module):
    def __init__(self, irreps_in, scalar_dim):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = self.irreps_in  # Ensure input and output irreps are the same
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=1)
        self.scalar_in_dim = scalar_dim

        self.tensor_prod = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_sh, self.irreps_in, shared_weights=False
        )
        self.tp_weights = nn.Parameter(torch.randn(self.tensor_prod.weight_numel).unsqueeze(0))
        self.fc = o3.Linear(self.irreps_in, self.irreps_in)

        self.attention_mlp = nn.Sequential(
            nn.Linear(2 * self.scalar_in_dim, 1),
            nn.LeakyReLU(0.2)
        )

        s_mul, v_mul = 0, 0
        for mul, ir in self.irreps_out:
            if ir.l == 0 and ir.p == 1:  # 0e
                s_mul = mul
            elif ir.l == 1 and ir.p == -1:  # 1o
                v_mul = mul

        if v_mul > 0:
            if s_mul < v_mul:
                raise ValueError(f'Not enough scalars ({s_mul}) to gate vectors ({v_mul})')

            s_gated_mul = s_mul - v_mul

            self.gate = Gate(
                f'{s_gated_mul}x0e' if s_gated_mul > 0 else '',
                [torch.relu] if s_gated_mul > 0 else [],
                f'{v_mul}x0e', [torch.sigmoid],
                f'{v_mul}x1o'
            )
        else:
            self.gate = Gate(f'{s_mul}x0e', [torch.relu])  # Gate with only scalars

        self.bn = BatchNorm(self.irreps_out)
        self.irreps_out = self.gate.irreps_out
        self.scalar_out_dim = sum(mul * ir.dim for mul, ir in self.irreps_out if ir.l == 0)

    def forward(self, node_h, pos, edge_index):
        row, col = edge_index
        edge_vec = pos[row] - pos[col]
        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')

        # Message
        msg = self.tensor_prod(node_h[col], edge_sh, self.tp_weights)

        # Attention
        h_row_scalars = node_h[row, :self.scalar_in_dim]
        h_col_scalars = node_h[col, :self.scalar_in_dim]
        attention_input = torch.cat([h_row_scalars, h_col_scalars], dim=-1)
        attention_logits = self.attention_mlp(attention_input).squeeze(-1)
        attention_weights = tg_softmax(attention_logits, row, num_nodes=node_h.shape[0])

        # Aggregate (weighted sum)
        msg_weighted = msg * attention_weights.unsqueeze(-1)
        node_h_agg = torch.zeros_like(node_h)
        node_h_agg.index_add_(0, row, msg_weighted)

        # Update
        node_h_new = self.fc(node_h_agg) + node_h
        node_h_new = self.bn(node_h_new)
        node_h_new = self.gate(node_h_new)

        return node_h_new


class Model(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.num_timesteps = self.hparams['num_timesteps']
        self.denoising_model = DenoisingGNN(
            self.hparams['hidden_dim'], num_atom_types=4
        )

        betas = self.cosine_schedule(self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def cosine_schedule(self, T, s=0.008):
        """Cosine schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'."""
        steps = T + 1
        t = torch.linspace(0, T, steps, dtype=torch.float32)
        f_t = torch.cos(((t / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        return torch.clip(betas, 0, 0.999)

    def q_sample(self, x_start, timestep, noise=None):
        """Forward diffusion process: noise the data."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timestep].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep].view(-1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _common_step(self, batch):
        """Calculates the loss for the reverse process using masks."""
        batch_size = batch.num_graphs
        timestep = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

        # x_start is atom features (scalars), pos_start is coordinates (vectors)
        atom_types_start = batch.atom_types
        pos_start = batch.pos

        # Noise scalars and vectors separately
        atom_types_noise = torch.randn_like(atom_types_start)
        pos_noise = torch.randn_like(pos_start)
        noisy_atom_types = self.q_sample(atom_types_start, timestep[batch.batch], noise=atom_types_noise)
        noisy_pos = self.q_sample(pos_start, timestep[batch.batch], noise=pos_noise)

        is_target_node = batch.backbone_mask & batch.central_mask
        if is_target_node.sum() == 0:
            return None

        target_node_indices = torch.where(is_target_node)[0]
        if len(target_node_indices) < 2:
            edge_candidates = torch.empty((2, 0), dtype=torch.long, device=batch.atom_types.device)
        else:
            row, col = torch.triu_indices(len(target_node_indices), len(target_node_indices), 1, device=batch.atom_types.device)
            edge_candidates = torch.stack([target_node_indices[row], target_node_indices[col]], dim=0)

        predicted_noise, predicted_edge_logits = self.denoising_model(
            atom_types=noisy_atom_types, pos=noisy_pos, edge_index=batch.edge_index, timestep=timestep, batch_map=batch.batch, batch_obj=batch,
            edge_candidates=edge_candidates
        )

        # Separate predicted noise into scalar and vector parts
        num_atom_types = self.denoising_model.node_ireps_in.dim
        pred_atom_types_noise = predicted_noise[:, :num_atom_types]
        pred_pos_noise = predicted_noise[:, num_atom_types:]

        atom_types_loss = nn.MSELoss()(pred_atom_types_noise[is_target_node], atom_types_noise[is_target_node])
        pos_loss = nn.MSELoss()(pred_pos_noise[is_target_node], pos_noise[is_target_node])
        node_loss = atom_types_loss + pos_loss

        if edge_candidates.shape[1] > 0:
            num_nodes_total = batch.num_nodes
            gt_adj_sparse = torch.sparse_coo_tensor(
                batch.edge_index,
                torch.ones(batch.edge_index.size(1), device=batch.atom_types.device),
                size=(num_nodes_total, num_nodes_total)
            ).to_dense()
            gt_adj_for_candidates = gt_adj_sparse[edge_candidates[0], edge_candidates[1]]
            edge_loss = nn.BCEWithLogitsLoss()(predicted_edge_logits, gt_adj_for_candidates)
        else:
            edge_loss = (predicted_edge_logits.sum() * 0.0)

        return node_loss + edge_loss

    def training_step(self, batch, _):
        loss = self._common_step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams['batch_size'], sync_dist=True)  # type: ignore
        return loss

    def validation_step(self, batch, _):
        loss = self._common_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams['batch_size'], sync_dist=True)  # type: ignore

    def test_step(self, batch, _):
        loss = self._common_step(batch)
        self.log('test_loss', loss, batch_size=self.hparams['batch_size'], sync_dist=True)  # type: ignore

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return optimizer

    @torch.no_grad()
    def sample(self, condition_graph, num_nodes):
        num_atom_types = self.denoising_model.node_ireps_in.dim
        atom_types_shape = (num_nodes, num_atom_types)
        pos_shape = (num_nodes, 3)

        atom_types = torch.randn(atom_types_shape, device=self.device)
        pos = torch.randn(pos_shape, device=self.device)

        edge_index = torch.stack(torch.meshgrid(
            torch.arange(num_nodes, device=self.device),
            torch.arange(num_nodes, device=self.device),
            indexing='ij'
        ), dim=0).view(2, -1)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        condition_graph.central_mask = torch.zeros(condition_graph.num_nodes, dtype=torch.bool, device=self.device)
        sample_batch_map = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        condition_graph.batch = torch.zeros(condition_graph.num_nodes, dtype=torch.long, device=self.device)

        for i in reversed(range(0, self.num_timesteps)):
            timestep = torch.full((1,), i, device=self.device, dtype=torch.long)
            # During sampling, we predict all-to-all edges for the generated nodes
            row, col = torch.triu_indices(num_nodes, num_nodes, 1, device=self.device)
            edge_candidates = torch.stack([row, col], dim=0)
            predicted_noise, _ = self.denoising_model(
                atom_types=atom_types, pos=pos, edge_index=edge_index, timestep=timestep, batch_map=sample_batch_map, batch_obj=condition_graph,
                edge_candidates=edge_candidates
            )

            pred_atom_types_noise = predicted_noise[:, :num_atom_types]
            pred_pos_noise = predicted_noise[:, num_atom_types:]

            alpha_t = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[i]

            # Reverse process for atom types
            model_mean_atom_types = (1 / torch.sqrt(alpha_t)) * \
                (atom_types - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred_atom_types_noise)

            # Reverse process for pos
            model_mean_pos = (1 / torch.sqrt(alpha_t)) * \
                (pos - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred_pos_noise)

            if i == 0:
                atom_types = model_mean_atom_types
                pos = model_mean_pos
            else:
                variance = (1 - self.alphas_cumprod[i-1]) / (1 - self.alphas_cumprod[i]) * self.betas[i]
                noise_atom_types = torch.randn_like(atom_types)
                noise_pos = torch.randn_like(pos)
                atom_types = model_mean_atom_types + torch.sqrt(variance) * noise_atom_types
                pos = model_mean_pos + torch.sqrt(variance) * noise_pos

        final_timestep = torch.full((1,), 0, device=self.device, dtype=torch.long)
        row, col = torch.triu_indices(num_nodes, num_nodes, 1, device=self.device)
        edge_candidates = torch.stack([row, col], dim=0)
        _, final_edge_logits = self.denoising_model(
            atom_types, pos, edge_index, final_timestep, sample_batch_map, condition_graph,
            edge_candidates=edge_candidates
        )
        final_edges = (torch.sigmoid(final_edge_logits) > 0.5).nonzero().t()

        return atom_types, pos, final_edges
