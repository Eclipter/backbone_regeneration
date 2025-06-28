import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


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

# A GNN for the denoising model, now with MessagePassing


class DenoisingGNN(nn.Module):
    def __init__(self, node_dim, hidden_dim=128):
        super().__init__()
        self.time_emb = TimestepEmbedding(hidden_dim)
        self.node_emb = nn.Linear(node_dim, hidden_dim)

        self.condition_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.node_out = nn.Linear(hidden_dim, node_dim)
        self.edge_out = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index, timestep, batch_map, batch_obj):
        time_emb = self.time_emb(timestep)
        node_emb = self.node_emb(x)

        # Autoregressive-friendly conditioning:
        # Condition on the full left nucleotide, and the bases of the central and right nucleotides.
        # This simulates generating a backbone from left to right.
        is_left = batch_obj.nucleotide_mask == 0
        is_central = batch_obj.central_mask
        is_right = batch_obj.nucleotide_mask == 2
        is_base = ~batch_obj.backbone_mask

        is_condition = is_left | (is_central & is_base) | (is_right & is_base)

        condition_nodes_x_features = batch_obj.x[is_condition]
        condition_nodes_pos = batch_obj.pos[is_condition]
        condition_nodes_x = torch.cat([condition_nodes_x_features, condition_nodes_pos], dim=1)
        condition_nodes_batch_map = batch_obj.batch[is_condition]

        num_graphs = batch_obj.num_graphs if hasattr(batch_obj, 'num_graphs') else 1

        if condition_nodes_x.size(0) == 0:
            pooled_condition = torch.zeros(num_graphs, self.node_emb.out_features, device=x.device)
        else:
            condition_emb = self.node_emb(condition_nodes_x)
            pooled_condition = global_mean_pool(condition_emb, condition_nodes_batch_map, size=num_graphs)

        time_h = self.time_proj(time_emb)
        condition_h = self.condition_proj(pooled_condition)

        # Correctly use batch_map which corresponds to x
        node_emb = node_emb + time_h[batch_map] + condition_h[batch_map]

        h = self.conv1(node_emb, edge_index).relu()
        h = self.conv2(h, edge_index).relu()

        node_noise_pred = self.node_out(h)

        num_nodes = x.size(0)
        row, col = torch.triu_indices(num_nodes, num_nodes, 1, device=h.device)

        edge_h = torch.cat([h[row], h[col]], dim=1)
        edge_logits = self.edge_out(edge_h).squeeze(-1)

        return node_noise_pred, edge_logits


class Model(nn.Module):
    def __init__(self, node_dim, hidden_dim=128, num_timesteps=1000, **kwargs):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.denoising_model = DenoisingGNN(node_dim, hidden_dim)

        betas = self.cosine_schedule(num_timesteps)
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

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timestep][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timestep][:, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, batch, timestep):
        """Calculates the loss for the reverse process using masks."""
        # Combine features and positions for the diffusion process
        x_start = torch.cat([batch.x, batch.pos], dim=1)

        node_noise = torch.randn_like(x_start)
        noisy_x = self.q_sample(x_start, timestep[batch.batch], noise=node_noise)

        # The denoising model now receives the concatenated tensor
        predicted_node_noise, predicted_edge_logits = self.denoising_model(
            x=noisy_x, edge_index=batch.edge_index, timestep=timestep, batch_map=batch.batch, batch_obj=batch
        )

        is_target_node = batch.backbone_mask & batch.central_mask
        if is_target_node.sum() == 0:
            return None

        # Loss is calculated on the concatenated features and positions
        node_loss = nn.MSELoss()(predicted_node_noise[is_target_node], node_noise[is_target_node])

        num_nodes_total = batch.num_nodes
        row, col = torch.triu_indices(num_nodes_total, num_nodes_total, 1, device=batch.x.device)

        # Create ground truth adjacency matrix for the batched graph
        gt_adj_sparse = torch.sparse_coo_tensor(
            batch.edge_index,
            torch.ones(batch.edge_index.size(1), device=batch.x.device),
            size=(num_nodes_total, num_nodes_total)
        )
        gt_adj_flat_triu = gt_adj_sparse.to_dense()[row, col]

        is_target_edge_mask = is_target_node[row] & is_target_node[col]

        if is_target_edge_mask.sum() == 0:
            return node_loss

        edge_loss = nn.BCEWithLogitsLoss()(
            predicted_edge_logits[is_target_edge_mask],
            gt_adj_flat_triu[is_target_edge_mask]
        )

        return node_loss + edge_loss

    def forward(self, batch):
        """
        A full forward pass for training.
        `batch` is the full batched graph from the dataloader.
        """
        batch_size = batch.num_graphs
        timestep = torch.randint(0, self.num_timesteps, (batch_size,), device=batch.x.device).long()

        # This will be changed later to use masks
        # For now, we keep the old logic to make the change incremental
        # We will need to split the batch here for the old p_losses to work

        # Placeholder for the next refactoring step
        # This will currently fail, but it's an intermediate step

        return self.p_losses(batch, timestep)

    @torch.no_grad()
    def sample(self, condition_graph, num_nodes):
        node_dim = self.denoising_model.node_emb.in_features
        shape = (num_nodes, node_dim)
        device = next(self.parameters()).device

        nodes_and_pos = torch.randn(shape, device=device)
        edge_index = torch.stack(torch.meshgrid(
            torch.arange(num_nodes, device=device),
            torch.arange(num_nodes, device=device),
            indexing='ij'
        ), dim=0).view(2, -1)
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        condition_graph.central_mask = torch.zeros(condition_graph.num_nodes, dtype=torch.bool, device=device)
        sample_batch_map = torch.zeros(num_nodes, dtype=torch.long, device=device)
        condition_graph.batch = torch.zeros(condition_graph.num_nodes, dtype=torch.long, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            timestep = torch.full((1,), i, device=device, dtype=torch.long)
            predicted_node_noise, _ = self.denoising_model(
                x=nodes_and_pos, edge_index=edge_index, timestep=timestep, batch_map=sample_batch_map, batch_obj=condition_graph
            )
            alpha_t = self.alphas[i].to(device)
            alpha_cumprod_t = self.alphas_cumprod[i].to(device)
            model_mean = (1 / torch.sqrt(alpha_t)) * (nodes_and_pos - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_node_noise)
            if i == 0:
                nodes_and_pos = model_mean
            else:
                variance = (1 - self.alphas_cumprod[i-1]) / (1 - self.alphas_cumprod[i]) * self.betas[i]
                noise = torch.randn_like(nodes_and_pos)
                nodes_and_pos = model_mean + torch.sqrt(variance) * noise

        final_timestep = torch.full((1,), 0, device=device, dtype=torch.long)
        _, final_edge_logits = self.denoising_model(nodes_and_pos, edge_index, final_timestep, sample_batch_map, condition_graph)
        final_edges = (torch.sigmoid(final_edge_logits) > 0.5).nonzero().t()

        num_features = self.denoising_model.node_emb.in_features - 3
        final_nodes = nodes_and_pos[:, :num_features]
        final_pos = nodes_and_pos[:, num_features:]

        return final_nodes, final_pos, final_edges
