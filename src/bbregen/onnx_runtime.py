import json
import os.path as osp

import numpy as np
import onnxruntime as ort
import torch

from .data import BASE_TO_INDEX, N_CHAIN_END_CLASSES
from .torsion_constants import LOG_TAU_M_MAX, LOG_TAU_M_MIN, N_LATENT, N_TORSIONS
from .wrapped_score_diffusion import decode_torsions, reverse_ve_score_step, ve_sigma_grid, wrap_angle


class OnnxSampler:
    """Inference-only VE sampler backed by an exported ONNX denoiser."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_dir, self.onnx_path, self.meta_path = _resolve_model_artifacts(model_path)
        with open(self.meta_path) as f:
            meta = json.load(f)
        hp = dict(meta['hyperparameters'])

        self.hparams = hp
        self.node_dim = int(meta['node_dim'])
        self.time_emb_dim = int(meta['time_emb_dim'])
        self.window_size = int(meta['window_size'])
        self.device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')

        providers = ['CPUExecutionProvider']
        if self.device.type == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

    def _time_mlp(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.time_emb_dim // 2
        scale = np.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=time.device, dtype=time.dtype) * (-scale))
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

    def _b_ws(self, batch):
        b = batch.num_graphs
        n = batch.torsions.size(0)
        ws = n // b
        return b, ws

    def _build_x(self, batch, x_t_latent, log_sigma_per_graph, sc):
        b, ws = self._b_ws(batch)
        rel_o = batch.rel_origins.view(b, ws, 3)
        rel_r = batch.rel_frames.view(b, ws, 9)
        pair_o = batch.pair_rel_origins.view(b, ws, 3)
        pair_r = batch.pair_rel_frames.view(b, ws, 3, 3).reshape(b, ws, 9)
        base = batch.base_types.view(b, ws, len(BASE_TO_INDEX))
        hp = batch.has_pair_nt.view(b, ws, 1).float()
        ce = batch.chain_end_class.view(b, ws, N_CHAIN_END_CLASSES)
        it = batch.is_target_nt.view(b, ws, 1)
        tidx = batch.target_nt_idx.long()
        bi = torch.arange(b, device=rel_o.device)
        pad = torch.zeros(
            b,
            ws,
            N_LATENT + self.time_emb_dim + N_TORSIONS + N_LATENT,
            device=rel_o.device,
            dtype=rel_o.dtype,
        )
        te_all = self._time_mlp(log_sigma_per_graph.float()).to(dtype=rel_o.dtype)
        o = 0
        pad[bi, tidx, o:o + N_LATENT] = x_t_latent
        o += N_LATENT
        pad[bi, tidx, o:o + self.time_emb_dim] = te_all
        o += self.time_emb_dim
        pad[bi, tidx, o:o + N_TORSIONS] = batch.torsion_mask.view(b, ws, N_TORSIONS)[bi, tidx].float()
        o += N_TORSIONS
        pad[bi, tidx, o:o + N_LATENT] = sc
        return torch.cat([rel_o, rel_r, pair_o, pair_r, base, hp, ce, it, pad], dim=-1)

    def forward_denoiser(self, batch, x_t_latent, log_sigma_per_graph, sc):
        b, _ = self._b_ws(batch)
        x = self._build_x(batch, x_t_latent, log_sigma_per_graph, sc)
        score_all_np = self.session.run(
            ['score'],
            {'node_features': x.detach().cpu().numpy().astype(np.float32, copy=False)},
        )[0]
        score_all = torch.from_numpy(np.asarray(score_all_np)).to(device=x.device, dtype=x.dtype)
        bi = torch.arange(b, device=score_all.device)
        return score_all[bi, batch.target_nt_idx.long()]

    @torch.no_grad()
    def p_sample_loop(self, batch):
        batch = batch.to(self.device)
        b = batch.num_graphs
        dev = batch.torsions.device
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
            torch.rand(b, N_TORSIONS, device=dev, dtype=dtype) * (2.0 * np.pi) - np.pi,
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
                np.log(float(sigma_cur_th.clamp(min=1e-8))),
                device=dev,
                dtype=dtype,
            )
            pred = self.forward_denoiser(batch, x_t, log_s, sc)
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
                x_t = torch.nan_to_num(x_t, nan=0.0, posinf=np.pi, neginf=-np.pi)
                theta = wrap_angle(x_t[..., :N_TORSIONS])
                logt = x_t[..., N_TORSIONS:N_LATENT]
                x_t = torch.cat([theta, logt], dim=-1)
        return decode_torsions(x_t)

    @torch.no_grad()
    def sample(self, batch):
        return self.p_sample_loop(batch)


def _resolve_model_artifacts(model_path: str) -> tuple[str, str, str]:
    if osp.isdir(model_path):
        model_dir = model_path
        onnx_path = osp.join(model_dir, 'model.onnx')
        meta_path = osp.join(model_dir, 'model.json')
    else:
        model_dir = osp.dirname(model_path) or '.'
        if model_path.endswith('.onnx'):
            onnx_path = model_path
            meta_path = osp.join(model_dir, 'model.json')
        elif model_path.endswith('.json'):
            meta_path = model_path
            onnx_path = osp.join(model_dir, 'model.onnx')
        else:
            raise ValueError('model_path must be a directory, .onnx file, or .json file')
    if not osp.isfile(onnx_path):
        raise FileNotFoundError(f'ONNX model not found: {onnx_path}')
    if not osp.isfile(meta_path):
        raise FileNotFoundError(f'ONNX metadata not found: {meta_path}')
    return model_dir, onnx_path, meta_path
