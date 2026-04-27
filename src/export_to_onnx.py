import json
import os
import os.path as osp
from argparse import ArgumentParser

import torch

from model import PytorchLightningModule
from utils import atom_to_idx, base_to_idx, find_best_checkpoint

# Diffusion schedule buffers registered in PytorchLightningModule.__init__.
# Exported JSON-side so that a consumer of model.onnx can reproduce sampling
# without re-deriving them from (beta_schedule, num_timesteps)
SCHEDULE_BUFFERS = (
    'betas',
    'alphas_cumprod',
    'alphas_cumprod_prev',
    'sqrt_alphas_cumprod',
    'sqrt_one_minus_alphas_cumprod',
    'posterior_variance',
    'posterior_log_variance_clipped',
    'posterior_mean_coef1',
    'posterior_mean_coef2',
)

# Keys from self.hparams that describe what the denoiser expects and how to
# wrap it into reverse-diffusion sampling at inference time
HPARAM_KEYS = ('hidden_dim', 'num_layers', 'num_timesteps', 'beta_schedule')


def resolve_run_dir(run):
    """Map a user-facing experiment id (e.g. ``fixed_swa/baseline``) to its log directory.

    A redundant leading ``logs/`` is stripped so both ``fixed_swa/...`` and
    ``logs/fixed_swa/...`` resolve to the same place.
    """
    if osp.isabs(run):
        return run
    log_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'logs')
    run_norm = osp.normpath(run)
    if run_norm.split(os.sep, 1)[0] == 'logs':
        run_norm = run_norm.split(os.sep, 1)[1] if os.sep in run_norm else ''
    return osp.normpath(osp.join(log_dir, run_norm))


def export_to_onnx(ckpt_path, out_dir=None, opset=17):
    """Export the EGNN denoiser from a Lightning checkpoint to ONNX.

    Writes two files into ``out_dir`` (defaults to the checkpoint directory):
        - model.onnx: a single denoising step, i.e. EGNNDiff(h, x, edge_index) -> (h, x, eps).
        - model.json: hyperparameters + schedule buffers + atom/base vocabularies.

    The full reverse diffusion loop is *not* exported: consumers run the loop
    themselves using buffers from model.json.
    """
    if out_dir is None:
        out_dir = osp.dirname(osp.abspath(ckpt_path))
    os.makedirs(out_dir, exist_ok=True)

    # Force fp32/CPU for a portable graph; training may have used 16-mixed.
    pl_module = (
        PytorchLightningModule
        .load_from_checkpoint(ckpt_path, map_location='cpu')
        .float()
        .eval()
    )
    gnn = pl_module.gnn

    # Dummy inputs exercise dynamic axes N (nodes) and E (edges).
    # All edge endpoints are valid node indices so the trace is well-formed.
    in_node_nf = gnn.embedding_in.in_features
    num_nodes, num_edges = 64, 256
    h = torch.randn(num_nodes, in_node_nf)
    x = torch.randn(num_nodes, 3)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

    onnx_path = osp.join(out_dir, 'model.onnx')
    torch.onnx.export(
        gnn,
        (h, x, edge_index),
        onnx_path,
        input_names=['h', 'x', 'edge_index'],
        output_names=['h_out', 'x_out', 'eps'],
        dynamic_axes={
            'h': {0: 'N'},
            'x': {0: 'N'},
            'edge_index': {1: 'E'},
            'h_out': {0: 'N'},
            'x_out': {0: 'N'},
            'eps': {0: 'N'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # Everything the ONNX graph deliberately leaves out goes into model.json.
    hp = pl_module.hparams
    schedule = {
        name: getattr(pl_module, name).detach().cpu().tolist()
        for name in SCHEDULE_BUFFERS
    }
    meta = {
        'hyperparameters': {k: getattr(hp, k) for k in HPARAM_KEYS},
        'atom_to_idx': atom_to_idx,
        'base_to_idx': base_to_idx,
        'schedule_buffers': schedule,
        'opset_version': opset,
        'source_checkpoint': osp.basename(ckpt_path),
    }
    json_path = osp.join(out_dir, 'model.json')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return onnx_path, json_path


MODEL_DIR = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..', 'model'))


def _parse_args():
    p = ArgumentParser(description='Export the diffusion denoiser GNN from Lightning runs to ONNX.')
    p.add_argument(
        '--run-dir',
        required=True,
        help='Experiment id relative to logs/ (e.g. "fixed_swa/baseline").',
    )
    p.add_argument('--opset', type=int, default=17, help='ONNX opset version (>=16 required for ScatterND).')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    ckpt_path = find_best_checkpoint(run_dir)
    print(f'Best checkpoint:         {ckpt_path}')
    onnx_path, json_path = export_to_onnx(ckpt_path, MODEL_DIR, args.opset)
    print(f'Exported ONNX graph:     {onnx_path}')
    print(f'Exported companion JSON: {json_path}')
