import json
import os
import os.path as osp
from argparse import ArgumentParser

import torch

from model import N_TORSIONS_LATENT, PytorchLightningModule
from utils import base_to_idx, find_best_checkpoint, resolve_run_dir

# Keys describing denoiser I/O and σ endpoints at inference time (VE score sampler rebuilds grids).
HPARAM_KEYS = (
    'hidden_dim',
    'num_heads',
    'num_layers',
    'num_timesteps',
    'angular_sigma_min',
    'angular_sigma_max',
    'tau_sigma_min',
    'tau_sigma_max',
    'tau_loss_weight',
    'score_loss_weighting',
)

# Inference window (nucleotides). ``torch.onnx.export`` fixes this axis in the graph; only
# batch ``B`` is dynamic — do not declare ``L`` in ``dynamic_axes`` or consumers may assume any length.
ONNX_EXPORT_SEQ_LEN = 3


def export_to_onnx(ckpt_path, out_dir=None, opset=17):
    """Export the torsion Transformer denoiser from a Lightning checkpoint to ONNX.

    Writes ``model.onnx`` (single forward: node features → predicted score) and ``model.json``.
    Tensors are ``[B, ONNX_EXPORT_SEQ_LEN, node_dim]`` and ``[B, ONNX_EXPORT_SEQ_LEN, N_TORSIONS_LATENT]``;
    only the batch axis ``B`` is dynamic in the ONNX graph.
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
    net = pl_module.denoiser
    d_in = pl_module.node_dim
    x = torch.randn(1, ONNX_EXPORT_SEQ_LEN, d_in)

    onnx_path = osp.join(out_dir, 'model.onnx')
    torch.onnx.export(
        net,
        (x,),
        onnx_path,
        input_names=['node_features'],
        output_names=['score'],
        dynamic_axes={
            'node_features': {0: 'B'},
            'score': {0: 'B'},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # Everything the ONNX graph deliberately leaves out goes into model.json.
    hp = pl_module.hparams
    meta = {
        'hyperparameters': {k: getattr(hp, k) for k in HPARAM_KEYS},
        'N_TORSIONS_LATENT': N_TORSIONS_LATENT,
        'base_to_idx': base_to_idx,
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
