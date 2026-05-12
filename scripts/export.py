"""Export the diffusion denoiser from a Lightning checkpoint to ONNX."""

import json
import os
import os.path as osp
from argparse import ArgumentParser

import torch

from bbregen.model import N_TORSIONS_LATENT, PytorchLightningModule
from bbregen.utils import base_to_idx, find_best_checkpoint, resolve_run_dir

# Keys describing denoiser I/O and sigma endpoints at inference time.
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
    'log_tau_init_noise_scale',
)

# The exported graph keeps the nucleotide window length fixed.
ONNX_EXPORT_SEQ_LEN = 3

MODEL_DIR = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..', 'model'))


def export_to_onnx(ckpt_path, out_dir=None, opset=17):
    """Export the torsion Transformer denoiser from a Lightning checkpoint to ONNX."""
    if out_dir is None:
        out_dir = osp.dirname(osp.abspath(ckpt_path))
    os.makedirs(out_dir, exist_ok=True)

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

    hp = pl_module.hparams

    def _hparam(key):
        if hasattr(hp, key):
            return getattr(hp, key)
        return hp.get(key)

    meta = {
        'hyperparameters': {k: _hparam(k) for k in HPARAM_KEYS},
        'N_TORSIONS_LATENT': N_TORSIONS_LATENT,
        'base_to_idx': base_to_idx,
        'node_dim': d_in,
        'time_emb_dim': pl_module.time_emb_dim,
        'window_size': ONNX_EXPORT_SEQ_LEN,
        'opset_version': opset,
        'source_checkpoint': osp.basename(ckpt_path),
    }
    json_path = osp.join(out_dir, 'model.json')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return onnx_path, json_path


def _parse_args():
    parser = ArgumentParser(description='Export the diffusion denoiser GNN from Lightning runs to ONNX.')
    parser.add_argument(
        '--run-dir',
        required=True,
        help='Experiment id relative to logs/ (e.g. "fixed_swa/baseline").',
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version (>=16 required for ScatterND).',
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    ckpt_path = find_best_checkpoint(run_dir)
    print(f'Best checkpoint:         {ckpt_path}')
    onnx_path, json_path = export_to_onnx(ckpt_path, MODEL_DIR, args.opset)
    print(f'Exported ONNX graph:     {onnx_path}')
    print(f'Exported companion JSON: {json_path}')


if __name__ == '__main__':
    main()
