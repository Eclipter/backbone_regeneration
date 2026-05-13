"""Export the score network from a Lightning checkpoint to ONNX."""

from argparse import ArgumentParser
import json
import os
import os.path as osp

import torch

from base2backbone.data import BASE_TO_INDEX
from base2backbone.model import N_TORSIONS_LATENT, BackboneLightningModule
from base2backbone.runtime import MODEL_DIR, find_best_checkpoint, resolve_run_dir

# Keys describing score-network I/O and sigma endpoints at inference time.
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
EXPORT_OPSET = 17


def export_to_onnx(ckpt_path: str, opset: int = EXPORT_OPSET):
    """Export the score network to ``MODEL_DIR`` (``model.onnx`` + ``model.json``)."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    pl_module = (
        BackboneLightningModule
        .load_from_checkpoint(ckpt_path, map_location='cpu')
        .float()
        .eval()
    )
    net = pl_module.score_network
    d_in = pl_module.node_dim
    x = torch.randn(1, ONNX_EXPORT_SEQ_LEN, d_in)

    onnx_path = osp.join(MODEL_DIR, 'model.onnx')
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
        'base_to_idx': BASE_TO_INDEX,
        'node_dim': d_in,
        'time_emb_dim': pl_module.time_emb_dim,
        'window_size': ONNX_EXPORT_SEQ_LEN,
        'opset_version': opset,
        'source_checkpoint': osp.basename(ckpt_path),
    }
    json_path = osp.join(MODEL_DIR, 'model.json')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return onnx_path, json_path


def main(run_id: str, opset: int = EXPORT_OPSET):
    run_dir = resolve_run_dir(run_id)
    ckpt_path = find_best_checkpoint(run_dir)
    print(f'Best checkpoint:         {ckpt_path}')
    onnx_path, json_path = export_to_onnx(ckpt_path, opset)
    print(f'Exported ONNX graph:     {onnx_path}')
    print(f'Exported companion JSON: {json_path}')


def _parse_args():
    parser = ArgumentParser(description='Export the best checkpoint of a training run to ONNX.')
    parser.add_argument(
        '--run-id',
        required=True,
        help='Experiment id relative to logs/ (e.g. "fixed_swa/baseline").',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args.run_id)
