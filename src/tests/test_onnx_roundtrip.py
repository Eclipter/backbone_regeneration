"""Round-trip ONNX export tests: PyTorch TorsionDenoiser vs ONNX Runtime (CPU, fp32).

``export.export_to_onnx`` fixes the sequence axis (see ``export.ONNX_EXPORT_SEQ_LEN``); tests vary
batch ``B`` only. True variable-length windows require a different export path (e.g. dynamo).
"""

import io
import json
from typing import TypedDict

import lightning
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from export import ONNX_EXPORT_SEQ_LEN, export_to_onnx
from model import N_TORSIONS_LATENT, PytorchLightningModule, TorsionDenoiser


class _DummyHParams(TypedDict):
    hidden_dim: int
    num_heads: int
    num_layers: int
    num_timesteps: int
    batch_size: int
    lr: float
    lr_scheduler: str | None
    lr_scheduler_patience: int
    lr_scheduler_threshold: float
    lr_scheduler_cooldown: int
    angular_sigma_min: float
    angular_sigma_max: float
    tau_sigma_min: float
    tau_sigma_max: float
    score_loss_weighting: str
    tau_loss_weight: float
    weight_decay: float
    closure_loss_weight: float
    closure_bond_weight: float
    closure_angle_weight: float
    closure_torsion_weight: float
    log_closure_metrics_train: bool
    log_closure_metrics_val: bool


_DUMMY_HP: _DummyHParams = {
    'hidden_dim': 64,
    'num_heads': 4,
    'num_layers': 2,
    'num_timesteps': 10,
    'batch_size': 1,
    'lr': 1e-3,
    'lr_scheduler': None,
    'lr_scheduler_patience': 5,
    'lr_scheduler_threshold': 0.1,
    'lr_scheduler_cooldown': 2,
    'angular_sigma_min': 1e-4,
    'angular_sigma_max': 0.5,
    'tau_sigma_min': 1e-4,
    'tau_sigma_max': 0.5,
    'score_loss_weighting': 'sigma2',
    'tau_loss_weight': 1.0,
    'weight_decay': 0.01,
    'closure_loss_weight': 0.0,
    'closure_bond_weight': 1.0,
    'closure_angle_weight': 1.0,
    'closure_torsion_weight': 1.0,
    'log_closure_metrics_train': False,
    'log_closure_metrics_val': True,
}
_MODULE = PytorchLightningModule(**_DUMMY_HP).float().eval()
NODE_DIM = _MODULE.node_dim


def _export(net, dummy_input, dest):
    # Match export.export_to_onnx tracing kwargs (portable CPU fp32 graph).
    torch.onnx.export(
        net,
        (dummy_input,),
        dest,
        input_names=['node_features'],
        output_names=['score'],
        dynamic_axes={
            'node_features': {0: 'B'},
            'score': {0: 'B'},
        },
        opset_version=17,
        do_constant_folding=True,
    )


def _save_dummy_checkpoint(tmp_path, hparams: _DummyHParams):
    pl_module = PytorchLightningModule(**hparams).float()
    ckpt_path = tmp_path / 'dummy.ckpt'
    torch.save(
        {
            'state_dict': pl_module.state_dict(),
            'hyper_parameters': dict(pl_module.hparams),
            'pytorch-lightning_version': lightning.__version__,
        },
        ckpt_path,
    )
    return str(ckpt_path)


def _main_onnx_opset_version(model_proto):
    for oi in model_proto.opset_import:
        if oi.domain in ('', 'ai.onnx'):
            return oi.version
    raise AssertionError('no main ONNX opset in model_proto')


@pytest.fixture(scope='module')
def exported_onnx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('onnx_export')
    onnx_path = tmp / 'model.onnx'
    net = TorsionDenoiser(
        NODE_DIM,
        hidden_dim=_DUMMY_HP['hidden_dim'],
        num_heads=_DUMMY_HP['num_heads'],
        num_layers=_DUMMY_HP['num_layers'],
    ).float().eval()
    dummy_input = torch.randn(1, ONNX_EXPORT_SEQ_LEN, NODE_DIM)
    _export(net, dummy_input, str(onnx_path))
    return net, str(onnx_path)


def test_onnx_valid(exported_onnx):
    _, onnx_path = exported_onnx
    model_proto = onnx.load(onnx_path)
    onnx.checker.check_model(model_proto)
    assert _main_onnx_opset_version(model_proto) == 17

    assert len(model_proto.graph.input) == 1
    assert len(model_proto.graph.output) == 1
    inp = model_proto.graph.input[0]
    outp = model_proto.graph.output[0]
    assert inp.name == 'node_features'
    assert outp.name == 'score'

    assert inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert outp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    in_shape = inp.type.tensor_type.shape
    assert in_shape.dim[0].dim_param == 'B'
    assert in_shape.dim[1].dim_value == ONNX_EXPORT_SEQ_LEN
    assert in_shape.dim[2].dim_value == NODE_DIM
    out_shape = outp.type.tensor_type.shape
    assert out_shape.dim[0].dim_param == 'B'
    assert out_shape.dim[1].dim_value == ONNX_EXPORT_SEQ_LEN
    assert out_shape.dim[2].dim_value == N_TORSIONS_LATENT


# ORT: sequence length is fixed in the ONNX graph (``ONNX_EXPORT_SEQ_LEN``).
@pytest.mark.parametrize('B,L', [
    (1, ONNX_EXPORT_SEQ_LEN),
    (2, ONNX_EXPORT_SEQ_LEN),
    (4, ONNX_EXPORT_SEQ_LEN),
    (8, ONNX_EXPORT_SEQ_LEN),
])
def test_numeric_match(exported_onnx, B, L):
    net, onnx_path = exported_onnx
    torch.manual_seed(0)
    x = torch.randn(B, L, NODE_DIM)
    with torch.no_grad():
        pt_out = net(x).numpy()

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = np.asarray(sess.run(['score'], {'node_features': x.numpy()})[0], dtype=np.float32)
    np.testing.assert_allclose(pt_out, ort_out, rtol=1e-4, atol=1e-5)


def test_deterministic_export(exported_onnx):
    net, _ = exported_onnx
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    dummy = torch.randn(1, ONNX_EXPORT_SEQ_LEN, NODE_DIM)
    _export(net, dummy, buf1)
    _export(net, dummy, buf2)
    assert buf1.getvalue() == buf2.getvalue()


@pytest.mark.parametrize('B,L', [(1, ONNX_EXPORT_SEQ_LEN), (3, ONNX_EXPORT_SEQ_LEN)])
def test_output_shape(exported_onnx, B, L):
    _, onnx_path = exported_onnx
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    x = torch.zeros(B, L, NODE_DIM)
    out = np.asarray(sess.run(['score'], {'node_features': x.numpy()})[0], dtype=np.float32)
    assert out.shape == (B, L, N_TORSIONS_LATENT)


def test_export_uses_correct_node_dim(tmp_path):
    ckpt_path = _save_dummy_checkpoint(tmp_path, _DUMMY_HP)
    onnx_path, json_path = export_to_onnx(ckpt_path, out_dir=str(tmp_path))

    proto = onnx.load(onnx_path)
    input_shape = proto.graph.input[0].type.tensor_type.shape
    assert input_shape.dim[1].dim_value == ONNX_EXPORT_SEQ_LEN
    last_dim = input_shape.dim[2].dim_value
    assert last_dim == NODE_DIM

    with open(json_path) as f:
        meta = json.load(f)
    assert 'hyperparameters' in meta
    hp = meta['hyperparameters']
    assert hp['score_loss_weighting'] == 'sigma2'
    assert 'angular_sigma_min' in hp and 'tau_sigma_max' in hp and 'tau_loss_weight' in hp
