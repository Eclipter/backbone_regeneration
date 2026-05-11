"""Smoke tests that val RMSD / predict paths stay on window builder and VE score layout."""

import inspect
from pathlib import Path

from model import PytorchLightningModule
from predict import inference_uses_window_builder


def test_val_rmsd_impl_calls_window_backbone_builder_only():
    src = inspect.getsource(PytorchLightningModule._compute_rmsd_per_graph_local)
    assert 'build_batch_window_backbone_from_torsions_torch' in src
    assert 'build_backbone_from_torsions_torch' not in src


def test_predict_default_uses_window_builder_flag():
    assert inference_uses_window_builder(False) is True
    assert inference_uses_window_builder(True) is False


def test_training_module_has_no_ddpm_alpha_bar_in_source():
    root = Path(__file__).resolve().parents[1]
    model_txt = (root / 'model.py').read_text()
    assert 'alpha_bar' not in model_txt
    assert 'alphas_cumulative_prod' not in model_txt


def test_wrapped_score_module_has_no_sincos_latent_width():
    wsd = Path(__file__).resolve().parents[1] / 'wrapped_score_diffusion.py'
    text = wsd.read_text()
    assert 'N_TORSIONS * 2' not in text
