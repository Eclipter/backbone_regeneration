"""Smoke tests that val RMSD / predict paths stay on window builder and VE score layout."""

import inspect
from pathlib import Path

from base2backbone.model import BackboneLightningModule
import base2backbone.inference as pred_mod


def test_val_rmsd_impl_calls_window_backbone_builder_only():
    src = inspect.getsource(BackboneLightningModule._compute_rmsd_per_graph_local)
    assert 'build_batch_window_backbone_from_torsions' in src
    assert 'build_backbone_from_torsions' not in src


def test_val_rmsd_metric_name():
    src = inspect.getsource(BackboneLightningModule._log_rmsd)
    assert "f'{prefix}_rmsd'" in src
    ck = Path(__file__).resolve().parents[1] / 'scripts' / 'train.py'
    assert "monitor='val_rmsd'" in ck.read_text()
    sched = inspect.getsource(BackboneLightningModule.configure_optimizers)
    assert "'monitor': 'val_rmsd'" in sched


def test_predict_merges_target_residue_only_from_cached_window():
    text = Path(pred_mod.__file__).read_text()
    assert '_merge_window_pred_for_residue' in text
    assert 'predictions.update(cached' not in text


def test_predict_has_full_window_inference_for_default_path():
    text = Path(pred_mod.__file__).read_text()
    assert '_predict_full_window_predictions_dict' in text


def test_predict_module_full_window_builder():
    src = inspect.getsource(pred_mod._predict_full_window_predictions_dict)
    assert 'build_batch_window_backbone_from_torsions' in src


def test_training_module_has_no_ddpm_alpha_bar_in_source():
    root = Path(__file__).resolve().parents[1] / 'src' / 'base2backbone'
    model_txt = (root / 'model.py').read_text()
    assert 'alpha_bar' not in model_txt
    assert 'alphas_cumulative_prod' not in model_txt


def test_wrapped_score_module_has_no_sincos_latent_width():
    wsd = Path(__file__).resolve().parents[1] / 'src' / 'base2backbone' / 'score_diffusion.py'
    text = wsd.read_text()
    assert 'N_TORSIONS * 2' not in text
