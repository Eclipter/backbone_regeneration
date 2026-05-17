"""Layout invariants for the packed torsion latent and builder API."""

import inspect
import re
from pathlib import Path

import pytest
import torch

from base2backbone.torsion_constants import (
    N_LATENT,
    N_TORSIONS,
    N_TORSIONS_LATENT,
)
from base2backbone.geometry.backbone import build_backbone_from_torsions


_SRC = Path(__file__).resolve().parents[1] / 'src' / 'base2backbone'
_MODEL_PY = _SRC / 'model.py'


def test_torsion_counts():
    assert N_TORSIONS == 6
    assert N_LATENT == 7
    assert N_TORSIONS_LATENT == N_LATENT


def test_model_score_network_output_is_n_latent():
    text = _MODEL_PY.read_text()
    assert re.search(
        r'self\.out\s*=\s*nn\.Linear\(\s*hidden_dim\s*,\s*N_LATENT\s*\)',
        text,
    )


def test_model_source_uses_delta_torsion_layout():
    text = _MODEL_PY.read_text()
    assert 'TOR_DELTA' not in text
    assert 'N_TORSIONS' in text


def test_training_logs_named_loss_not_per_torsion_delta():
    text = _MODEL_PY.read_text()
    assert "'train/loss'" in text or '"train/loss"' in text
    assert "'diagnostics/train/closure_loss'" in text or '"diagnostics/train/closure_loss"' in text


def test_coordinate_builders_have_no_delta_parameter():
    sig = inspect.signature(build_backbone_from_torsions)
    assert 'delta' not in sig.parameters


@pytest.mark.parametrize(
    'restype', ('A', 'C', 'G', 'T'),
)
def test_backbone_torch_runs_with_nine_angles(restype):
    ri = torch.tensor([{'A': 0, 'C': 1, 'G': 2, 'T': 3}[restype]])
    theta = torch.zeros(1, N_TORSIONS)
    tau = torch.tensor([0.35])
    bb = build_backbone_from_torsions(theta, tau, ri, None)
    assert bb
    assert all(v.shape[0] == 1 for v in bb.values())
