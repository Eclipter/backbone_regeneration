import numpy as np

from base2backbone.data import BACKBONE_ATOMS
from base2backbone.eval import local_backbone_rmsd
from base2backbone.runtime.tensorboard import collect_scalar_history, scalars_to_dataframe


class _Scalar:
    def __init__(self, step: int, value: float):
        self.step = step
        self.value = value


class _FakeAccumulator:
    def __init__(self, scalars_by_tag):
        self._scalars_by_tag = scalars_by_tag

    def Scalars(self, tag):
        return self._scalars_by_tag[tag]

    def Tags(self):
        return {'scalars': list(self._scalars_by_tag)}


def test_scalars_to_dataframe_preserves_step_value_pairs():
    ea = _FakeAccumulator({
        'val/rmsd/avg': [_Scalar(0, 1.5), _Scalar(1, 1.25)],
    })

    got = scalars_to_dataframe(ea, 'val/rmsd/avg')

    assert got.to_dict('records') == [
        {'epoch': 0, 'value': 1.5},
        {'epoch': 1, 'value': 1.25},
    ]


def test_collect_scalar_history_filters_and_labels(monkeypatch):
    accumulators = {
        'events.a': _FakeAccumulator({
            'train/loss': [_Scalar(0, 4.0)],
            'ignored': [_Scalar(0, 99.0)],
        }),
        'events.b': _FakeAccumulator({
            'val/rmsd/avg': [_Scalar(0, 2.0), _Scalar(1, 1.0)],
        }),
    }

    def _fake_loader(path):
        return accumulators[path]

    monkeypatch.setattr(
        'base2backbone.runtime.tensorboard.load_event_accumulator',
        _fake_loader,
    )

    got = collect_scalar_history(
        ['events.a', 'events.b'],
        {
            'train/loss': ('avg', 'train/loss'),
            'val/rmsd/avg': ('avg', 'val/rmsd'),
        },
    )

    assert got.to_dict('records') == [
        {'epoch': 0, 'mode': 'avg', 'metric': 'train/loss', 'value': 4.0},
        {'epoch': 0, 'mode': 'avg', 'metric': 'val/rmsd', 'value': 2.0},
        {'epoch': 1, 'mode': 'avg', 'metric': 'val/rmsd', 'value': 1.0},
    ]


def test_local_backbone_rmsd_is_permutation_invariant_for_phosphate_oxygens():
    n_atoms = len(BACKBONE_ATOMS)
    pred = np.full((n_atoms, 3), np.nan, dtype=np.float64)
    gt = np.full((n_atoms, 3), np.nan, dtype=np.float64)

    pred[BACKBONE_ATOMS.index('P')] = np.array([0.0, 0.0, 0.0])
    gt[BACKBONE_ATOMS.index('P')] = np.array([0.0, 0.0, 0.0])

    pred[BACKBONE_ATOMS.index('OP1')] = np.array([1.0, 0.0, 0.0])
    pred[BACKBONE_ATOMS.index('OP2')] = np.array([0.0, 1.0, 0.0])
    gt[BACKBONE_ATOMS.index('OP1')] = np.array([0.0, 1.0, 0.0])
    gt[BACKBONE_ATOMS.index('OP2')] = np.array([1.0, 0.0, 0.0])

    assert local_backbone_rmsd(pred, gt) == 0.0
