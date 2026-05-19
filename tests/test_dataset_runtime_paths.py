from typing import Any, cast

import torch
from torch_geometric.data import Data

from base2backbone.data import BACKBONE_ATOMS
from base2backbone.dataset import WindowTargetDataset
from base2backbone.eval.knn_baseline import dataset_samples_cached


class _DummyBase:
    def __init__(self, data):
        self.window_size = 3
        self._data = data
        self.data_list = ['/tmp/1abc/0.pt']

    def get(self, idx):  # noqa: ARG002
        return self._data


def _payload(ws: int, tidx: int, fill: float, segid: str, resid: int) -> dict[str, Any]:
    is_target = torch.zeros(ws, dtype=torch.float32)
    is_target[tidx] = 1.0
    return {
        'target_nt_idx': torch.tensor(tidx, dtype=torch.long),
        'target_segid': segid,
        'target_resid': resid,
        'is_target_nt': is_target,
        'rel_origins': torch.full((ws, 3), fill, dtype=torch.float32),
        'rel_frames': torch.full((ws, 3, 3), fill, dtype=torch.float32),
        'pair_rel_origins': torch.full((ws, 3), fill + 1.0, dtype=torch.float32),
        'pair_rel_frames': torch.full((ws, 3, 3), fill + 1.0, dtype=torch.float32),
    }


def test_window_target_dataset_uses_precomputed_payloads(monkeypatch):
    ws = 3
    data = Data(
        nt_origins_world=torch.zeros(ws, 3, dtype=torch.float32),
        nt_frames_world=torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(ws, 3, 3).clone(),
        pair_origins_world=torch.zeros(ws, 3, dtype=torch.float32),
        pair_frames_world=torch.zeros(ws, 3, 3, dtype=torch.float32),
        is_chain_edge_nt=torch.tensor([True, False, False], dtype=torch.bool),
    )
    data._precomputed_target_payloads = {
        'central': _payload(ws, 1, 3.0, 'A', 11),
        'edge': _payload(ws, 0, 7.0, 'B', 17),
    }
    ds = WindowTargetDataset(cast(Any, _DummyBase(data)), [0], [True])

    def _fail(*args, **kwargs):  # noqa: ARG001
        raise AssertionError('runtime payload build should not run when precomputed fields exist')

    monkeypatch.setattr('base2backbone.dataset._build_target_payload', _fail)

    sample = ds[1]

    assert sample.target_nt_idx.item() == 0
    assert sample.target_segid == 'B'
    assert sample.target_resid == 17
    assert torch.equal(sample.rel_origins, data._precomputed_target_payloads['edge']['rel_origins'])
    assert torch.equal(sample.pair_rel_frames, data._precomputed_target_payloads['edge']['pair_rel_frames'])
    assert '_precomputed_target_payloads' not in sample._store
    assert not hasattr(data, 'rel_origins')


def test_dataset_samples_cached_uses_precomputed_target_residue_ids():
    ws = 3
    n_bb = len(BACKBONE_ATOMS)
    data = Data(
        bb_xyz_world=torch.zeros(ws, n_bb, 3, dtype=torch.float32),
        nt_origins_world=torch.zeros(ws, 3, dtype=torch.float32),
        nt_frames_world=torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(ws, 3, 3).clone(),
        pair_origins_world=torch.zeros(ws, 3, dtype=torch.float32),
        pair_frames_world=torch.zeros(ws, 3, 3, dtype=torch.float32),
        is_chain_edge_nt=torch.tensor([True, False, False], dtype=torch.bool),
        base_types=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        has_pair_nt=torch.tensor([True, False, True], dtype=torch.bool),
        chain_end_class=torch.zeros(ws, 3, dtype=torch.float32),
    )
    data._precomputed_target_payloads = {
        'central': _payload(ws, 1, 3.0, 'A', 11),
        'edge': _payload(ws, 0, 7.0, 'B', 17),
    }
    ds = WindowTargetDataset(cast(Any, _DummyBase(data)), [0], [True])

    _, _, metas, _ = dataset_samples_cached(
        ds,
        'train',
        {
            '1abc': {
                'deposit_group_id': 'group-1',
                'dna_sequence_tokens': frozenset({'ACG'}),
            },
        },
    )

    assert [meta['segid'] for meta in metas] == ['A', 'B']
    assert [meta['resid'] for meta in metas] == [11, 17]
