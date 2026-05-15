from typing import Any, cast

import torch
from torch_geometric.data import Data

from base2backbone.dataset import WindowTargetDataset


class _DummyBase:
    def __init__(self, data):
        self.window_size = 3
        self._data = data

    def get(self, idx):  # noqa: ARG002
        return self._data


def _payload(ws: int, tidx: int, fill: float) -> dict[str, torch.Tensor]:
    is_target = torch.zeros(ws, dtype=torch.float32)
    is_target[tidx] = 1.0
    return {
        'target_nt_idx': torch.tensor(tidx, dtype=torch.long),
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
        'central': _payload(ws, 1, 3.0),
        'edge': _payload(ws, 0, 7.0),
    }
    ds = WindowTargetDataset(cast(Any, _DummyBase(data)), [0], [True])

    def _fail(*args, **kwargs):  # noqa: ARG001
        raise AssertionError('runtime payload build should not run when precomputed fields exist')

    monkeypatch.setattr('base2backbone.dataset._build_target_payload', _fail)

    sample = ds[1]

    assert sample.target_nt_idx.item() == 0
    assert torch.equal(sample.rel_origins, data._precomputed_target_payloads['edge']['rel_origins'])
    assert torch.equal(sample.pair_rel_frames, data._precomputed_target_payloads['edge']['pair_rel_frames'])
    assert '_precomputed_target_payloads' not in sample._store
    assert not hasattr(data, 'rel_origins')
