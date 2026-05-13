"""Stable runtime loading helpers for analysis scripts."""

import os.path as osp
from dataclasses import dataclass
from glob import glob
from typing import Any

import torch

from ..model import BackboneLightningModule
from .paths import find_best_checkpoint, resolve_run_dir


@dataclass(frozen=True)
class AnalysisRunArtifacts:
    run_id: str
    run_dir: str
    ckpt_path: str
    test_dataset_path: str
    event_files: list[str]
    test_dataset: Any
    target_modes: tuple[str, ...]
    test_indices_per_mode: dict[str, list[int]]
    test_datasets: dict[str, torch.utils.data.Subset]
    model: BackboneLightningModule
    device: str


def load_analysis_run_artifacts(run_id: str) -> AnalysisRunArtifacts:
    run_dir = resolve_run_dir(run_id)
    ckpt_path = find_best_checkpoint(run_dir)
    test_dataset_path = osp.join(run_dir, 'test_dataset.pt')
    event_files = glob(osp.join(run_dir, 'events.*'))

    try:
        test_dataset = torch.load(test_dataset_path, weights_only=False)
    except FileNotFoundError:
        raise FileNotFoundError(f'`{test_dataset_path}` not found. Ensure training completed.')

    target_modes = ('all', 'central', 'edge')
    test_indices_per_mode = {
        'all': list(range(len(test_dataset))),
        'central': list(test_dataset.central_virtual),
        'edge': list(test_dataset.edge_virtual),
    }
    test_datasets = {
        mode: torch.utils.data.Subset(test_dataset, indices)
        for mode, indices in test_indices_per_mode.items()
    }

    try:
        model = BackboneLightningModule.load_from_checkpoint(
            ckpt_path,
            weights_only=False,
            map_location='cpu',
        ).eval()
    except FileNotFoundError:
        raise FileNotFoundError(f'Checkpoint `{ckpt_path}` not found.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    return AnalysisRunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        ckpt_path=ckpt_path,
        test_dataset_path=test_dataset_path,
        event_files=event_files,
        test_dataset=test_dataset,
        target_modes=target_modes,
        test_indices_per_mode=test_indices_per_mode,
        test_datasets=test_datasets,
        model=model,
        device=device,
    )
