"""Stable runtime loading helpers for analysis scripts."""

import os.path as osp
from dataclasses import dataclass
from glob import glob

import torch

from ..model import BackboneLightningModule
from .paths import find_best_checkpoint, resolve_run_dir


@dataclass(frozen=True)
class AnalysisRunArtifacts:
    run_id: str
    run_dir: str
    ckpt_path: str
    event_files: list[str]
    target_modes: tuple[str, ...]
    model: BackboneLightningModule
    device: str


def load_analysis_run_artifacts(run_id: str) -> AnalysisRunArtifacts:
    run_dir = resolve_run_dir(run_id)
    ckpt_path = find_best_checkpoint(run_dir)
    event_files = glob(osp.join(run_dir, 'events.*'))

    target_modes = ('avg', 'central', 'edge')

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
        event_files=event_files,
        target_modes=target_modes,
        model=model,
        device=device,
    )
