"""Structure export runners for baselines and the trained model."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from ..data import parse_dna
from ..inference import (
    _build_output_universe as build_output_universe,
    _predict_backbone_from_chain_records as predict_backbone_from_chain_records,
    write_structure,
)

if TYPE_CHECKING:
    from .knn_baseline import KnnBaselineState


class BackboneSampler(Protocol):
    def sample(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        ...


class FixedTorsionSampler:
    """Return constant torsion angles and tau_m for every graph in the batch."""

    def __init__(self, mean_theta: np.ndarray, mean_tau: float):
        self._mean_theta = torch.as_tensor(mean_theta, dtype=torch.float32)
        self._mean_tau = float(mean_tau)

    def sample(self, batch):
        device = batch.torsions.device
        batch_size = int(batch.num_graphs)
        theta = self._mean_theta.to(device).unsqueeze(0).expand(batch_size, -1)
        tau = torch.full(
            (batch_size,),
            self._mean_tau,
            device=device,
            dtype=torch.float32,
        )
        return theta, tau


def export_backbone_pdb(
    input_path: str | Path,
    output_path: str | Path,
    sampler: BackboneSampler,
    device: str,
    *,
    window_size: int,
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    try:
        _, chain_records = parse_dna(
            str(input_path),
            use_full_nucleotide=False,
            window_size=window_size,
        )
        predictions = predict_backbone_from_chain_records(
            [chain_records],
            sampler,
            device,
        )[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            _, full_chain_records = parse_dna(
                str(input_path),
                use_full_nucleotide=True,
                window_size=window_size,
            )
        write_structure(
            build_output_universe(full_chain_records, predictions),
            output_path,
        )
    except Exception as exc:
        return {
            'success': False,
            'wall_time_s': time.perf_counter() - t0,
            'returncode': -999,
            'output_pdb': None,
            'stdout': '',
            'stderr': repr(exc),
        }

    return {
        'success': True,
        'wall_time_s': time.perf_counter() - t0,
        'returncode': 0,
        'output_pdb': output_path,
        'stdout': '',
        'stderr': '',
    }


def run_mean_baseline_structure(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    mean_theta: np.ndarray,
    mean_tau: float,
    device: str,
    window_size: int,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = output_dir / f'{input_path.stem}_mean_baseline.pdb'
    return export_backbone_pdb(
        input_path,
        output_pdb,
        FixedTorsionSampler(mean_theta, mean_tau),
        device,
        window_size=window_size,
    )


def run_knn_baseline_structure(
    input_path: str | Path,
    output_dir: str | Path,
    knn_state: 'KnnBaselineState',
    *,
    window_size: int,
) -> dict[str, Any]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = output_dir / f'{input_path.stem}_knn_baseline.pdb'
    t0 = time.perf_counter()

    try:
        predictions = knn_state.predictions_for_pdb(input_path.stem)
        if not predictions:
            raise RuntimeError('no kNN predictions for structure')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            _, full_chain_records = parse_dna(
                str(input_path),
                use_full_nucleotide=True,
                window_size=window_size,
            )
        write_structure(
            build_output_universe(full_chain_records, predictions),
            output_pdb,
        )
    except Exception as exc:
        return {
            'success': False,
            'wall_time_s': time.perf_counter() - t0,
            'returncode': -999,
            'output_pdb': None,
            'stdout': '',
            'stderr': repr(exc),
        }

    return {
        'success': True,
        'wall_time_s': time.perf_counter() - t0,
        'returncode': 0,
        'output_pdb': output_pdb,
        'stdout': '',
        'stderr': '',
    }


