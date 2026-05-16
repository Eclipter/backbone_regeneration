"""Sweep ODE sampler steps on the run's saved test split; report mean local RMSD."""

import os

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from base2backbone.runtime.progress import PROGRESS_BAR_COLOR
from base2backbone.runtime.run_artifacts import load_analysis_run_artifacts

RUN_ID = 'torsions/6/CLOSURE_LOSS_WEIGHT=0.001_CLOSURE_ANGLE_WEIGHT=0.1'
NUM_TIMESTEPS_LIST = [5, 10, 15, 30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def main() -> None:
    run_artifacts = load_analysis_run_artifacts(RUN_ID)
    model = run_artifacts.model
    device = run_artifacts.device
    # Held-out split persisted as `test_dataset.pt` during training (not the val loader).
    test_dataset = run_artifacts.test_dataset

    batch_size = int(model.hparams['batch_size'])
    num_workers = min(len(os.sched_getaffinity(0)), 16) if hasattr(os, 'sched_getaffinity') else 0
    loader_kwargs: dict = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': torch.device(device).type == 'cuda',
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['multiprocessing_context'] = 'spawn'
        loader_kwargs['prefetch_factor'] = 4

    loader = DataLoader(test_dataset, **loader_kwargs)

    for num_timesteps in NUM_TIMESTEPS_LIST:
        per_graph = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f'num_timesteps={num_timesteps}', colour=PROGRESS_BAR_COLOR):
                batch = batch.to(device)
                pred_theta, pred_tau_m = model.sample(
                    batch,
                    num_timesteps=num_timesteps,
                )
                rmsd = model._compute_rmsd_per_graph_local(pred_theta, pred_tau_m, batch)
                finite = torch.isfinite(rmsd)
                if finite.any():
                    per_graph.append(rmsd[finite].detach().cpu())
        if not per_graph:
            print(f'num_timesteps: {num_timesteps}, RMSD: no finite values')
            continue
        mean_rmsd = torch.cat(per_graph).mean().item()
        print(f'num_timesteps: {num_timesteps}, mean RMSD (Å): {mean_rmsd:.4f}')


if __name__ == '__main__':
    main()
