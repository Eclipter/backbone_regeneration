import logging
import os
import os.path as osp
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightning.pytorch as pl
import torch
from config import BASE, EXPERIMENTS, RUN_NAME, SEED
from lightning.pytorch.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         StochasticWeightAveraging)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning_utilities.core.rank_zero import rank_zero_info

from bbregen.dataset import DNADataModule
from bbregen.model import PytorchLightningModule

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# Suppress PyTorch FutureWarning about functools.partial in DDP comm hooks (Python 3.13 compatibility)
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.distributed.algorithms.ddp_comm_hooks')


def _make_run_version(cfg, baseline):
    # Short, unique version: only the fields that differ from the shared baseline.
    diff = {k: cfg[k] for k in cfg if k in baseline and cfg[k] != baseline[k]}
    return '_'.join(f'{k}={v}' for k, v in diff.items()) or 'baseline'


def _get_run_paths(cfg):
    log_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'logs')

    run_version = cfg['RUN_VERSION']
    if cfg['RUN_NAME']:
        run_name = cfg['RUN_NAME']
        if cfg['START_FROM_LAST_CKPT']:
            candidate = osp.join(log_dir, run_name, run_version, 'checkpoints', 'last.ckpt')
            ckpt_path = candidate if osp.isfile(candidate) else None
        else:
            shutil.rmtree(osp.join(log_dir, run_name, run_version), ignore_errors=True)
            ckpt_path = None
    else:
        run_name = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
        ckpt_path = None

    return log_dir, run_name, run_version, ckpt_path


def train_one(cfg):
    pl.seed_everything(cfg['SEED'], workers=True, verbose=False)

    data_module = DNADataModule(
        batch_size=cfg['BATCH_SIZE'],
        edge_weight=cfg['EDGE_WEIGHT'],
        seed=cfg['SEED'],
    )
    pl_module = PytorchLightningModule(
        hidden_dim=cfg['HIDDEN_DIM'],
        num_heads=cfg['NUM_HEADS'],
        num_layers=cfg['NUM_LAYERS'],
        num_timesteps=cfg['NUM_TIMESTEPS'],
        batch_size=cfg['BATCH_SIZE'],
        lr=cfg['LR'],
        lr_scheduler=cfg['LR_SCHEDULER'],
        lr_scheduler_patience=cfg['LR_SCHEDULER_PATIENCE'],
        lr_scheduler_cooldown=cfg['LR_SCHEDULER_COOLDOWN'],
        lr_scheduler_threshold=cfg['LR_SCHEDULER_THRESHOLD'],
        angular_sigma_min=cfg['ANGULAR_SIGMA_MIN'],
        angular_sigma_max=cfg['ANGULAR_SIGMA_MAX'],
        tau_sigma_min=cfg['TAU_SIGMA_MIN'],
        tau_sigma_max=cfg['TAU_SIGMA_MAX'],
        score_loss_weighting=cfg['SCORE_LOSS_WEIGHTING'],
        tau_loss_weight=cfg['TAU_LOSS_WEIGHT'],
        weight_decay=cfg['WEIGHT_DECAY'],
        closure_loss_weight=cfg['CLOSURE_LOSS_WEIGHT'],
        closure_bond_weight=cfg['CLOSURE_BOND_WEIGHT'],
        closure_angle_weight=cfg['CLOSURE_ANGLE_WEIGHT'],
        closure_torsion_weight=cfg['CLOSURE_TORSION_WEIGHT'],
        log_closure_metrics_train=cfg['LOG_CLOSURE_METRICS_TRAIN'],
        log_closure_metrics_val=cfg['LOG_CLOSURE_METRICS_VAL'],
    )

    log_dir, run_name, run_version, ckpt_path = _get_run_paths(cfg)

    # Compile only when not resuming: torch.compile changes param keys (denoiser._orig_mod.*) and
    # breaks Lightning's checkpoint load_order (eager ckpt vs compiled module).
    if cfg['TORCH_COMPILE']:
        if torch.cuda.is_available():
            if ckpt_path is None:
                rank_zero_info(
                    'torch.compile: wrapping denoiser (first *_step will JIT-compile; exclude from timings)'
                )
                # OptimizedModule is not TorsionDenoiser; runtime API matches nn.Module.forward.
                pl_module.denoiser = torch.compile(pl_module.denoiser)  # type: ignore[assignment]
            else:
                rank_zero_info(
                    'TORCH_COMPILE set but resuming from checkpoint: training denoiser eager '
                    '(compile-before-load mismatches state_dict; compile-after-load needs pre-DDP hook).'
                )
        else:
            rank_zero_info('TORCH_COMPILE is True but CUDA is unavailable; skipping torch.compile')

    num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))

    # Initialize logger
    logger = TensorBoardLogger(log_dir, name=run_name, version=run_version, default_hp_metric=False)

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}',
        monitor='val_rmsd',
        every_n_epochs=2,
        save_top_k=10,
        save_last=True,
        mode='min',
        enable_version_counter=False
    )
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    if cfg['SWA']:
        swa = StochasticWeightAveraging(
            swa_lrs=cfg['SWA_LR']*cfg['LR'],
            swa_epoch_start=cfg['SWA_EPOCH_START']
        )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg['NUM_EPOCHS'],
        gradient_clip_val=1,
        num_nodes=num_nodes,
        devices=([0, 1] if os.uname().nodename.partition('.')[0] == 'node07' else 'auto'),  # TODO: remove
        logger=logger,
        callbacks=[
            checkpoint_callback,
            lr_callback,
            *([swa] if cfg['SWA'] else [])  # type: ignore
        ],
        enable_model_summary=False
    )

    # Train and test
    trainer.fit(pl_module, datamodule=data_module, ckpt_path=ckpt_path, weights_only=False)
    trainer.test(pl_module, datamodule=data_module, ckpt_path='best', weights_only=False)


def main():
    torch.set_float32_matmul_precision('high')

    # Mute litmodels / litlogger advertisements from Lightning
    logging.getLogger('lightning.pytorch.utilities.rank_zero').addFilter(
        lambda r: not any(s in r.getMessage().lower() for s in ('litmodels', 'litlogger'))
    )

    for exp in EXPERIMENTS:
        run_cfg = {**BASE, **exp}
        run_cfg['RUN_NAME'] = RUN_NAME
        run_cfg['SEED'] = SEED
        run_cfg['RUN_VERSION'] = _make_run_version(run_cfg, BASE)
        rank_zero_info(f'\n\033[1;38;5;93mRunning experiment: {run_cfg["RUN_VERSION"]}\033[0m')
        train_one(run_cfg)


if __name__ == '__main__':
    main()
