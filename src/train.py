import logging
import os
import os.path as osp
import shutil
import warnings
from datetime import datetime

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (LearningRateMonitor, ModelCheckpoint,
                                         StochasticWeightAveraging)
from lightning.pytorch.loggers import TensorBoardLogger

from config import BASE, EXPERIMENTS, RUN_NAME, SEED
from dataset import DNADataModule
from model import PytorchLightningModule

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
    pl.seed_everything(SEED, workers=True)

    data_module = DNADataModule(batch_size=cfg['BATCH_SIZE'])
    pl_module = PytorchLightningModule(
        hidden_dim=cfg['HIDDEN_DIM'],
        num_layers=cfg['NUM_LAYERS'],
        num_timesteps=cfg['NUM_TIMESTEPS'],
        sampling_steps=cfg['SAMPLING_STEPS'],
        batch_size=cfg['BATCH_SIZE'],
        lr=cfg['LR'],
        lr_scheduler=cfg['LR_SCHEDULER'],
        lr_scheduler_patience=cfg['LR_SCHEDULER_PATIENCE'],
        lr_scheduler_threshold=cfg['LR_SCHEDULER_THRESHOLD'],
        beta_schedule=cfg['BETA_SCHEDULE'],
    )

    num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))

    # Initialize logger
    log_dir, run_name, run_version, ckpt_path = _get_run_paths(cfg)
    logger = TensorBoardLogger(log_dir, name=run_name, version=run_version, default_hp_metric=False)

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        # filename='{epoch}',
        monitor='val_rmse',
        every_n_epochs=5,
        save_top_k=10,
        save_last=True,
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
        precision='16-mixed',
        log_every_n_steps=-1,
        num_nodes=num_nodes,
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
        run_cfg['RUN_VERSION'] = _make_run_version(run_cfg, BASE)
        train_one(run_cfg)


if __name__ == '__main__':
    main()
