import logging
import os
import os.path as osp
import shutil
import warnings
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import TensorBoardLogger

import config
from dataset import DNADataModule
from model import PytorchLightningModule
from utils import VisualizationCallback

# Suppress PyTorch FutureWarning about functools.partial in DDP comm hooks (Python 3.13 compatibility)
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.distributed.algorithms.ddp_comm_hooks')


def main():
    torch.set_float32_matmul_precision('high')

    # A workaround for mlx4 compatibility issues
    os.environ.setdefault('NCCL_IB_DISABLE', '1')

    data_module = DNADataModule(batch_size=config.BATCH_SIZE)
    pl_module = PytorchLightningModule(
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_timesteps=config.NUM_TIMESTEPS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR
    )

    # Initialize logger
    log_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'logs')
    if config.CKPT_PATH:
        config.RUN_NAME = None
    if config.RUN_NAME:
        run_path = osp.join(log_dir, config.RUN_NAME)
        if osp.exists(run_path):
            shutil.rmtree(run_path, ignore_errors=True)
    if config.CKPT_PATH:
        run_name = config.CKPT_PATH.split('/')[5]
    elif config.RUN_NAME:
        run_name = config.RUN_NAME
    else:
        run_name = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    pl_logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
    pl_logger.addFilter(lambda r: 'litlogger' not in r.getMessage())  # Mute LitLogger tip
    logger = TensorBoardLogger(log_dir, name='', version=run_name, default_hp_metric=False)

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_rmse',
        save_last=True
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_rmse',
        patience=100,
    )
    swa = StochasticWeightAveraging(
        swa_lrs=0.1*config.LR,
        swa_epoch_start=50
    )

    # Initialize trainer
    trainer = pl.Trainer(
        strategy='auto',
        gradient_clip_val=1,
        max_epochs=-1,
        overfit_batches=1,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            swa,
            VisualizationCallback()
        ],
        enable_progress_bar=False,
        enable_model_summary=False
    )

    # Train and test
    trainer.fit(pl_module, datamodule=data_module, ckpt_path=config.CKPT_PATH)
    trainer.test(datamodule=data_module, ckpt_path='best')


if __name__ == '__main__':
    main()
