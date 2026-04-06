import logging
import os
import os.path as osp
import shutil
import warnings
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import config
from dataset import DNADataModule
from model import PytorchLightningModule

# Suppress PyTorch FutureWarning about functools.partial in DDP comm hooks (Python 3.13 compatibility)
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.distributed.algorithms.ddp_comm_hooks')


def main():
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(config.SEED, workers=True)

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
    ckpt_path = None
    if config.RUN_NAME:
        run_name = config.RUN_NAME
        if config.START_FROM_LAST_CKPT:
            candidate = osp.join(log_dir, run_name, 'checkpoints', 'last.ckpt')
            ckpt_path = candidate if osp.isfile(candidate) else None
        else:
            run_path = osp.join(log_dir, run_name)
            shutil.rmtree(run_path, ignore_errors=True)
    else:
        run_name = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    pl_logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
    pl_logger.addFilter(lambda r: 'litlogger' not in r.getMessage())  # Mute LitLogger tip
    logger = TensorBoardLogger(log_dir, name='', version=run_name, default_hp_metric=False)

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_rmse',
        every_n_epochs=5,
        save_top_k=10,
        save_last=True,
        enable_version_counter=False
    )
    swa = StochasticWeightAveraging(
        swa_lrs=0.1*config.LR,
        swa_epoch_start=100
    )

    strategy = (
        DDPStrategy(find_unused_parameters=True)
        if torch.cuda.device_count() > 1
        else 'auto'
    )

    num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.NUM_EPOCHS,
        gradient_clip_val=1,
        log_every_n_steps=-1,
        num_nodes=num_nodes,
        strategy=strategy,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            # swa,
        ],
        profiler=config.PROFILER,
        enable_progress_bar=False,
        enable_model_summary=False
    )

    # Train and test
    trainer.fit(pl_module, datamodule=data_module, ckpt_path=ckpt_path, weights_only=False)
    trainer.test(datamodule=data_module, ckpt_path='best', weights_only=False)


if __name__ == '__main__':
    main()
