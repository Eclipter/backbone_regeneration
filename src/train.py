import logging
import os.path as osp
import warnings
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import config
from dataset import DNADataModule
from model import Model
from utils import NoUnusedParametersWarningFilter, VisualizationCallback


def main():
    torch.set_float32_matmul_precision('high')

    # Suppress warnings
    logging.getLogger('torch.distributed.ddp_model_wrapper').addFilter(
        NoUnusedParametersWarningFilter()
    )
    warnings.filterwarnings('ignore', 'The `srun` command is available on your.*')

    data_module = DNADataModule(
        bond_threshold=config.BOND_THRESHOLD,
        batch_size=config.BATCH_SIZE
    )

    pl_module = Model(
        hidden_dim=config.HIDDEN_DIM,
        num_timesteps=config.NUM_TIMESTEPS,
        lr=config.LR,
        batch_size=config.BATCH_SIZE
    )

    log_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'logs')
    current_time = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    logger = TensorBoardLogger(log_dir, name='', version=current_time)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=10,
        save_last=True
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy()
    else:
        strategy = 'auto'
    trainer = pl.Trainer(
        strategy=strategy,
        log_every_n_steps=1,  # To disable warning
        # precision='16-mixed',
        max_epochs=-1,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            VisualizationCallback()
        ],
        enable_progress_bar=False,
        enable_model_summary=False
    )

    trainer.fit(pl_module, datamodule=data_module)

    # Test on a single GPU for accurate metrics
    if trainer.global_rank == 0:
        test_trainer = pl.Trainer(
            devices=1,
            logger=logger,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        test_trainer.test(
            pl_module,
            datamodule=data_module,
            ckpt_path=checkpoint_callback.best_model_path
        )


if __name__ == '__main__':
    main()
