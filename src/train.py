import logging
import os
import os.path as osp
import shutil
import subprocess
import time
import warnings
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import config
from dataset import DNADataModule
from model import PytorchLightningModule
from utils import NoUnusedParametersWarningFilter, VisualizationCallback


def main():
    torch.set_float32_matmul_precision('high')

    # Suppress warnings
    logging.getLogger('torch.distributed.ddp_model_wrapper').addFilter(
        NoUnusedParametersWarningFilter()
    )
    warnings.filterwarnings('ignore', 'The `srun` command is available on your.*')

    # Initialize lightning modules
    data_module = DNADataModule(batch_size=config.BATCH_SIZE)
    pl_module = PytorchLightningModule(
        hidden_dim=config.HIDDEN_DIM,
        num_timesteps=config.NUM_TIMESTEPS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR
    )

    # Initialize logger
    log_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', 'logs')
    # On rank 0, kill previous tensorboard and remove old logs
    if os.environ.get('LOCAL_RANK', '0') == '0':
        subprocess.run('pkill -f tensorboard', shell=True, stdout=subprocess.DEVNULL)
        time.sleep(1)
    if config.CKPT_PATH:
        config.RUN_NAME = None
    if config.RUN_NAME:
        run_path = osp.join(log_dir, config.RUN_NAME)
        if osp.exists(run_path):
            shutil.rmtree(run_path)
    if config.CKPT_PATH:
        run_name = config.CKPT_PATH.split('/')[5]
    elif config.RUN_NAME:
        run_name = config.RUN_NAME
    else:
        run_name = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    logger = TensorBoardLogger(log_dir, name='', version=run_name, default_hp_metric='test_combined_score')

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_combined_score',
        save_last=True
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_combined_score',
        patience=700,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        strategy='auto',
        gradient_clip_val=1,
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

    # Launch TensorBoard on rank 0
    if trainer.is_global_zero:
        subprocess.Popen(
            'tensorboard --logdir logs',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        print(f'\nTensorBoard is running at http://127.0.0.1:6006\n')

    # Train and test
    trainer.fit(pl_module, datamodule=data_module, ckpt_path=config.CKPT_PATH)
    trainer.test(datamodule=data_module, ckpt_path='best')


if __name__ == '__main__':
    main()
