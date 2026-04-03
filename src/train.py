import logging
import os
import os.path as osp
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


def resolve_run_name(log_dir, run_name):
    run_path = osp.join(log_dir, run_name)
    if not osp.exists(run_path):
        return run_name

    # Never delete an existing run directory during debug experiments.
    timestamp = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    return osp.join(run_name, timestamp)


def main():
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(config.SEED, workers=True)

    # A workaround for mlx4 compatibility issues
    os.environ.setdefault('NCCL_IB_DISABLE', '1')

    data_module = DNADataModule(
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        overfit_single_sample=config.OVERFIT_SINGLE_SAMPLE,
        overfit_sample_index=config.OVERFIT_SAMPLE_INDEX
    )
    pl_module = PytorchLightningModule(
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_timesteps=config.NUM_TIMESTEPS,
        batch_size=config.BATCH_SIZE,
        lr=config.LR,
        beta_schedule=config.BETA_SCHEDULE,
        train_loss_type=config.TRAIN_LOSS_TYPE,
        prediction_target=config.PREDICTION_TARGET,
        eps_directional_head=config.EPS_DIRECTIONAL_HEAD,
        eps_use_local_head=config.EPS_USE_LOCAL_HEAD,
        eps_normalize_agg=config.EPS_NORMALIZE_AGG,
        use_edge_attr=config.USE_EDGE_ATTR,
        use_torsion_features=config.USE_TORSION_FEATURES,
        train_t_max=config.TRAIN_T_MAX,
        debug_fixed_t=config.DEBUG_FIXED_T,
        debug_fixed_noise=config.DEBUG_FIXED_NOISE,
        debug_eval_t=config.DEBUG_EVAL_T,
        debug_eval_snr=config.DEBUG_EVAL_SNR,
        eval_full_sampling=config.EVAL_FULL_SAMPLING,
        val_full_sampling=config.VAL_FULL_SAMPLING,
        val_gen_every_n_epochs=config.VAL_GEN_EVERY_N_EPOCHS,
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
            run_name = resolve_run_name(log_dir, run_name)
    else:
        run_name = datetime.now().strftime('%Y.%m.%d_%H:%M:%S')
    pl_logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
    pl_logger.addFilter(lambda r: 'litlogger' not in r.getMessage())  # Mute LitLogger tip
    logger = TensorBoardLogger(log_dir, name='', version=run_name, default_hp_metric=False)
    use_single_device = config.OVERFIT_SINGLE_SAMPLE and config.DEBUG_SINGLE_DEVICE
    checkpoint_metric = 'val_rmse'
    if config.VAL_FULL_SAMPLING and config.VAL_GEN_EVERY_N_EPOCHS == 1:
        checkpoint_metric = 'val_gen_rmse'
    strategy = 'auto'
    if not use_single_device and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # The model contains mutually exclusive branches, so some parameters stay unused each step.
        strategy = 'ddp_find_unused_parameters_true'

    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_metric,
        save_last=True,
        enable_version_counter=False
    )
    early_stopping_callback = EarlyStopping(
        monitor=checkpoint_metric,
        patience=500,
    )
    swa = StochasticWeightAveraging(
        swa_lrs=0.1*config.LR,
        swa_epoch_start=300
    )

    # Initialize trainer
    trainer = pl.Trainer(
        gradient_clip_val=config.GRADIENT_CLIP_VAL,
        max_epochs=config.MAX_EPOCHS,
        overfit_batches=config.OVERFIT_BATCHES,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if use_single_device else 'auto',
        strategy=strategy,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            # early_stopping_callback,
            # swa,
            # VisualizationCallback()
        ],
        profiler=config.PROFILER,
        log_every_n_steps=1,
        enable_progress_bar=False,
        enable_model_summary=False
    )

    # Train and test
    trainer.fit(pl_module, datamodule=data_module, ckpt_path=ckpt_path, weights_only=False)
    trainer.test(datamodule=data_module, ckpt_path='best', weights_only=False)


if __name__ == '__main__':
    main()
