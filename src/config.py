# Base config: applied to every run. Change values here to affect all experiments at once
BASE = dict(
    HIDDEN_DIM=256,
    NUM_LAYERS=5,
    NUM_TIMESTEPS=200,
    BATCH_SIZE=2**11,
    LR=1e-3,
    LR_SCHEDULER='ReduceLROnPlateau',  # set to None to disable
    LR_SCHEDULER_PATIENCE=10,
    LR_SCHEDULER_COOLDOWN=5,
    LR_SCHEDULER_THRESHOLD=0.1,
    SWA=False,
    SWA_LR=0.1,
    SWA_EPOCH_START=10,
    NUM_EPOCHS=50,
    BETA_SCHEDULE='cosine',  # 'linear' | 'cosine'
    START_FROM_LAST_CKPT=True,
)

# One entry = one experiment. Put ONLY the deltas from BASE here
EXPERIMENTS = [
    {},  # baseline (matches BASE)
    # {'LR_SCHEDULER': None, 'SWA': True}
    {'LR_SCHEDULER_PATIENCE': 5}
]

# Run path under `logs/`
RUN_NAME = 'fixed_equivariance'

SEED = 42


######### TO DOs ########
# MAJOR:
# Chemical losses
# DDIM sampler

# MINOR:
# v-prediction + self-conditioning
# Apply root to val/test RMSE on epoch end, after averaging over batches
# Consider ODE sampler over SDE one
# Diffuse inside latent space instead of the euclidean one
# Distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC
########################
