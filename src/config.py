# Base config: applied to every run. Change values here to affect all experiments at once
BASE = dict(
    HIDDEN_DIM=256,
    NUM_LAYERS=5,
    NUM_TIMESTEPS=200,
    BATCH_SIZE=2**12,
    LR=1e-3,
    LR_SCHEDULER='ReduceLROnPlateau',  # set to None to disable
    LR_SCHEDULER_THRESHOLD=0.1,
    SWA=False,
    SWA_LR=0.1,
    BETA_SCHEDULE='cosine',  # 'linear' | 'cosine'
    START_FROM_LAST_CKPT=True,
)

# Target-mode-specific overrides. Every experiment is run once per entry below
PER_MODE = {
    'central': {'LR_SCHEDULER_PATIENCE': 10, 'SWA_EPOCH_START': 30, 'NUM_EPOCHS': 50},
    'edge':    {'LR_SCHEDULER_PATIENCE': 20, 'SWA_EPOCH_START': 100, 'NUM_EPOCHS': 150},
}

# One entry = one experiment. Put ONLY the deltas from BASE here
# Each experiment is run for every target_mode from PER_MODE above
EXPERIMENTS = [
    {},  # baseline (matches BASE)
    {'SWA': True},
]

# Run path under `logs/`
RUN_NAME = 'structure_wise_split/{target_mode}'

SEED = 42


######### TO DOs ########
# MAJOR:
# Diffuse inside latent space instead of the euclidean one
# Distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC

# MINOR:
# Experiment with StochasticWeightAveraging
# Experiment with beta schedule
# Consider ODE sampler over SDE one
# Consider pairwise distance instead of RMSE
########################
