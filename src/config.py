# Base config: applied to every run. Change values here to affect all experiments at once
BASE = dict(
    HIDDEN_DIM=256,
    NUM_LAYERS=8,
    NUM_TIMESTEPS=200,
    BATCH_SIZE=2500,
    LR=1e-3,
    WEIGHT_DECAY=0.05,
    LR_SCHEDULER=None,  # 'ReduceLROnPlateau' | None
    LR_SCHEDULER_PATIENCE=10,
    LR_SCHEDULER_COOLDOWN=5,
    LR_SCHEDULER_THRESHOLD=0.1,
    SWA=True,
    SWA_LR=0.1,
    SWA_EPOCH_START=17,
    NUM_EPOCHS=30,
    BETA_SCHEDULE='linear',  # 'linear' | 'cosine'
    START_FROM_LAST_CKPT=True,
)

# One entry = one experiment. Put ONLY the deltas from BASE here
EXPERIMENTS = [
    # {},  # baseline (matches BASE)
    # {'NUM_TIMESTEPS': 250},
    # {'NUM_TIMESTEPS': 300},
    {'NUM_TIMESTEPS': 400},
    {'NUM_TIMESTEPS': 500}
]

# Run path under `logs/`
RUN_NAME = 'fixed_phosphorus'

SEED = 42

######### TO DOs ########
# MAJOR:
# Chemical losses

# MINOR:
# v-prediction + self-conditioning
# Diffuse inside latent space instead of the euclidean one
# Distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC
########################
