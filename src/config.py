# Base config: applied to every run. Change values here to affect all experiments at once
BASE = dict(
    HIDDEN_DIM=256,
    NUM_LAYERS=3,
    NUM_TIMESTEPS=200,
    BATCH_SIZE=4000,
    LR=1e-3,
    WEIGHT_DECAY=0.01,
    EDGE_WEIGHT=0.3,  # from 0 (for central-only), to 1 (for edge-only)
    LR_SCHEDULER=None,  # 'ReduceLROnPlateau' | None
    LR_SCHEDULER_PATIENCE=10,
    LR_SCHEDULER_COOLDOWN=5,
    LR_SCHEDULER_THRESHOLD=0.1,
    SWA=True,
    SWA_LR=0.1,
    SWA_EPOCH_START=10,
    NUM_EPOCHS=20,
    BETA_SCHEDULE='linear',  # 'linear' | 'cosine'
    START_FROM_LAST_CKPT=True,
)

# One entry = one experiment. Put ONLY the deltas from BASE here
EXPERIMENTS = [
    {},  # baseline (matches BASE)
]

# Run path under `logs/`
RUN_NAME = 'non_equivariant_gnn'

SEED = 42

######### TO DOs ########
# MAJOR:
# Chemical losses
# Check chemicality

# MINOR:
# Self-conditioning
# Compile
# Distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC
########################
