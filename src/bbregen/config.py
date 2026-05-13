import math

# Base config: applied to every run. Change values here to affect all experiments at once
BASE = dict(
    HIDDEN_DIM=256,  # Must be divisible by NUM_HEADS
    NUM_HEADS=8,
    NUM_LAYERS=3,
    NUM_TIMESTEPS=200,
    BATCH_SIZE=20000,
    LR=1e-3,
    WEIGHT_DECAY=0.01,
    EDGE_WEIGHT=0.3,  # from 0 (for central-only), to 1 (for edge-only)
    LR_SCHEDULER=None,  # 'ReduceLROnPlateau' | None
    LR_SCHEDULER_PATIENCE=10,
    LR_SCHEDULER_COOLDOWN=5,
    LR_SCHEDULER_THRESHOLD=0.1,
    SWA=True,
    SWA_LR=0.1,
    SWA_EPOCH_START=80,
    NUM_EPOCHS=100,
    ANGULAR_SIGMA_MIN=0.01 * math.pi,
    ANGULAR_SIGMA_MAX=math.pi,
    TAU_SIGMA_MIN=0.01 * math.pi,
    TAU_SIGMA_MAX=math.pi,
    SCORE_LOSS_WEIGHTING='sigma2',
    TAU_LOSS_WEIGHT=1.0,
    CLOSURE_LOSS_WEIGHT=0.0,
    CLOSURE_BOND_WEIGHT=1.0,
    CLOSURE_ANGLE_WEIGHT=1.0,
    CLOSURE_TORSION_WEIGHT=1.0,
    LOG_CLOSURE_METRICS_TRAIN=False,
    LOG_CLOSURE_METRICS_VAL=True,
    TORCH_COMPILE=True,
    START_FROM_LAST_CKPT=True,
)

# One entry = one experiment. Put ONLY the overrides from BASE here
EXPERIMENTS = [
    {},  # baseline (matches BASE)
]

# Run path under `logs/`
RUN_NAME = 'torsions/1'

SEED = 42

######### TO DOs ########
# MAJOR:
# Consider bridge-matching

# MINOR:
# Decrease batch size
# Experiment with large window sizes
# Experiment with edge weight
# Distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC
########################
