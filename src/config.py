# Base config: applied to every run. Change values here to affect all experiments at once
BASE = dict(
    HIDDEN_DIM=256,  # Must be divisible by NUM_HEADS
    NUM_HEADS=8,
    NUM_LAYERS=3,
    NUM_TIMESTEPS=100,
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
    SWA_EPOCH_START=40,
    NUM_EPOCHS=50,
    BETA_SCHEDULE='linear',  # 'linear' | 'cosine'
    START_FROM_LAST_CKPT=False,
)

# One entry = one experiment. Put ONLY the deltas from BASE here
EXPERIMENTS = [
    {},  # baseline (matches BASE)
    {'NUM_HEADS': 3, 'HIDDEN_DIM': 255},
    {'NUM_LAYERS': 5},
    # {'NUM_TIMESTEPS': 150}
]

# Run path under `logs/`
RUN_NAME = 'torsions'

SEED = 42

######### TO DOs ########
# MAJOR:
# Measure distribution of RMSD between different stochastic runs
# Compare with rigid molecular model fitting
# Include info about the complement chain
# Round-trip test ONNX export

# MINOR:
# Decrease batch size
# Experiment with large window sizes
# Experiment with edge weight
# Try DDIM
# Compile
# Distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC
########################
