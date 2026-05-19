from math import pi

# Base config: applied to every run. Change values here to affect all experiments at once
BASE = dict(
    HIDDEN_DIM=256,  # Must be divisible by NUM_HEADS
    NUM_HEADS=8,
    NUM_LAYERS=3,
    NUM_TIMESTEPS=200,
    BATCH_SIZE=50000,
    LR=1e-3,
    WEIGHT_DECAY=0.01,
    EDGE_WEIGHT=0.3,  # from 0 (for central-only), to 1 (for edge-only)
    LR_SCHEDULER=None,  # 'ReduceLROnPlateau' | None
    LR_SCHEDULER_PATIENCE=10,
    LR_SCHEDULER_COOLDOWN=5,
    LR_SCHEDULER_THRESHOLD=0.1,
    SWA=True,
    SWA_LR=0.1,
    SWA_EPOCH_START=100,
    NUM_EPOCHS=200,
    ANGULAR_SIGMA_MIN=0.01*pi,  # from 0 (no noise), to ANGULAR_SIGMA_MAX
    ANGULAR_SIGMA_MAX=pi,  # from ANGULAR_SIGMA_MIN, to pi (wrapped angles ~ uniform)
    TAU_SIGMA_MIN=0.01,  # from 0 (no noise), to TAU_SIGMA_MAX
    TAU_SIGMA_MAX=1.5,  # from TAU_SIGMA_MIN, to ~3.4 (= log(TAU_M_MAX/TAU_M_MIN))
    SCORE_LOSS_WEIGHTING='sigma2',  # 'sigma2' | 'none'
    TAU_LOSS_WEIGHT=1,  # [0, +inf)
    CLOSURE_LOSS_WEIGHT=1e-3,  # [0, +inf)
    CLOSURE_BOND_WEIGHT=1,  # [0, +inf)
    CLOSURE_ANGLE_WEIGHT=0.3,  # [0, +inf)
    CLOSURE_TORSION_WEIGHT=1,  # [0, +inf)
    # Closure σ (normalization for squared deviations); fail thresholds in bridge_closure.py are in σ-units.
    CLOSURE_SIGMA_BOND_A=0.05,  # (0, +inf) Å
    CLOSURE_SIGMA_ANGLE_DEG=10,  # (0, +inf) deg
    CLOSURE_SIGMA_TORSION_RAD=0.35,  # (0, +inf) rad
    RUN_NAME='torsions/11',  # Run path under `logs/`
    DATASET_MANIFEST='thesis_2026-05-17.json',  # Set to None to fetch from RCSB PDB API and save into latest.json
    SEED=42,
    TORCH_COMPILE=True,
    START_FROM_LAST_CKPT=True,
)

# One entry = one experiment. Put ONLY the overrides from BASE here
EXPERIMENTS = [
    # {},  # baseline (matches BASE)
    {'EDGE_WEIGHT': 0.5},
]

######### TO DOs ########
# Consider bridge-matching
########################
