import os


def _get_env(name, default, cast):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return cast(raw_value)


def _get_env_bool(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _get_env_optional_int(name, default):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == '':
        return default
    if raw_value.strip().lower() == 'none':
        return None
    return int(raw_value)


def _get_env_optional_float(name, default):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == '':
        return default
    if raw_value.strip().lower() == 'none':
        return None
    return float(raw_value)


HIDDEN_DIM = _get_env('HIDDEN_DIM', 256, int)
NUM_LAYERS = _get_env('NUM_LAYERS', 5, int)
NUM_TIMESTEPS = _get_env('NUM_TIMESTEPS', 200, int)
BATCH_SIZE = _get_env('BATCH_SIZE', 1, int)  # 2**11
LR = _get_env('LR', 1e-4, float)
SEED = _get_env('SEED', 0, int)
MAX_EPOCHS = _get_env('MAX_EPOCHS', 5000, int)
GRADIENT_CLIP_VAL = _get_env('GRADIENT_CLIP_VAL', 1.0, float)
OVERFIT_BATCHES = _get_env('OVERFIT_BATCHES', 1, int)
BETA_SCHEDULE = os.getenv('BETA_SCHEDULE', 'linear')
TRAIN_LOSS_TYPE = os.getenv('TRAIN_LOSS_TYPE', 'mse')
PREDICTION_TARGET = os.getenv('PREDICTION_TARGET', 'epsilon')
EPS_DIRECTIONAL_HEAD = _get_env_bool('EPS_DIRECTIONAL_HEAD', False)
EPS_USE_LOCAL_HEAD = _get_env_bool('EPS_USE_LOCAL_HEAD', True)
EPS_NORMALIZE_AGG = _get_env_bool('EPS_NORMALIZE_AGG', False)
USE_EDGE_ATTR = _get_env_bool('USE_EDGE_ATTR', False)

# Deterministic one-sample debug mode for diffusion sanity checks.
OVERFIT_SINGLE_SAMPLE = _get_env_bool('OVERFIT_SINGLE_SAMPLE', True)
OVERFIT_SAMPLE_INDEX = _get_env('OVERFIT_SAMPLE_INDEX', 0, int)
DEBUG_FIXED_T = _get_env_optional_int('DEBUG_FIXED_T', 100)
DEBUG_FIXED_NOISE = _get_env_bool('DEBUG_FIXED_NOISE', True)
DEBUG_EVAL_T = _get_env_optional_int('DEBUG_EVAL_T', 100)
# SNR-anchored eval: pick the t with alpha_bar closest to this value (overrides DEBUG_EVAL_T).
# E.g. DEBUG_EVAL_SNR=0.5 → t at which SNR = 0 (signal = noise).
DEBUG_EVAL_SNR = _get_env_optional_float('DEBUG_EVAL_SNR', None)
DEBUG_SINGLE_DEVICE = _get_env_bool('DEBUG_SINGLE_DEVICE', True)
TRAIN_T_MAX = _get_env_optional_int('TRAIN_T_MAX', None)
# Run full T-step reverse diffusion in test_step and log test_gen_rmse (slow).
EVAL_FULL_SAMPLING = _get_env_bool('EVAL_FULL_SAMPLING', False)
VAL_FULL_SAMPLING = _get_env_bool('VAL_FULL_SAMPLING', False)
VAL_GEN_EVERY_N_EPOCHS = _get_env('VAL_GEN_EVERY_N_EPOCHS', 1, int)
USE_TORSION_FEATURES = _get_env_bool('USE_TORSION_FEATURES', False)

# Set RUN_NAME if you want to have a custom name for the run instead of the current time
# To run in a folder, set RUN_NAME to "folder/run_name"
RUN_NAME = os.getenv('RUN_NAME', f'debug/overfit1graph_det__{HIDDEN_DIM}_{NUM_LAYERS}_{NUM_TIMESTEPS}_{LR}')
# RUN_NAME = f'fixed/overfit1graph_fixed__{HIDDEN_DIM}_{NUM_LAYERS}_{NUM_TIMESTEPS}_{LR}'

START_FROM_LAST_CKPT = _get_env_bool('START_FROM_LAST_CKPT', False)

# None | 'simple' | 'advanced'
PROFILER = os.getenv('PROFILER')

# MAJOR:
# TODO: diffuse inside latent space instead of the euclidean one
# TODO: distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC

# MINOR:
# TODO: experiment with StochasticWeightAveraging
# TODO: experiment with beta schedule
# TODO: try L1 loss for training only
# TODO: consider ODE sampler over SDE one
# TODO: consider pairwise distance instead of RMSE
