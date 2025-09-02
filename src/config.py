HIDDEN_DIM = 128
NUM_TIMESTEPS = 500
BATCH_SIZE = 2**10
LR = 1e-3

# Leave only CKPT_PATH = None if you want to run a new experiment
CKPT_PATH = None
# CKPT_PATH = '/home/v_sidorov/backbone_regeneration/logs/test/checkpoints/last.ckpt'

# Set RUN_NAME if you want to have a custom name for the run instead of the current time. It will be used only if CKPT_PATH is None
RUN_NAME = 'sequential_val/test'


# TODO: consider adding StochasticWeightAveraging
