HIDDEN_DIM = 128
NUM_TIMESTEPS = 1000
BATCH_SIZE = 128
PATIENCE = 200

# Set CKPT_PATH = None if you want to run a new experiment
# CKPT_PATH = '/home/v_sidorov/backbone_regeneration/logs/mse_losses/checkpoints/last.ckpt'
CKPT_PATH = None

# Set RUN_NAME if you want to have a custom name for the run instead of the current time. It will be used only if CKPT_PATH is None
RUN_NAME = 'scheduler'


# TODO: check if precision alters steps on graph
# TODO: consider adding StochasticWeightAveraging
