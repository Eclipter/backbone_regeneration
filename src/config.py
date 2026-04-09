#### Hyperparameters ####
HIDDEN_DIM = 256
NUM_LAYERS = 5
NUM_TIMESTEPS = 200
BATCH_SIZE = 2**11
LR = 1e-3
#########################


## Training parameters ##
SEED = 42

NUM_EPOCHS = 200

# Set RUN_NAME if you want to have a custom name for the run instead of the current time
# To run in a folder, set RUN_NAME to "folder/run_name"
RUN_NAME = f'different_timesteps_mse/{HIDDEN_DIM}_{NUM_LAYERS}_{NUM_TIMESTEPS}_{LR}'
# RUN_NAME = 'test'

START_FROM_LAST_CKPT = False

# None | 'simple' | 'advanced'
PROFILER = 'simple'
#########################


######### TO DOs ########
# MAJOR:
# Try 16-mixed precision
# Diffuse inside latent space instead of the euclidean one
# Distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC

# MINOR:
# Experiment with StochasticWeightAveraging
# Experiment with beta schedule
# Try L1 loss for training only
# Consider ODE sampler over SDE one
# Consider pairwise distance instead of RMSE
########################
