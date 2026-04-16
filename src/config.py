#### Hyperparameters ####
HIDDEN_DIM = 256
NUM_LAYERS = 5
NUM_TIMESTEPS = 200
BATCH_SIZE = 2**11
LR = 1e-3
#########################


##### Run parameters ####
SEED = 42

NUM_EPOCHS = 100

# Set RUN_NAME if you want to have a custom name for the run instead of the current time
# To run in a folder, set RUN_NAME to "folder/run_name"
RUN_NAME = 'window_size_5/reproduce'
# RUN_NAME = 'test'
RUN_VERSION = f'{HIDDEN_DIM}_{NUM_LAYERS}_{NUM_TIMESTEPS}_{LR}'

START_FROM_LAST_CKPT = True
#########################


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
