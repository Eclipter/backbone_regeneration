HIDDEN_DIM = 256
NUM_LAYERS = 5
NUM_TIMESTEPS = 200
BATCH_SIZE = 2**4
LR = 1e-5

# Set RUN_NAME if you want to have a custom name for the run instead of the current time
# To run in a folder, set RUN_NAME to "folder/run_name"
# RUN_NAME = f'overfit1batch_{HIDDEN_DIM}_{NUM_LAYERS}_{NUM_TIMESTEPS}_{LR}'
RUN_NAME = 'test'

START_FROM_LAST_CKPT = False


# MAJOR:
# TODO: diffuse inside latent space instead of the euclidean one
# TODO: distillate like in paper: https://openreview.net/forum?id=8NuN5UzXLC

# MINOR:
# TODO: experiment with StochasticWeightAveraging
# TODO: experiment with beta schedule
# TODO: try L1 loss for training only
# TODO: consider ODE sampler over SDE one
# TODO: consider pairwise distance instead of RMSE
