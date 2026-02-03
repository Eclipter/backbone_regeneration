HIDDEN_DIM = 256
NUM_LAYERS = 3
NUM_TIMESTEPS = 200
BATCH_SIZE = 2**10
LR = 5e-4

# Leave only CKPT_PATH = None if you want to run a new experiment
CKPT_PATH = None
# CKPT_PATH = '/home/v_sidorov/backbone_regeneration/logs/overfit_dim512_red_on_plateau/checkpoints/last.ckpt'

# Set RUN_NAME if you want to have a custom name for the run instead of the current time. It will be used only if CKPT_PATH is None
RUN_NAME = 'test'

# TODO: consider adding StochasticWeightAveraging
# TODO: experiment with cosine beta schedule
# TODO: try L1 loss for training only
# TODO: align windows by nucleic acids and not by centroids
# TODO: order graph not to predict atom names
# TODO: consider ODE sampler over SDE one
# TODO: consider pairwise distance instead of RMSE
