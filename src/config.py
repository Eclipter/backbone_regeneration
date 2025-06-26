import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
HIDDEN_DIM = 64
LR = 1e-3
NUM_EPOCHS = 100
NUM_TIMESTEPS = 200

# TODO: consider adding these features to the model
SUPPORTED_EDGES = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
ATOMIC_NUMBERS = [6, 7, 8, 15, 16]
