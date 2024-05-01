import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
# CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 50
FEATURES_OF_DISCRIMINATOR = 64
FEATURES_OF_GENERATOR = 64