import torch

BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")