from data.dataset import OxfordPetSegDataset
from model.unet import UNET
from utils.visualize import show_predictions
import torch
from config import DEVICE

model = UNET().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))

dataset = OxfordPetSegDataset('./data', train=False)
show_predictions(model, dataset, num=10, save=True, save_dir="results")