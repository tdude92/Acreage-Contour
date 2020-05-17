import torch
import torch.nn as nn
import cv2

from model import UNet

# Constants
MODEL_ID = "0"
ON_CUDA  = torch.cuda.is_available()

N_EPOCHS = 200
BATCH_SIZE = 16
LEARN_RATE = 0.01

if ON_CUDA:
    print("GPU available. Training with CUDA.")
    device = "cuda:0"
else:
    print("GPU not available. Training with CPU.")
    device = "cpu"
