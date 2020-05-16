import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p = 0.5)
        self.pipeline = nn.Sequential(
            nn.Conv2d()
        )