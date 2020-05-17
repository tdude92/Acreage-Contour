import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel_size = 3):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        return block

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p = 0.5)
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")

        self.conv1_down = self.conv_block(3, 64)
        self.conv2_down = self.conv_block(64, 128)
        self.conv3_down = self.conv_block(128, 256)

        self.conv_bottleneck = self.conv_block(256, 512)

        self.conv3_up1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        self.conv2_up1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.conv1_up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.conv3_up2 = self.conv_block(512, 256)
        self.conv2_up2 = self.conv_block(256, 128)
        self.conv1_up2 = self.conv_block(128, 64)

        self.output_block = nn.Sequential(
            nn.Conv2d(64, 21, kernel_size = 1, stride = 1),
            nn.LogSoftmax(dim = 0)
        )

    
    def forward(self, x):
        # Contraction
        # Tensor Dim: (3, 256, 256)

        conv1_down_out = self.conv1_down(x)
        # Tensor Dim: (64, 256, 256)

        x = self.max_pool(conv1_down_out)
        conv2_down_out = self.conv2_down(x)
        # Tensor Dim: (128, 128, 128)

        x = self.max_pool(conv2_down_out)
        conv3_down_out = self.conv3_down(x)
        # Tensor Dim: (256, 64, 64)

        x = self.max_pool(conv3_down_out)
        x = self.conv_bottleneck(x) # <-- Bottleneck
        # Tensor Dim: (512, 32, 32)

        # Expansion
        x = self.conv3_up1(x)
        x = self.upsample(x)
        x = x + conv3_down_out
        x = self.conv3_up2(x)
        # Tensor Dim: (256, 64, 64)

        x = self.conv2_up1(x)
        x = self.upsample(x)
        x = x + conv2_down_out
        x = self.conv2_up2(x)
        # Tensor Dim: (128, 128, 128)

        x = self.conv1_up1(x)
        x = self.upsample(x)
        x = x + conv1_down_out
        x = self.conv1_up2(x)
        # Tensor Dim: (64, 256, 256)

        x = self.output_block(x)
        # Tensor Dim: (21, 256, 256)

        return x
