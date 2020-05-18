import torch
import numpy as np
import cv2
from model import UNet, LC2RGB
import time

model = UNet()
model.load_state_dict(torch.load("model/dummy.pth"))

img = np.load("data/train/np_data/0.npy")
lc_labels = np.load("data/train/land_cover_labels/0.npy").astype("long")
output = model.generate(torch.Tensor(img))

img = (img + 1) / 2
output = output / 255
ground_truth = np.zeros((256, 256, 3))

for i in range(len(lc_labels)):
    for j in range(len(lc_labels)):
        ground_truth[i][j] = LC2RGB[lc_labels[i][j]]
ground_truth = ground_truth / 255

cv2.imshow("Input", cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
cv2.imshow("Output", cv2.cvtColor(output.astype("float32"), cv2.COLOR_RGB2BGR))
cv2.imshow("Ground Truth", cv2.cvtColor(ground_truth.astype("float32"), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)