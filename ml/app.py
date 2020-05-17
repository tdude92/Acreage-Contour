# RUN LIKE THIS:
#     python3 app.py <FILE_NAME>

import torch
import cv2
import sys
import os
from model import UNet

MODEL_PATH = "model/1.pth"

file_name = sys.argv[1] # eg. something.jpg

model = UNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location = "cpu"))

image = cv2.imread("../backend/uploads/" + file_name)
image = cv2.resize(image, (256, 256))

image = cv2.resize(image, (256, 256))

image = cv2.resize(image, (256, 256))

image = torch.Tensor(2*(image/255) - 1).permute(2, 0, 1)
image = model.generate(image)

cv2.imwrite("../backend/outputs/" + file_name)
os.remove("../backend/uploads/" + file_name)
