import torch
import torch.nn as nn
import numpy as np
import cv2
import random
import os

from model import UNet, LC2RGB

os.makedirs("out", exist_ok = True)

# Constants
MODEL_ID = "1"
CLIP_VALUE = 1
ON_CUDA  = torch.cuda.is_available()

N_EPOCHS = 500
BATCH_SIZE = 8
LEARN_RATE = 0.001

if ON_CUDA:
    print("GPU available. Training with CUDA.")
    DEVICE = "cuda:0"
else:
    print("GPU not available. Training with CPU.")
    DEVICE = "cpu"


# Load data.
data_loader = []
labels_loader = []
n_files = len(os.listdir("data/train/np_data"))
for file in range(n_files - n_files%BATCH_SIZE):
    data_loader.append(np.load("data/train/np_data/" + str(file) + ".npy"))
    labels_loader.append(np.load("data/train/land_cover_labels/" + str(file) + ".npy"))

# Group data into batches.
data_loader = torch.Tensor(np.stack(data_loader)).view(-1, BATCH_SIZE, 3, 256, 256)
labels_loader = torch.Tensor(np.stack(labels_loader)).type(torch.LongTensor).view(-1, BATCH_SIZE, 256, 256)
# print(torch.cuda.memory_allocated())
# Initialize model.
model = UNet()
try:
    model.load_state_dict(torch.load("model/" + MODEL_ID + ".pth"))
    print("Loaded " + MODEL_ID + ".pth")
except FileNotFoundError:
    torch.save(model.state_dict(), "model/" + MODEL_ID + ".pth")
    print("Created " + MODEL_ID + ".pth")

if ON_CUDA:
    model.cuda()

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARN_RATE)

for epoch in range(N_EPOCHS):
    avg_loss = 0

    for step in range(len(data_loader)):
        optimizer.zero_grad()
        images = data_loader[step]

        labels = labels_loader[step]
        if ON_CUDA:
            images = images.cuda()
            labels = labels.cuda()

        output = model.forward(images)
        train_loss = criterion(output, labels)

        avg_loss += train_loss.item()

        train_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
        optimizer.step()

        print("\rEpoch " + str(epoch) + "    Step " + str(step + 1) + "/" + str(len(data_loader)) + "    loss: " + str(round(avg_loss/(step+1), 4)), end = "" )
    print("\nEpoch " + str(epoch) + " Loss: " + str(avg_loss / len(data_loader)))
    print()
    torch.save(model.state_dict(), "model/" + MODEL_ID + ".pth")

    # Generate random sample.
    i = random.randint(0, len(data_loader) - 1)
    j = random.randint(0, BATCH_SIZE - 1)
    sample = data_loader[i][j].cuda()

    ground_truth = np.zeros((256, 256, 3))
    lc_labels = labels_loader[i][j]
    for i in range(len(lc_labels)):
        for j in range(len(lc_labels)):
            ground_truth[i][j] = LC2RGB[lc_labels[i][j]]

    output = model.generate(sample)

    sample = 255 * (cv2.cvtColor(sample.cpu().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR) + 1) /2
    output = cv2.cvtColor(output.astype("float32"), cv2.COLOR_RGB2BGR)
    ground_truth = cv2.cvtColor(ground_truth.astype("float32"), cv2.COLOR_RGB2BGR)

    os.makedirs("out/" + str(epoch), exist_ok = True)
    cv2.imwrite("out/" + str(epoch) + "/input.jpg", sample)
    cv2.imwrite("out/" + str(epoch) + "/output.jpg", output)
    cv2.imwrite("out/" + str(epoch) + "/ground_truth.jpg", ground_truth)
