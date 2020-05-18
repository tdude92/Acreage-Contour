import os
import numpy as np

os.makedirs("./data/train/nlcd_labels")
os.makedirs("./data/train/land_cover_labels")
os.makedirs("./data/train/np_data")

os.makedirs("./data/test/nlcd_labels")
os.makedirs("./data/test/land_cover_labels")
os.makedirs("./data/test/np_data")

train_patch_path = "../ny_1m_2013/ny_1m_2013_extended-train_patches/"
val_patch_path = "../ny_1m_2013/ny_1m_2013_extended-val_patches/"

# Table that maps NLCD labels to a label between 0 - 21.
nlcd2nlcd_label = {11: 1, 12: 2, 21: 3, 22: 4, 23: 5, 24: 6, 31: 7,
                   41: 8, 42: 9, 43: 10, 51: 11, 52: 12, 71: 13, 72: 14,
                   73: 15, 74: 16, 81: 17, 82: 18, 90: 19, 95: 20, 255: 0, 0: 0}

ctr = 0
for file in os.listdir(train_patch_path):
    data = np.load(train_patch_path + file)
    data = data["arr_0"].squeeze()

    # Get data
    np_data = data[:3]
    land_cover_labels = data[8]
    nlcd_labels = data[9]

    # Normalize np_data to between -1 and 1.
    np_data = 2*(np_data/255) - 1

    # Fix labels.
    for i in range(len(land_cover_labels)):
        for j in range(len(land_cover_labels[i])):
            if land_cover_labels[i][j] == 15:
                land_cover_labels[i][j] = 6
            else:
                land_cover_labels[i][j] -= 1
    
    for i in range(len(nlcd_labels)):
        for j in range(len(nlcd_labels[i])):
            nlcd_labels[i][j] = nlcd2nlcd_label[nlcd_labels[i][j]]
    
    np.save("./data/train/np_data/" + str(ctr) + ".npy", np_data)
    np.save("./data/train/land_cover_labels/" + str(ctr) + ".npy", land_cover_labels)
    np.save("./data/train/nlcd_labels/" + str(ctr) + ".npy", nlcd_labels)
    
    print("[Train Set] " + file + " preprocessed (" + str(ctr) + ".npy).")
    ctr += 1


ctr = 0
for file in os.listdir(val_patch_path):
    data = np.load(val_patch_path + file)
    data = data["arr_0"].squeeze()

    # Get data
    np_data = data[:3]
    land_cover_labels = data[8]
    nlcd_labels = data[9]

    # Normalize np_data to between -1 and 1.
    np_data = 2*(np_data/255) - 1

    # Fix labels.
    for i in range(len(land_cover_labels)):
        for j in range(len(land_cover_labels[i])):
            if land_cover_labels[i][j] == 15:
                land_cover_labels[i][j] = 6
            else:
                land_cover_labels[i][j] -= 1
    
    for i in range(len(nlcd_labels)):
        for j in range(len(nlcd_labels[i])):
            nlcd_labels[i][j] = nlcd2nlcd_label[nlcd_labels[i][j]]
    
    np.save("./data/test/np_data/" + str(ctr) + ".npy", np_data)
    np.save("./data/test/land_cover_labels/" + str(ctr) + ".npy", land_cover_labels)
    np.save("./data/test/nlcd_labels/" + str(ctr) + ".npy", nlcd_labels)
    
    print("[Validation Set] " + file + " preprocessed (" + str(ctr) + ".npy).")
    ctr += 1