import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

# ✅ Update this path to point to your dataset folder
# Example: "C:/Users/YourName/Datasets/PetImages"
DATADIR = "C:/Users/vishw/Downloads/PetImages"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is not None:
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

create_training_data()
print(f"Total training samples: {len(training_data)}")

# Shuffle to avoid bias
random.shuffle(training_data)

# Split features and labels
X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

# Convert to NumPy arrays
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(Y)

# ✅ Save with pickle
with open("X.pickle", "wb") as f:
    pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("Y.pickle", "wb") as f:
    pickle.dump(Y, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Data serialized to X.pickle and Y.pickle")
