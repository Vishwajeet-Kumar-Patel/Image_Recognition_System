import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np

# Optional: Set memory growth for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load preprocessed data
with open("X.pickle", "rb") as f:
    X = pickle.load(f)

with open("Y.pickle", "rb") as f:
    y = pickle.load(f)

# Normalize image data for quicker processing
X = X / 255.0

# Convert labels to numpy array in case they're not
y = np.array(y)

# Build CNN model
model = Sequential()

# Convolutional Layer 1
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X, y,
    batch_size=50,
    epochs=10,
    validation_split=0.3
)
 
# Print summary
model.summary()

# Example prediction (use a real sample from X if available)
sample = X[0].reshape(1, *X.shape[1:])
prediction = model.predict(sample)
print("Prediction:", prediction)
