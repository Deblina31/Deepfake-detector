from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Build a real simple CNN model
model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (real vs fake)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dummy training on random data (so it becomes valid)
import numpy as np
X_dummy = np.random.rand(10, 128, 128, 3)
y_dummy = np.random.randint(0, 2, size=(10, 1))
model.fit(X_dummy, y_dummy, epochs=1, verbose=1)

# Save it
os.makedirs("model", exist_ok=True)
model.save("model/cnn_model.h5")

print("✅ Fully valid dummy model saved at model/cnn_model.h5")
