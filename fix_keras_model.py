import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import json

# Generate Realistic Dummy Data (same as create_dummy_model.py)
np.random.seed(42)

# Generate 500 records of people who WILL repay
repays = np.random.normal(loc=[100000, 50000, 750, 0.5, 6], scale=[20000, 10000, 50, 0.2, 3], size=(500, 5))
y_repays = np.ones(500)

# Generate 500 records of people who will NOT repay (High Risk)
defaults = np.random.normal(loc=[30000, 200000, 500, 6.6, 2], scale=[10000, 50000, 80, 2, 1], size=(500, 5))
y_defaults = np.zeros(500)

X_data = np.vstack((repays, defaults))
y_data = np.concatenate((y_repays, y_defaults))

# Ensure no negative scaling artifacts
X_data[X_data < 0] = 1

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Keras Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='swish', input_shape=(5,)),
    tf.keras.layers.Dense(8, activation='swish'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y_data, epochs=50, batch_size=32, verbose=0)
model.save("loan_model.h5")
print("Saved 5-feature Keras model to loan_model.h5")
