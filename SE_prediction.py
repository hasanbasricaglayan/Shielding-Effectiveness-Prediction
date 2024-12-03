# Shielding Effectiveness (SE) Prediction with Deep Learning

import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Getting the data ready
data = np.transpose(np.loadtxt("data1000.txt"))

# Create input matrix
X = data[:, :8]  # jusqu'à la 8ème colonne incluse

# Create output matrix
Y = data[:, 8:]  # à partir de la 9ème colonne

# Plot a random SE curve
freq = range(0, 500)
curve = random.randrange(1, 1001)

plt.figure(dpi=1000)
plt.figure(1)
plt.plot(freq, Y[curve-1, :], "b-")
plt.xlabel("Frequency")
plt.ylabel("SE [dB]")
plt.title("Measured Shielding Effectiveness ($%s^{th}$ curve)" % curve)

# Scale input data
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X-X_mean)/X_std

# Split training data and validation data
X_train = X[:900, :]  # 90 %
X_val = X[900:, :]  # 10 %

Y_train = Y[:900, :]  # 90 %
Y_val = Y[900:, :]  # 10 %

# Design the network
model = Sequential()
model.add(Dense(128, input_dim=8, kernel_initializer="uniform", activation="selu"))
model.add(Dense(128, kernel_initializer="uniform", activation="selu"))
model.add(Dense(128, kernel_initializer="uniform", activation="selu"))
model.add(Dense(128, kernel_initializer="uniform", activation="selu"))
model.add(Dense(128, kernel_initializer="uniform", activation="selu"))
model.add(Dense(128, kernel_initializer="uniform", activation="selu"))
model.add(Dense(500, kernel_initializer="uniform"))
model.summary()

# Compile and fit the model
epo = 150
model.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])
hist = model.fit(X_train, Y_train, epochs=epo, batch_size=10, validation_data=(X_val, Y_val))
train_loss = hist.history["loss"]
valid_loss = hist.history["val_loss"]
ep = range(1, epo+1)

plt.figure(dpi=1000)
plt.figure(2)
plt.plot(ep, train_loss, "b-", label="Training error")
plt.plot(ep, valid_loss, "r-", label="Validation error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Training and validation losses")
plt.legend()

# Evaluate the model
Y_pred = model.predict(X_val)
curve_1 = random.randrange(1, 31)
curve_2 = random.randrange(31, 61)
curve_3 = random.randrange(61, 101)

plt.figure(dpi=1000)
plt.figure(3)
plt.plot(freq, Y_val[curve_1-1, :], "b-", label="Exact output")
plt.plot(freq, Y_pred[curve_1-1, :], "r-", label="Predicted output")
plt.xlabel("Frequency")
plt.ylabel("SE [dB]")
plt.title("Exact and Predicted Shielding Effectiveness ($%s^{th}$ curve)" % curve_1)
plt.legend()

plt.figure(dpi=1000)
plt.figure(4)
plt.plot(freq, Y_val[curve_2-1, :], "b-", label="Exact output")
plt.plot(freq, Y_pred[curve_2-1, :], "r-", label="Predicted output")
plt.xlabel("Frequency")
plt.ylabel("SE [dB]")
plt.title("Exact and Predicted Shielding Effectiveness ($%s^{th}$ curve)" % curve_2)
plt.legend()

plt.figure(dpi=1000)
plt.figure(5)
plt.plot(freq, Y_val[curve_3-1, :], "b-", label="Exact output")
plt.plot(freq, Y_pred[curve_3-1, :], "r-", label="Predicted output")
plt.xlabel("Frequency")
plt.ylabel("SE [dB]")
plt.title("Exact and Predicted Shielding Effectiveness ($%s^{th}$ curve)" % curve_3)
plt.legend()

# Show figures
plt.show()
