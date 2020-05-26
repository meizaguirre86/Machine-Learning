import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import datetime

from tensorflow.python.keras.callbacks import TensorBoard

data = np.zeros((10000, 3))
print(data.shape)
for i in range(10000):
    data[i,0:2] = np.random.randint(low = 1, high= 1000, size=2)
    data[i,2] = np.sum(data[i,0:2], axis = 0)

data = data.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(data[:,0:2], data[:,2], test_size=0.3)


y_train = np.reshape(y_train, (7000,1))
y_test = np.reshape(y_test, (3000,1))

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(10, activation='relu', input_dim=2))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001), 'mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
log_dir = r"C:\Users\meiza\Python\logs\logs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard = TensorBoard(log_dir=log_dir)

model.fit(X_train, y_train, epochs=50, callbacks=[tensorboard])
