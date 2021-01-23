import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('normout.csv')


x = data[['Time','Source','Destination','Length','CIP','CIP CM','CIP I/O','ENIP','GVCP','NBNS','SSDP','TCP']].copy()
y = data[['Safe']].copy()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)#, random_state = 0)

model = keras.Sequential([
    keras.layers.Dense(500, activation=tf.nn.relu),
    keras.layers.Dense(500, activation=tf.nn.softmax),
    keras.layers.Dense(500, activation=tf.nn.softmax),
    keras.layers.Dense(500, activation=tf.nn.softmax),
    keras.layers.Dense(500, activation=tf.nn.softmax),
    keras.layers.Dense(100, activation=tf.nn.softmax),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_x.as_matrix(), train_y.as_matrix(), epochs=10)

test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test Accuracy: ', test_acc, '   Test Loss: ', test_loss)
