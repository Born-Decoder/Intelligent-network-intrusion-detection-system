import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Reshape, Flatten
from keras.models import Sequential

data = pd.read_csv('normout.csv')


x = data[['Time','Source','Destination','Length','CIP','CIP CM','CIP I/O','ENIP','GVCP','NBNS','SSDP','TCP']].copy()
y = data[['Safe']].copy()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)#, random_state = 0)
train_x=train_x.as_matrix()
train_y=train_y.as_matrix()
test_x=test_x.as_matrix()
test_y=test_y.as_matrix()
model = Sequential()

model.add(Dense(100, activation='softmax'))
model.add(Reshape((1,100)))
model.add(Dense(100, activation='softmax'))
model.add(LSTM(100, return_sequences=True))
model.add(Flatten())
model.add(Dense(100, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10)

test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test Accuracy: ', test_acc, '   Test Loss: ', test_loss)
