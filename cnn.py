import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential


data = pd.read_csv('normout.csv')

x = data[['Time','Source','Destination','Length','CIP','CIP CM','CIP I/O','ENIP','GVCP','NBNS','SSDP','TCP']].copy()
y = data[['Safe']].copy()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)#, random_state = 0)
train_x = train_x.values
train_y = train_y.values
train_y = train_y[:,0]
test_x = test_x.values
test_y = test_y.values
test_y = test_y[:,0]

model = Sequential()

model.add(Conv2D(32,input_shape=(,12),return_sequences=True))
model.add(Conv2D(32,return_sequences=True))
#model.add(Flatten())
model.add(Dense(100, activation='softmax'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_x, train_y, epochs=10)

test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test Accuracy: ', test_acc, '   Test Loss: ', test_loss)
