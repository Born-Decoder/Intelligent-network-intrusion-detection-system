import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dbn.tensorflow import SupervisedDBNClassification
from sklearn.metrics.classification import accuracy_score

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

classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='sigmoid',
                                         dropout_p=0.2)

classifier.fit(train_x, train_y)

classifier.save('model.pkl')
# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

Y_pred = classifier.predict(test_x)
print('Done.\nAccuracy: %f' % accuracy_score(test_y, Y_pred))
