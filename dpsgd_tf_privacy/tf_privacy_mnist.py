# -*- coding: utf-8 -*-
"""tf-privacy_mnist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KD51qfd-6DVMAgUuH-MYBK442DAe3rYl
"""

# !pip install tensorflow
!pip install sklearn
!pip install tensorflow-privacy

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
import numpy as np

# load and preprocess data

train, test = keras.datasets.mnist.load_data()
train_data, train_labels = train
test_data, test_labels = test

train_data = np.array(train_data, dtype=np.float32)/255
test_data = np.array(test_data, dtype=np.float32)/255

# print(train_data.shape)
# print(test_data.shape)

# (n_images, x_shape, y_shape, channels) is dim 4, can be reduced to dim 2 by  (n_images, x*y_shape)
train_data = train_data.reshape(train_data.shape[0], 28*28)
test_data = test_data.reshape(test_data.shape[0], 28*28)

train_labels = np.array(train_labels, dtype=np.int32)
test_labels = np.array(test_labels, dtype=np.int32)

train_labels  = keras.utils.to_categorical(train_labels, num_classes=10)
test_labels = keras.utils.to_categorical(test_labels, num_classes=10)

assert train_data.min() == 0.
assert train_data.max() == 1.
assert test_data.min() == 0.
assert test_data.max() == 1.

#pca
pca = PCA(n_components=60)
train_data = pca.fit_transform(train_data)
test_data = pca.transform(test_data)

#hyperparameters
epochs = 100
batch_size = 600
#dp hyperparameters
l2_norm_clip = 0.4
noise_multiplier = 1.0
num_microbatches = 600 #ensure that batch size is an integer multiple of num_microbatches
learning_rate = 0.05
delta = 1e-5

#building a model
model = keras.Sequential([
    keras.layers.Dense(units = 1000,
                       activation = 'relu',
                       input_shape  = (60,)
                      ),
    keras.layers.Dense(units = 10, 
                       activation = 'softmax')
])

#defining optimizer and loss
optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
    l2_norm_clip = l2_norm_clip,
    noise_multiplier = noise_multiplier,
    num_microbatches = num_microbatches,
    learning_rate = learning_rate
)

loss = keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.losses.Reduction.NONE)

#compiling the model
model.compile(optimizer = optimizer, loss = loss, metrics = [keras.metrics.CategoricalAccuracy()])

model.summary()

model.fit(train_data, train_labels,
          epochs = epochs,
          validation_data = (test_data, test_labels),
          batch_size = batch_size
)

#compute privacy loss
compute_dp_sgd_privacy.compute_dp_sgd_privacy(n = train_data.shape[0],
                                              batch_size = batch_size,
                                              noise_multiplier = noise_multiplier,
                                              epochs = epochs,
                                              delta = delta
)

#baseline code

# mnist = keras.datasets.mnist
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)

# pca = PCA(n_components=60)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# minmax = MinMaxScaler()
# X_train_pca = minmax.fit_transform(X_train_pca)
# X_test_pca = minmax.transform(X_test_pca)


# model = Sequential()
# model.add(Dense(1000, activation='relu', input_shape=(60,)))
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.summary()

# model.fit(X_train_pca, y_train, epochs=100, batch_size=32, validation_data=(X_test_pca, y_test))

