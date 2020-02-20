import sys
sys.path.append('../')
from autoencoder.make_model import autoencoder
from keras.datasets import mnist
import numpy as np


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
ae = autoencoder(x_train.shape[0], 'Dense', [10, 5, 10], ['relu', 'relu', 'relu'])
ae.compile_model(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

assert len(ae.model.layers) == 3

ae.fit_model(x_train, batch_size = 10, epochs = 1)