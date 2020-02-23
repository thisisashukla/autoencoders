import pytest
import numpy as np
from tensorflow.keras.datasets import mnist


@pytest.fixture
def mnist_input():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    return x_train
