import pytest
import numpy as np
import scipy.io as sio
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split


@pytest.fixture
def mnist_input_flat                                                                                                                      ():

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    return x_train

@pytest.fixture
def mnist_input_image():

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.

    x_train = np.expand_dims(x_train, axis = 0)

    return x_train

# @pytest.fixture
# def hyperspectral_input():

#     data = sio.loadmat('../dataset/hyperspectral/Indian_pines.mat')['indian_pines'].astype(float)
#     gt = sio.loadmat('../dataset/hyperspectral/ip_gt.mat')

    


    

