import pytest
import sys
sys.path.append('../')

from autoencoder.models import DenseAE, ConvAE

@pytest.mark.model
def test_dense(mnist_input):
    
    layers = [10, 5, 10]
    ae = DenseAE(mnist_input.shape[1], layers, ['relu', 'relu', 'relu'])
    ae.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    assert len(ae.model.layers) == len(layers) + 1
