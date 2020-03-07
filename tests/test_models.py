import pytest
import sys
sys.path.append('../')

from autoencoder.models import DenseAE, ConvAE

@pytest.mark.model
def test_dense(mnist_input_flat):
    
    layers = [10, 5]
    activations = ['relu', 'relu']
    ae = DenseAE(mnist_input_flat.shape[1], layers, activations)
    ae.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    assert len(ae.encoder.layers) == len(ae.decoder.layers)
    assert len(ae.encoder.layers) + len(ae.decoder.layers) == 2*len(layers)


@pytest.mark.model
def test_conv(mnist_input_image):

    layers = [10, 5]
    activations = ['relu', 'relu']
    kernels = [2, 2]
    ae = ConvAE(mnist_input_image.shape[1:], layers, activations, kernels)
    ae.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=['accuracy'])

    assert len(ae.encoder.layers) == len(ae.decoder.layers)
    assert len(ae.encoder.layers) + len(ae.decoder.layers) == 2*len(layers)
