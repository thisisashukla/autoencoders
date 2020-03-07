from autoencoder.models import DenseAE, ConvAE
import pytest
import sys
sys.path.append('../')


@pytest.mark.train
def test_dense(mnist_input_flat):

    layers = [10, 5]
    activations = ['relu', 'relu']
    ae = DenseAE(mnist_input_flat.shape[1], layers, activations)
    ae.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    assert len(ae.encoder.layers) == len(ae.decoder.layers)
    assert len(ae.encoder.layers) + len(ae.decoder.layers) == 2*len(layers)

    weights_before = ae.model.get_weights()

    ae.fit(mnist_input_flat, batch_size=10, epochs=1)

    weights_after = ae.model.get_weights()

    assert all([l != 0 for l in ae.model.history.history['loss']])
    for b, a in zip(weights_before, weights_after):
        assert (b!=a).any()
    
    encoded = ae.encoder.predict(mnist_input_flat)
    assert encoded.shape == (mnist_input_flat.shape[0], layers[-1])
    assert ae.decoder.predict(encoded).shape == mnist_input_flat.shape

@pytest.mark.train
def test_conv(mnist_input_image):

    layers = [16, 8]
    activations = ['relu', 'relu']
    kernels = [3, 3]
    ae = ConvAE(mnist_input_image.shape[1:], layers, activations, kernels)
    ae.compile(optimizer='adadelta', loss='binary_crossentropy',
               metrics=['accuracy'])

    assert len(ae.encoder.layers) == len(ae.decoder.layers)
    assert len(ae.encoder.layers) + len(ae.decoder.layers) == 2*len(layers)

    weights_before = ae.model.get_weights()

    ae.fit(mnist_input_image, batch_size=10, epochs=1)

    weights_after = ae.model.get_weights()

    assert all([l != 0 for l in ae.model.history.history['loss']])
    for b, a in zip(weights_before, weights_after):
        assert (b != a).any()


    encoded = ae.encoder.predict(mnist_input_image)
    assert encoded.shape == (mnist_input_image.shape[0], mnist_input_image.shape[1], mnist_input_image.shape[2], layers[-1])
    assert ae.decoder.predict(encoded).shape == mnist_input_image.shape
