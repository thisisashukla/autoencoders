from autoencoder.models import DenseAE
import pytest
import sys
sys.path.append('../')


@pytest.mark.train
def test_dense(mnist_input):

    layers = [10, 5, 10]
    ae = DenseAE(mnist_input.shape[1], layers, ['relu', 'relu', 'relu'])
    ae.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=['accuracy'])

    weights_before = ae.model.get_weights()

    ae.fit(mnist_input, batch_size=10, epochs=1)

    weights_after = ae.model.get_weights()

    assert all([l != 0 for l in ae.model.history.history['loss']])
    for b, a in zip(weights_before, weights_after):
        assert (b!=a).any()
    
