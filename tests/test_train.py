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

    ae.fit(mnist_input, batch_size=10, epochs=1)

    ae.compile_and_fit(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],
                       data=mnist_input, batch_size=10, epochs=1)
