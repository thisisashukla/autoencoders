from abc import ABC, abstractmethod
from tensorflow.keras import Sequential

class LayerException(Exception):
    pass

class Autoencoder(ABC):

    def __init__(self, input_shape, enc_layers, activations):

        if len(enc_layers)%2 != 0:
            raise LayerException('Number of layers in the encoder should be even!')
        self.input_shape = input_shape
        self.enc_layers = enc_layers
        self.activations = activations
        self.encoder = Sequential()
        self.decoder = Sequential()
        self.model = Sequential()

        assert len(self.enc_layers) == len(self.activations)

    @abstractmethod
    def compile(self):        
        pass

    def fit(self, data, batch_size, epochs=10, verbose = 0):
        self.model.fit(data, data, epochs=epochs, batch_size=batch_size, verbose = verbose)

    def compile_and_fit(self, optimizer, loss, metrics, data, batch_size, epochs=10):
        self.compile(optimizer, loss, metrics)
        self.fit(data, batch_size, epochs)

    def encode(self, data):
        return self.encoder.predict(data)
    
    def decode(self, data):
        return self.decoder.predict(data)

