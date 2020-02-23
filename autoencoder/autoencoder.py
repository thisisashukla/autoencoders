from abc import ABC, abstractmethod
from tensorflow.keras import Sequential

class Autoencoder(ABC):

    def __init__(self, input_shape, layer_widths, activations):

        self.input_shape = input_shape
        self.layer_widths = layer_widths
        self.activations = activations
        self.model = Sequential()

    @abstractmethod
    def compile(self):        
        pass

    def fit(self, data, batch_size, epochs=10, verbose = 0):

        self.model.fit(data, data, epochs=epochs, batch_size=batch_size, verbose = verbose)

    def compile_and_fit(self, optimizer, loss, metrics, data, batch_size, epochs=10):

        self.compile(optimizer, loss, metrics)
        self.fit(data, batch_size, epochs)
