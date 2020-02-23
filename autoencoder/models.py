from abc import ABC, abstractmethod
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D

class Autoencoder(ABC):

    def __init__(self, input_shape, layer_widths, activations):

        self.input_shape = input_shape
        self.layer_widths = layer_widths
        self.activations = activations
        self.model = Sequential()

    @abstractmethod
    def compile(self):        
        pass

    def fit(self, data, batch_size, epochs=10):

        self.model.fit(data, data, epochs=epochs, batch_size=batch_size)

    def compile_and_fit(self, optimizer, loss, metrics, data, batch_size, epochs=10):

        self.compile(optimizer, loss, metrics)
        self.fit(data, batch_size, epochs)


class DenseAE(Autoencoder):

    def __init__(self, input_shape, layer_widths, activations):

        super(DenseAE, self).__init__(input_shape, layer_widths, activations)

    def compile(self, optimizer, loss, metrics):

        self.model.add(Dense(self.layer_widths[0], 
                            activation = self.activations[0], input_dim = self.input_shape))

        for width, act in zip(self.layer_widths[1:], self.activations[1:]):
            self.model.add(Dense(width, activation = act))
        
        self.model.add(Dense(self.input_shape, activation = 'softmax'))        
        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
        

class ConvAE(Autoencoder):
    
    def __init__(self, input_shape, layer_widths, activations):

        super(ConvAE, self).__init__(input_shape, layer_widths, activations)

    def compile(self, optimizer, loss, metrics):

        self.model.add(Dense(self.layer_widths[0],
                             activation=self.activations[0], input_dim=self.input_shape))

        for width, act in zip(self.layer_widths[1:], self.activations[1:]):
            self.model.add(Dense(width, activation=act))

        self.model.add(Dense(self.input_shape, activation='softmax'))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    

            

