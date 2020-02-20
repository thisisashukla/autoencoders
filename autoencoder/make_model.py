from keras import Sequential
from keras.layers import Dense, Conv2D


class autoencoder():

    def __init__(self, input_shape, layer_type, layer_widths, activations):

        self.input_shape = input_shape
        self.layer_type = layer_type
        self.layer_widths = layer_widths
        self.activations = activations
        self.model = Sequential()

    def compile_model(self, optimizer, loss, metrics):

        Layer = Dense
        if self.layer_type == 'Conv2D':
            Layer = Conv2D

            
        self.model.add(Layer(self.layer_widths[0], 
                            activation = self.activations[0], input_dim = self.input_shape))

        for width, act in zip(self.layer_widths[1:], self.activations[1:]):
            
            self.model.add(Layer(width, activation = act))
        
        self.model.compile(optimizer = optimizer, loss = loss, metric = metrics)

    def fit_model(self, data, batch_size, epochs = 10):

        self.model.fit(data, data, epochs = epochs, batch_size = batch_size)

    def compile_and_fit(self, optimizer, loss, metrics, data, batch_size, epochs = 10):

        self.compile(optimizer, loss, metrics)
        self.fit_model(data, batch_size, epochs)

    

            

