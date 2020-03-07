from tensorflow.keras.layers import Dense, Conv2D
from autoencoder.autoencoder import Autoencoder


class DenseAE(Autoencoder):

    def __init__(self, input_shape, enc_layers, activations):
        super(DenseAE, self).__init__(input_shape, enc_layers, activations)

    def compile(self, optimizer, loss, metrics):
        for i, (width, act) in enumerate(zip(self.enc_layers, self.activations)):
            if i == 0:
                self.encoder.add(Dense(width, activation = act, input_dim = self.input_shape))
            else:
                self.encoder.add(Dense(width, activation = act))

        for i, (width, act) in enumerate(zip(self.enc_layers[::-1][1:], self.activations[::-1][1:])):
            if i == 0:
                self.decoder.add(Dense(width, activation = act, input_dim = self.enc_layers[-1]))
            else:
                self.decoder.add(Dense(width, activation = act))

        self.decoder.add(Dense(self.input_shape, activation = 'softmax'))        
        self.model.add(self.encoder)
        self.model.add(self.decoder)

        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)


class ConvAE(Autoencoder):
    
    def __init__(self, input_shape, enc_layers, activations, kernel_sizes):
        super(ConvAE, self).__init__(input_shape, enc_layers, activations)
        self.kernel_sizes = kernel_sizes
        self.decoder_input_shape = (input_shape[0], input_shape[1], enc_layers[-1])

    def compile(self, optimizer, loss, metrics):
        for i, (width, act, kernel) in enumerate(zip(self.enc_layers, self.activations, self.kernel_sizes)):
            if i == 0:
                self.encoder.add(Conv2D(filters = width, kernel_size = kernel, activation = act, padding = 'same', input_shape = self.input_shape))
            else:
                self.encoder.add(Conv2D(filters = width, kernel_size = kernel, activation = act, padding = 'same'))

        for i, (width, act, kernel) in enumerate(zip(self.enc_layers[::-1][1:], 
                                                     self.activations[::-1][1:], self.kernel_sizes[::-1][1:])):
            if i == 0:
                self.decoder.add(Conv2D(filters = width, kernel_size = kernel, activation = act, padding = 'same', input_shape = self.decoder_input_shape))
            else:
                self.decoder.add(Conv2D(filters = width, kernel_size = kernel, activation = act, padding = 'same'))

        self.decoder.add(Conv2D(filters = self.input_shape[-1], kernel_size = self.kernel_sizes[0], activation = 'sigmoid', padding = 'same'))        
        self.model.add(self.encoder)
        self.model.add(self.decoder)

        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

            

