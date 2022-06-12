import keras
import keras.layers


class ECGAutoencoder(keras.Model):
    def __init__(self, input_size, layer_sizes, activations, encoder_output_index):
        super().__init__()

        self.input_length = keras.Input(shape=(input_size,))
        self.dense_layers = [keras.layers.Dense(size, activation)
                             for size, activation in zip(layer_sizes, activations)]

        self.encoder_layers = self.dense_layers[:encoder_output_index + 1]
        self.decoder_layers = self.dense_layers[encoder_output_index + 1:]
        tensor = self.input_length

        for index, layer in enumerate(self.dense_layers):
            tensor = layer(tensor)
            if index == encoder_output_index:
                encoded_tensor = tensor

        self.autoencoder = keras.Model(self.input_length, tensor)
        self.encoder = keras.Model(self.input_length, encoded_tensor)
        encoded_input = keras.Input(shape=(layer_sizes[encoder_output_index],))
        tensor = encoded_input

        for layer in self.decoder_layers:
            tensor = layer(tensor)
        self.decoder = keras.Model(encoded_input, tensor)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def call(self, inputs, training=None, mask=None):
        return self.decoder_layers[-1]
