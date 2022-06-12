import pandas as pd
from visualize import *
from ECGAutoencoder import *


def main():
    x_train = pd.read_csv('./datasets/mitbih_train.csv', header=None).to_numpy()[:, :-1]
    x_test = pd.read_csv('./datasets/mitbih_test.csv', header=None).to_numpy()[:, :-1]

    input_size = x_train.shape[1]
    layers = [(64, 'relu'), (16, 'relu'), (64, 'relu'), (input_size, 'sigmoid')]
    layer_sizes = [layer[0] for layer in layers]
    activations = [layer[1] for layer in layers]
    encoder_output_index = 1

    model = ECGAutoencoder(input_size, layer_sizes, activations, encoder_output_index)

    MLP_data = model.autoencoder.fit(x_train, x_train, epochs=15, batch_size=256, shuffle=True,
                                     validation_data=(x_test, x_test))
    visualize_accuracy(MLP_data.history)

    encoded_data = model.encoder.predict(x_test)
    decoded_data = model.decoder.predict(encoded_data)

    visualize_results(x_test, encoded_data, decoded_data)


if __name__ == '__main__':
    main()
