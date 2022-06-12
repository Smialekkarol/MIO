import matplotlib.pyplot as plt
import random


def visualize_accuracy(history):
    plt.clf()
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('network accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png', dpi=300, bbox_inches='tight')


def visualize_results(x_test, encoded_ecg, decoded_ecg):
    n = 5
    indices = [random.randint(0, len(x_test)) for _ in range(n)]
    plt.figure(figsize=(n * 3, 4))

    for index, value in enumerate(indices):
        plt.subplot(3, n, index + 1)
        plt.plot(range(len(x_test[value])), x_test[value])

        plt.subplot(3, n, index + 1 + n)
        plt.plot(range(len(encoded_ecg[value])), encoded_ecg[value])

        plt.subplot(3, n, index + 1 + 2 * n)
        plt.plot(range(len(decoded_ecg[value])), decoded_ecg[value])

    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')