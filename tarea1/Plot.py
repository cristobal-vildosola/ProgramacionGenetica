import math

import matplotlib.pyplot as plt


def plot_mats(hidden_layer_size, mats):
    plt.figure()

    n = len(hidden_layer_size)
    sqrt = math.sqrt(n)
    a = math.ceil(sqrt)

    for i in range(n):
        plt.subplot(a, a, i + 1)
        plt.title(f'{hidden_layer_size[i]} neuronas')

        plt.imshow(mats[i][0], cmap='RdYlGn')
        plt.colorbar()

    return plt.show()


def plot_performance(hidden_layer_size, score_mean, score_std):
    plt.figure()

    # linea con barras de error
    plt.errorbar(hidden_layer_size, score_mean, score_std, ecolor='r')

    plt.title('Puntaje del Clasificador según Número de Neuronas')
    plt.xlabel('número de neuronas')
    plt.ylabel('puntaje')
    plt.ylim(0, 1)

    return plt.show(block=True)

