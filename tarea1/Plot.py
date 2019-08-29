import math

import matplotlib.pyplot as plt
import numpy as np


# normalize matrix to [0, 1] using axis 1
def normalize_matrix(mat: np.ndarray):
    sums = np.sum(mat, axis=1)
    for i in range(len(mat)):
        mat[i, :] = np.round(mat[i, :] / sums[i], 4)

    return mat


def plot_mats(hidden_layer_size, mats):
    plt.figure()

    n = len(hidden_layer_size)
    sqrt = math.sqrt(n)
    a = math.ceil(sqrt)

    for i in range(n):
        plt.subplot(a, a, i + 1)
        plt.title(f'{hidden_layer_size[i]} neuronas')

        plt.imshow(normalize_matrix(mats[i]), cmap='RdYlGn')
        plt.colorbar()

    plt.tight_layout()
    return plt.show(block=False)


def plot_performance(hidden_layer_sizes, accs):
    fig, ax = plt.subplots()

    # lines with error bars
    rects = ax.bar(range(len(accs)), accs, width=0.5, tick_label=hidden_layer_sizes)
    # plt.xticks(range(len(accs)), hidden_layer_sizes)

    for rect in rects:
        ax.annotate(f'{rect.get_height() * 100:.1f}%', color='white', weight='bold',
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, -5), textcoords="offset points",
                    ha='center', va='top')

    plt.title('Puntaje del Clasificador según Número de Neuronas')
    plt.xlabel('número de neuronas')
    plt.ylabel('puntaje')
    plt.ylim(0, 1)

    fig.tight_layout()
    return plt.show(block=True)


def plot_evolution(evolution, color1='#088a13', color2='#660bb5'):
    # plt configuration
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title('Evolución de función de pérdida y precisión al entrenar la red')

    # first line
    line1 = ax1.plot(evolution['iter'], evolution['acc'], linestyle='-', color=color1)
    ax1.tick_params('y', colors=color1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('precisión')
    ax1.set_xlabel('iteración')

    # second line
    line2 = ax2.plot(evolution['iter'], evolution['loss'], linestyle='--', color=color2)
    ax2.tick_params('y', colors=color2)
    ax2.set_ylim(0)
    ax2.set_ylabel('pérdida')

    # legend
    plt.legend(line1 + line2, ['precisión', 'pérdida'])

    fig.tight_layout()
    return plt.show()
