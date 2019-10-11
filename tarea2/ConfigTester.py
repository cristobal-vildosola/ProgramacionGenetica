import time

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy

from WordFinder import guess_word


def heatmap(mat, xlabels, ylabels, title, norm=None):
    plt.imshow(mat, origin='lower', norm=norm)
    plt.colorbar()

    # add ticks labels
    plt.xticks(range(len(xlabels)), labels=xlabels)
    plt.yticks(range(len(ylabels)), labels=ylabels)

    plt.title(title)
    plt.xlabel('tamaño de la población')
    plt.ylabel('tasa de mutación')
    return


def test_configs():
    word = 'hola como le baila?'

    repetitions = 5
    population_size = range(100, 1001, 100)
    mutation = [i / 10 for i in range(1, 11)]

    iterations = numpy.zeros([len(mutation), len(population_size)])
    times = numpy.zeros([len(mutation), len(population_size)])

    for i in range(len(iterations)):
        for j in range(len(iterations[0])):
            print(f'using {population_size[j]} individuals with mutation rate = {mutation[i]}')

            results = ([], [])
            for _ in range(repetitions):
                t0 = time.time()
                results[0].append(
                    guess_word(
                        word, size=population_size[j], mutation_rate=mutation[i], iters=-1, verbose=False,
                    ))
                results[1].append(time.time() - t0)

                print(f'\tfound word in {results[0][-1]} iterations ({results[1][-1]:.1f} seconds)')
            iterations[i][j] = sum(results[0]) / repetitions
            times[i][j] = sum(results[1]) / repetitions

    # plot iterations
    plt.subplot(1, 2, 1)
    heatmap(
        iterations, norm=colors.LogNorm(),
        title='Generaciones necesarias para llegar al óptimo',
        xlabels=population_size, ylabels=mutation,
    )

    plt.subplot(1, 2, 2)
    heatmap(
        times,
        title='Tiempo necesario para llegar al óptimo (segundos)',
        xlabels=population_size, ylabels=mutation,
    )

    plt.show()
    return


if __name__ == '__main__':
    test_configs()
