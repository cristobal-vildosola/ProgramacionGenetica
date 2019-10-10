import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy

from WordFinder import guess_word


def test_configs():
    word = 'hola lkahsdlaksjda'  # 'La InTeliGENciA ArtiFicIal Va A cOnquIsTar eL MUndO'

    repetitions = 3
    population_size = [25, 50, 100, 250, 500, 1000]
    mutation = [i / 100 for i in range(1, 6)]

    iterations = numpy.zeros([len(mutation), len(population_size)])

    for i in range(len(iterations)):
        for j in range(len(iterations[0])):
            print(f'using {population_size[j]} individuals with mutation rate = {mutation[i]}')

            results = []
            for _ in range(repetitions):
                results.append(
                    guess_word(
                        word, size=population_size[j], mutation_rate=mutation[i], iters=-1, verbose=False,
                    ))

                print(f'\tfound word in {results[-1]} iterations')
            iterations[i][j] = sum(results) / repetitions

    # plot results
    plt.imshow(iterations, origin='lower', norm=colors.LogNorm())
    plt.colorbar()

    # add ticks labels
    plt.xticks(range(len(population_size)), labels=population_size)
    plt.yticks(range(len(mutation)),  labels=mutation)

    plt.title('Desempeño usando distintas configuraciones')
    plt.xlabel('tamaño de la población')
    plt.ylabel('tasa de mutación')

    plt.show()
    return


if __name__ == '__main__':
    test_configs()
