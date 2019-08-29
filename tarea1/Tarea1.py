import numpy as np

from tarea1.Data import normalize, read_csv, separate_data
from tarea1.NeuralNetwork import MLP
from tarea1.Plot import plot_evolution, plot_mats, plot_performance


def main(hidden_layer_sizes=(10, 50, 100), iterations=100, show_evolution=False):
    np.random.seed(42)

    # obtener datos, normalizar y separar
    data = read_csv('dataset_digitos.txt')
    normalize(data)
    [training, validation] = separate_data(data, 0.8, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    print('Preprocessing done\n')

    # entrenar
    clfs = []
    n = len(hidden_layer_sizes)

    for i in range(n):
        print(f'Training with {hidden_layer_sizes[i]} hidden neurons')

        clf = MLP(hidden_layer_sizes=hidden_layer_sizes[i],
                  data=training[:, :-1], labels=training[:, -1])
        evolution = clf.train(iterations=iterations, learning_rate=0.1, history=show_evolution)
        if show_evolution:
            plot_evolution(evolution)

        clfs.append(clf)
        print()

    # validate
    conf_mats = []
    accs = []
    for i in range(n):
        acc, conf_mat = clfs[i].validate(data=validation[:, :-1], labels=validation[:, -1])
        conf_mats.append(conf_mat)
        accs.append(acc)

    if n < 9:  # graficar matrices de confusion (cuando son pocas)
        plot_mats(hidden_layer_sizes, conf_mats)

    # graficar rendimiento por numero de neuronas
    plot_performance(hidden_layer_sizes, accs)
    return


if __name__ == '__main__':
    main(
        hidden_layer_sizes=[25, 50, 100, (30, 20), (80, 30)],
        iterations=1000,
        show_evolution=False  # change to True to see acc/loss evolution
    )
