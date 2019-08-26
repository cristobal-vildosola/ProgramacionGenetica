from tarea1.Data import *
from tarea1.Plot import *
from tarea1.NeuralNetwork import *


def main(hidden_layer_sizes=(10, 50, 100), repetitions=5, iterations=100):
    # obtener datos, normalizar y separar
    data = read_csv('dataset_digitos.txt')
    normalize(data)
    [training, validation] = separate_data(data, 0.8, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

    print('Preprocessing done')

    # entrenar
    clfs = []
    n = len(hidden_layer_sizes)
    for i in range(n):
        clfs.append([])

        for j in range(repetitions):
            clf = MLP(hidden_size=hidden_layer_sizes[i],
                      data=training[:, :-1], labels=training[:, -1],
                      iterations=iterations)
            clfs[i].append(clf)

        print(f'Training with {hidden_layer_sizes[i]} hidden neurons done\n')

    # validar
    conf_mat = []
    for i in range(n):
        conf_mat.append([])

        for j in range(repetitions):
            conf_mat[i].append(
                clfs[i][j].conf_matrix(
                    data=validation[:, :-1], labels=validation[:, -1])
            )

    # obtener puntajes
    scores = []
    for i in range(n):
        scores.append([])

        for j in range(repetitions):
            scores[i].append(conf_score(conf_mat[i][j]))

    print('Validation done')

    if n < 9:  # graficar matrices de confusion (cuando son pocas)
        plot_mats(hidden_layer_sizes, conf_mat)

    # calcular estadisticas
    score_mean = numpy.mean(scores, axis=1)
    score_std = numpy.std(scores, axis=1)

    # graficar rendimiento por numero de neuronas
    plot_performance(hidden_layer_sizes, score_mean, score_std)
    return


if __name__ == '__main__':
    main(hidden_layer_sizes=(25, 50, 100, 200), repetitions=5, iterations=1000)
