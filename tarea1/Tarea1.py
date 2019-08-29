import numpy as np

from src.Data import normalize, read_csv, separate_data
from src.NeuralNetwork import MLP
from src.Plot import plot_evolution, plot_mats, plot_accuracy
from src.Activation import ActivationFunc, Tanh, Sigmoid


def main(
        hidden_layers_sizes: list = (10, 50, 100),
        iterations: int = 100,
        learning_rate: float = 0.01,
        hidden_layers_acts: list = Tanh(),
        output_act: ActivationFunc = Sigmoid(),
        show_evolution: bool = False,
        verbose: bool = False,
        seed: int = 42,
):
    # set random seed
    np.random.seed(seed)

    # load data, normalize and separate training and validations sets
    data = read_csv('dataset_digitos.csv')
    normalize(data)
    [training, validation] = separate_data(data, 0.8)
    print('Data processing done\n')

    # train classifiers using different hidden layers sizes
    clfs = []
    n = len(hidden_layers_sizes)

    for i in range(n):
        print(f'Training with {hidden_layers_sizes[i]} hidden neurons')

        clf = MLP(
            data=training[:, :-1], labels=training[:, -1],
            hidden_layers_sizes=hidden_layers_sizes[i],
            hidden_layers_acts=hidden_layers_acts[i],
            output_act=output_act,
        )

        evolution = clf.train(
            iterations=iterations,
            learning_rate=learning_rate,
            history=show_evolution,
            verbose=verbose,
        )

        if show_evolution:
            plot_evolution(evolution)
        clfs.append(clf)
        print()

    # test performance
    conf_mats = []
    accs = []
    for i in range(n):
        acc, conf_mat = clfs[i].validate(data=validation[:, :-1], labels=validation[:, -1])
        conf_mats.append(conf_mat)
        accs.append(acc)

    if n < 9:  # plot confusion matrix
        plot_mats(hidden_layers_sizes, conf_mats)

    # plot accuracy obtained
    plot_accuracy(hidden_layers_sizes, accs)
    return


if __name__ == '__main__':
    main(
        # list of hidden sizes to try (each element represents a different classiffier)
        hidden_layers_sizes=[25, 50, 100, (30, 20), (80, 30)],

        # numer of iterations to train
        iterations=1000,

        # learning rate for training
        learning_rate=0.1,

        # change activation functions in hidden layers
        hidden_layers_acts=[Tanh(), Tanh(), Tanh(), Tanh(), [Tanh(), Sigmoid()]],

        # change activation function in output layer
        output_act=Sigmoid(),

        # if true show training progress
        verbose=True,

        # if true plot acc/loss evolution (will slow down training considerably)
        show_evolution=True,

        # random seed
        seed=42,
    )
