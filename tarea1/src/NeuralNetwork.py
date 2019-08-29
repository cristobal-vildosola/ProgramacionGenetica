import time
from typing import Union, List, Tuple

import numpy as np

from .Activation import ActivationFunc, Tanh, Sigmoid


def mean_square_error(predictions, labels):
    return np.sum((0.5 * (predictions - labels) ** 2).mean(axis=1)) / labels.shape[1]


class MLP:
    def __init__(
            self,
            data: np.ndarray,
            labels: Union[list, np.ndarray],
            hidden_layers_sizes: Union[int, List[int], Tuple[int], np.ndarray],
            hidden_layers_acts: Union[ActivationFunc, List[ActivationFunc], Tuple[ActivationFunc]] = Tanh(),
            output_act: ActivationFunc = Sigmoid()
    ):
        """
        Create a Multi Layer Perceptron with the given parameters.

        :param data: Example input data, it is saved for default training and used for getting the number of inputs.
        :param labels: Example output data, it is saved for default training and used for getting the number of outputs.
        :param hidden_layers_sizes: Hidden layers sizes. Can be an int for a single hidden layer, or a
            list/tuple/ndarray for multiple hidden layers.
        :param hidden_layers_acts: Hidden layers activation functions. Can be one function to use for all the hidden
            layers, or a list/tuple to define one for each layer.
        :param output_act: Output layer activation function.
        """

        # check data and labels
        assert isinstance(data, np.ndarray), 'data must be a numpy array'
        assert len(data.shape) == 2, 'data must be 2D matrix'
        assert isinstance(labels, (list, np.ndarray)), 'labels must be a list or numpy array'
        assert len(data) == len(labels), 'there must be the same number of data points and labels'

        # check valid hidden layer sizes
        if not isinstance(hidden_layers_sizes, (list, tuple, np.ndarray)):
            hidden_layers_sizes = [hidden_layers_sizes]
        assert isinstance(hidden_layers_sizes, int) or \
               (isinstance(hidden_layers_sizes, list) and all(isinstance(x, int) for x in hidden_layers_sizes)) or \
               (isinstance(hidden_layers_sizes, tuple) and all(isinstance(x, int) for x in hidden_layers_sizes)) or \
               (isinstance(hidden_layers_sizes, np.ndarray) and np.issubdtype(hidden_layers_sizes.dtype, np.integer)), \
            'hidden_sizes must be an int or a list of ints'
        assert min(hidden_layers_sizes) > 0, 'hidden sizes should all be greater than 0'

        # check activation functions
        if not isinstance(hidden_layers_acts, (list, tuple)):
            hidden_layers_acts = [hidden_layers_acts] * len(hidden_layers_sizes)
        assert all(issubclass(type(act), ActivationFunc) for act in hidden_layers_acts), \
            'activation function must extend ActivationFunc class'
        assert issubclass(type(output_act), ActivationFunc), \
            'output activation function must extend ActivationFunc class'

        # check consistency between layers and activation functions
        assert len(hidden_layers_acts) == len(hidden_layers_sizes), \
            'number of activation functions must be equal to number of hidden layers'

        self.data = data
        self.labels = labels
        self.hidden_layers_sizes = hidden_layers_sizes
        self.hidden_layers_acts = hidden_layers_acts
        self.output_act = output_act
        self.classes = np.array(list(set(labels)))

        # generate weights
        self._weights = []
        self._bias = []
        self.generate_params()

        # generate one hot encoding
        self.encoding = {}
        codes = np.eye(len(self.classes))

        for i in range(len(self.classes)):
            self.encoding[self.classes[i]] = codes[i]

    def generate_params(self):
        """
        Generate initial weights and bias for the network
        """
        # first layer
        self._weights.append(np.random.randn(self.hidden_layers_sizes[0], self.data.shape[1]))
        self._bias.append(np.zeros((self.hidden_layers_sizes[0], 1)))

        # intermediate layers
        for i in range(1, len(self.hidden_layers_sizes)):
            self._weights.append(np.random.randn(self.hidden_layers_sizes[i], self.hidden_layers_sizes[i - 1]))
            self._bias.append(np.zeros((self.hidden_layers_sizes[i], 1)))

        # output layer
        self._weights.append(np.random.randn(len(self.classes), self.hidden_layers_sizes[-1]))
        self._bias.append(np.zeros((len(self.classes), 1)))

        return

    def forward_prop(self, data: np.ndarray):
        """
        forward data through the network.

        :param data:
        :return: output of the network and cache of activation value in each layer.
        """

        # fix input format
        data = data.T

        cache = []
        activation = data

        for i in range(len(self._weights) - 1):
            z = self._weights[i] @ activation + self._bias[i]
            activation = self.hidden_layers_acts[i].apply(z)

            # save activation values for back propagation
            cache.append(activation)

        z = self._weights[-1] @ activation + self._bias[-1]
        output = self.output_act.apply(z)
        cache.append(output)

        return output, cache

    def backward_prop(self, data: np.ndarray, labels: np.ndarray, cache: list):
        """
        Calculate gradients using back propagation.

        :param data: input of the network.
        :param labels: exected output.
        :param cache: activation value in each layer of the network.

        :return: 2 lists containing the weights and bias gradients.
        """

        # fix input format
        data = data.T

        n = data.shape[1]
        d_z = [np.array(0)] * len(self._weights)
        weight_grads = [np.array(0)] * len(self._weights)
        bias_grads = [np.array(0)] * len(self._weights)

        # last layer diff
        d_z[-1] = (cache[-1] - labels) / n
        weight_grads[-1] = d_z[-1] @ cache[-2].T
        bias_grads[-1] = np.sum(d_z[-1], axis=1, keepdims=True)

        # hidden layers diff
        for i in range(len(self._weights) - 2, 0, -1):
            d_z[i] = np.multiply(
                self._weights[i + 1].T @ d_z[i + 1],
                self.hidden_layers_acts[i].derivative(cache[i]))

            weight_grads[i] = np.dot(d_z[i], cache[i - 1].T)
            bias_grads[i] = np.sum(d_z[i], axis=1, keepdims=True)

        # first layer diff
        d_z[0] = np.multiply(
            self._weights[1].T @ d_z[1],
            self.hidden_layers_acts[0].derivative(cache[0]))

        weight_grads[0] = np.dot(d_z[0], data.T)
        bias_grads[0] = np.sum(d_z[0], axis=1, keepdims=True)

        return weight_grads, bias_grads

    def update_parameters(self, weight_grads: list, bias_grads: list, learning_rate: float):
        """
        Update the network parameters.

        :param weight_grads: gradients for the weights (calculated using back propagation).
        :param bias_grads: gradients for the bias (calculated using back propagation).
        :param learning_rate: ponderation factor for the grads.
        """
        for i in range(len(self._weights)):
            self._weights[i] -= learning_rate * weight_grads[i]
            self._bias[i] -= learning_rate * bias_grads[i]
        return

    def train(self,
              iterations: int = 1000,
              learning_rate: float = 0.1,
              verbose: bool = False,
              history: bool = False,
              data: np.ndarray = None,
              labels: Union[list, np.ndarray] = None):
        """
        Train the network.

        :param iterations: number of iterations over the data.
        :param learning_rate: learning rate for updating the network parameters.
        :param verbose: if true print network progress every 10%.
        :param history: if true register network acc and loss each epoch.
        :param data: if given, use instead of initialization data.
        :param labels: if given, use instead of initialization data.

        :return: if history is True, evolution of the network.
        """

        # check params
        assert isinstance(iterations, int), f'iterations must be int, not {type(iterations)}'
        assert isinstance(learning_rate, float), f'learning_rate must be floar, not {type(learning_rate)}'
        assert isinstance(verbose, bool), f'verbose must be bool, not {type(verbose)}'
        assert isinstance(history, bool), f'history must be bool, not {type(history)}'

        # check that data and label are both or none provided
        assert data is None and labels is None or data is not None and labels is not None, \
            'If you provide data then you should provide labels for it too'

        # check data when provided
        if data is not None:
            assert isinstance(data, np.ndarray) and len(data.shape) == 2, 'data must be a 2-dimensional numpy array'
            assert isinstance(labels, (list, np.ndarray)), 'labels must be a list or numpy array'
            assert len(data) == len(labels), 'there must be the same number of data points and labels'

        # use default data when theres no data provided
        if data is None:
            data = self.data
            labels = self.labels

        # transform labels to one hot encoding
        encoded_labels = np.array([self.encoding[label] for label in labels]).T

        # train
        output, _ = self.forward_prop(data)
        evolution = {'iter': [],
                     'loss': [],
                     'acc': []}
        t0 = time.time()

        for i in range(1, iterations + 1):
            output, cache = self.forward_prop(data)
            weight_grads, bias_grads = self.backward_prop(data, encoded_labels, cache)
            self.update_parameters(weight_grads, bias_grads, learning_rate)

            # print progress every 1/5 iterations
            if verbose and i % (max(iterations // 5, 1)) == 0:
                loss = mean_square_error(output, encoded_labels)
                print(f'loss after iteration {i}/{iterations}: {loss:.6f}')

            # save evolution every 1/50 iterations
            if history:
                evolution['iter'].append(i)
                evolution['loss'].append(mean_square_error(output, encoded_labels))
                evolution['acc'].append(self.validate(data, labels)[0])

        # show training results
        output, _ = self.forward_prop(data)
        loss = mean_square_error(output, encoded_labels)
        acc = self.validate(data, labels)[0]
        print(f'Done training, loss: {loss:.6f} acc: {acc:.3f} ({time.time() - t0:.1f} seconds)')

        return evolution

    def predict(self, data: np.ndarray):
        """
        Predict using the network.

        :param data: data to use as input.

        :return: a list with the predicted class for each point.
        """
        # check data
        assert isinstance(data, np.ndarray) and len(data.shape) == 2, 'data must be a 2-dimensional numpy array'

        # retrieve network output
        output, _ = self.forward_prop(data)

        # return original classes
        indexes = np.argmax(output, axis=0)
        return self.classes[indexes]

    def validate(self, data: np.ndarray, labels: Union[list, np.ndarray]):
        """
        Validate network performance.

        :param data: data to use as input.
        :param labels: expected output for each data point.

        :return: the network accuracy and resulting confusion matrix.
        """
        # check data and labels
        assert isinstance(data, np.ndarray) and len(data.shape) == 2, 'data must be a 2-dimensional numpy array'
        assert isinstance(labels, (list, np.ndarray)), 'labels must be a list or numpy array'
        assert len(data) == len(labels), 'there must be the same number of data points and labels'

        # retrieve predictions
        predictions = self.predict(data)

        # obtain conf matrix
        n_class = len(self.classes)
        conf_mat = np.zeros((n_class, n_class))

        for i in range(len(data)):
            predicted = np.argmax(self.encoding[predictions[i]])
            real = np.argmax(self.encoding[labels[i]])
            conf_mat[predicted][real] += 1

        # calculate acc
        acc = np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat)

        return acc, conf_mat


def main():
    # Set the seed to make result reproducible
    np.random.seed(42)

    # The 4 training examples by columns
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = ['false', 'true', 'true', 'false']

    mlp = MLP(data, labels, 4)
    evolution = mlp.train(iterations=1000, learning_rate=0.1, verbose=True, history=True)

    from .Plot import plot_evolution
    plot_evolution(evolution)

    x_test = np.array([[0, 0], [1, 0], [1, 1]])
    y_predict = mlp.predict(x_test)

    # Print the result
    print(f'Neural Network prediction for example {x_test[1]} is {y_predict[1]}')
    return


if __name__ == '__main__':
    main()
