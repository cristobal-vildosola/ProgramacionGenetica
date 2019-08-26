import time

import numpy


def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))


def mean_square_error(predictions, labels):
    return numpy.sum((0.5 * (predictions - labels) ** 2).mean(axis=1)) / labels.shape[1]


def conf_score(conf_mat):
    total = 0
    for i in range(len(conf_mat)):
        total += conf_mat[i][i]
    return total / len(conf_mat)


class MLP:
    def __init__(self, data, labels, hidden_size, iterations=1000, learning_rate=0.01, verbose=False):
        self.hidden_size = hidden_size
        self.classes = numpy.array(list(set(labels)))

        self.weights_1 = numpy.random.randn(self.hidden_size, data.shape[1])
        self.bias_1 = numpy.zeros((self.hidden_size, 1))
        self.weights_2 = numpy.random.randn(len(self.classes), self.hidden_size)
        self.bias_2 = numpy.zeros((len(self.classes), 1))

        self.encoding = {}
        for i in range(len(self.classes)):
            one_hot_encoding = [0] * len(self.classes)
            one_hot_encoding[i] = 1
            self.encoding[self.classes[i]] = one_hot_encoding

        self.learning_rate = learning_rate
        self.train(data, labels, iterations=iterations, verbose=verbose)

    def forward_prop(self, data):
        # Layer 1
        z1 = numpy.dot(self.weights_1, data) + self.bias_1
        a1 = numpy.tanh(z1)

        # Layer 2
        z2 = numpy.dot(self.weights_2, a1) + self.bias_2
        a2 = sigmoid(z2)

        cache = {
            "a1": a1,
            "a2": a2
        }
        return a2, cache

    # Apply the backpropagation
    def backward_prop(self, data, labels, cache):
        a1 = cache["a1"]
        a2 = cache["a2"]

        n = data.shape[1]

        # Compute the difference between the predicted value and the real values
        d_z2 = a2 - labels
        d_w2 = numpy.dot(d_z2, a1.T) / n
        db2 = numpy.sum(d_z2, axis=1, keepdims=True) / n

        # Because d/dx tanh(x) = 1 - tanh^2(x)
        d_z1 = numpy.multiply(numpy.dot(self.weights_2.T, d_z2), 1 - numpy.power(a1, 2))
        d_w1 = numpy.dot(d_z1, data.T) / n
        db1 = numpy.sum(d_z1, axis=1, keepdims=True) / n

        return {
            "d_w1": d_w1,
            "db1": db1,
            "d_w2": d_w2,
            "db2": db2
        }

    def update_parameters(self, grads, learning_rate):
        self.weights_1 = self.weights_1 - learning_rate * grads["d_w1"]
        self.bias_1 = self.bias_1 - learning_rate * grads["db1"]
        self.weights_2 = self.weights_2 - learning_rate * grads["d_w2"]
        self.bias_2 = self.bias_2 - learning_rate * grads["db2"]
        return

    def train(self, data, labels, iterations=1000, learning_rate=0.01, verbose=False):
        # fix input format
        data = data.T
        encoded_labels = numpy.array([self.encoding[label] for label in labels]).T

        # train
        t0 = time.time()

        for i in range(1, iterations + 1):
            output, cache = self.forward_prop(data)
            grads = self.backward_prop(data, encoded_labels, cache)
            self.update_parameters(grads, learning_rate)

            if verbose and i % (iterations // 5) == 0:
                loss = mean_square_error(output, encoded_labels)
                print(f'loss after iteration {i}: {loss:.4f}')

        output, _ = self.forward_prop(data)
        loss = mean_square_error(output, encoded_labels)
        print(f'Done training, loss: {loss:.6f} ({time.time() - t0:.1f} seconds)')
        return

    def predict(self, data):
        output, _ = self.forward_prop(data.T)

        # return original classes
        indexes = numpy.argmax(output, axis=0)
        return self.classes[indexes]

    def conf_matrix(self, data, labels):
        # obtener predicciones
        predictions = self.predict(data)

        # obtener matriz de confusion
        n_class = len(self.classes)
        conf_mat = numpy.zeros((n_class, n_class))

        for i in range(len(data)):
            prediccion = int(predictions[i])
            real = int(labels[i])
            conf_mat[prediccion][real] += 1

        # normalizar matriz a [0, 1] usando eje x
        for i in range(n_class):
            total = sum(conf_mat[i])
            for j in range(n_class):
                conf_mat[i][j] = round(conf_mat[i][j] / total, 4)

        return conf_mat


def main():
    # Set the seed to make result reproducible
    numpy.random.seed(42)

    # The 4 training examples by columns
    data = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = ['false', 'true', 'true', 'false']

    mlp = MLP(data, labels, 4, iterations=10000, learning_rate=0.01, verbose=True)

    x_test = numpy.array([[0, 0], [1, 0], [1, 1]])
    y_predict = mlp.predict(x_test)

    # Print the result
    print(f'Neural Network prediction for example {x_test[1]} is {y_predict[1]}')
    return


if __name__ == '__main__':
    main()
