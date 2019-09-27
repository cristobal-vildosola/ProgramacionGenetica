import numpy
import random
import matplotlib.pyplot as plt


class SigmoidPerceptron:
    def __init__(self, learning_rate=0.1, dims=2):
        self.weights = [random.uniform(-2, 2) for _ in range(dims)]
        self.bias = random.uniform(-2, 2)

        self.learning_rate = learning_rate

    def output(self, x):
        z = numpy.dot(x, self.weights) + self.bias
        activation = 1 / (1 + numpy.exp(-z))
        return activation

    def predict(self, x):
        return 1 if self.output(x) > 0.5 else 0

    def train(self, x, output):
        diff = output - self.output(x)

        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * diff * x[i]
        self.bias += self.learning_rate * diff

        return

    def evaluate(self, points, outputs):
        precission = 0
        for i in range(len(points)):
            precission += self.predict(points[i]) == outputs[i]

        return precission / len(points) * 100


def main():
    random.seed('holi ke ase')
    perceptron = SigmoidPerceptron()

    n = 1000
    iterations = 10
    batch_size = n // 5

    a = random.uniform(-1, 1)
    b = random.uniform(-5, 5)

    line_x = [-11, 11]
    line_y = [a * x + b for x in line_x]

    training_points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(n)]
    training_outputs = [a * x + b > y for (x, y) in training_points]

    test_points = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(n)]
    test_outputs = [a * x + b > y for (x, y) in test_points]

    precissions = []

    for it in range(iterations):
        # shuffle training points each epoch
        if (it + 1) % 5 == 0:
            random.shuffle(training_points)
            training_outputs = [a * x + b > y for (x, y) in training_points]

        step = (it % 5)
        for i in range(batch_size * step, batch_size * (step + 1)):
            perceptron.train(training_points[i], training_outputs[i])

        precission = perceptron.evaluate(test_points, test_outputs)
        precissions.append(precission)

        points_0 = []
        points_1 = []
        for x in test_points:
            if perceptron.predict(x):
                points_0.append(x)
            else:
                points_1.append(x)

        plt.figure()
        plt.plot(line_x, line_y, '-g')
        plt.plot(*zip(*points_0), '.r')
        plt.plot(*zip(*points_1), '.b')
        plt.title(f'iteration {it + 1} got {precission:.1f}% acc')
        plt.show()

    plt.figure()
    plt.plot(range(iterations), precissions, '.-b')
    plt.title(f'precission on each iteration')
    plt.show()

    return


if __name__ == '__main__':
    main()
