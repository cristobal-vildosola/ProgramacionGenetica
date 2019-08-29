from abc import ABC, abstractmethod

import numpy


class ActivationFunc(ABC):
    @staticmethod
    @abstractmethod
    def apply(x):
        pass

    @staticmethod
    @abstractmethod
    def derivative(x):
        pass


class Sigmoid(ActivationFunc):
    @staticmethod
    def apply(x: numpy.ndarray):
        return 1 / (1 + numpy.exp(-x))

    @staticmethod
    def derivative(x: numpy.ndarray):
        return x * (1 - x)


class Tanh(ActivationFunc):
    @staticmethod
    def apply(x: numpy.ndarray):
        return numpy.tanh(x)

    @staticmethod
    def derivative(x: numpy.ndarray):
        return 1 - numpy.power(x, 2)


class Step(ActivationFunc):
    @staticmethod
    def apply(x: numpy.ndarray):
        res = numpy.zeros_like(x)
        res[x >= 0] = 1
        return res

    @staticmethod
    def derivative(x: numpy.ndarray):
        return numpy.zeros_like(x)
