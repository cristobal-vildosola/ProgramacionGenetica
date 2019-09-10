import random
import string
from abc import ABC, abstractmethod
from typing import Callable, List, Union, Tuple

import numpy
import matplotlib.pyplot as plt


class Param(ABC):
    @abstractmethod
    def mutate(self):
        pass


class Char(Param):
    def __init__(self, alphabet: Union[List[str], Tuple[str]] = tuple(string.printable[:-5])):
        self.alphabet = alphabet
        self.char = random.choice(alphabet)

    def mutate(self):
        return Char(self.alphabet)


class Individual:
    def __init__(self, params: List[Param], to_string: Callable = None):
        self.params: List[Param] = params
        self.to_string = to_string

    @staticmethod
    def crossover(one, other):
        index = random.randint(0, len(one.params) - 1)
        new_params = one.params[:index] + other.params[index:]

        index = random.randint(0, len(one.params) - 1)
        new_params[index] = new_params[index].mutate()

        return Individual(new_params, one.to_string)

    def __str__(self):
        if self.to_string:
            return self.to_string(self)
        return ''


class Population:
    def __init__(self,
                 params: List[Param],
                 fitness: Callable,
                 size: int = 100,
                 tournament_size=5,
                 to_string: Callable = None):

        self.params: List[Param] = params
        self.fitness: Callable = fitness
        self.size: int = size
        self.tournament_size = tournament_size

        self.individuals: List[Individual] = []
        self.fitnesses: List[float] = numpy.zeros(size)
        self.to_string = to_string

        self.gen_individuals()

    def gen_individuals(self):
        self.individuals = []

        for _ in range(self.size):
            new_params = [x.mutate() for x in self.params]
            self.individuals.append(Individual(new_params, to_string=self.to_string))

        return

    def evolve(self):
        # calculate fitness for all individuals
        self.calc_fitnesses()

        max_fit = max(self.fitnesses)
        min_fit = min(self.fitnesses)
        mean_fit = numpy.mean(self.fitnesses)

        best_ind = self.individuals[int(numpy.argmax(self.fitnesses))]

        # select 2*n individuals in tournaments
        selected = []
        for _ in range(self.size * 2):
            selected.append(self.tournament())

        # generate new individuals
        new_individuals = []
        for i in range(0, self.size * 2, 2):
            new_individuals.append(Individual.crossover(selected[i], selected[i + 1]))

        self.individuals = new_individuals
        return min_fit, mean_fit, max_fit, str(best_ind)

    def calc_fitnesses(self):
        for i in range(self.size):
            self.fitnesses[i] = self.fitness(self.individuals[i])
        return

    def tournament(self):
        candidates = random.sample(range(self.size), k=self.tournament_size)

        best = candidates[0]
        for i in candidates:
            if self.fitnesses[i] > self.fitnesses[best]:
                best = i

        return self.individuals[best]


def guess_word(word='askdjhas', iters=30, size=100, tournament_size=5, plot=True, verbose=True):
    if iters < 1:
        iters = 10000

    params = [Char() for _ in range(len(word))]

    def fitness(ind: Individual):
        guess = str(ind)

        score = 0
        for i in range(len(word)):
            score += word[i] == guess[i]

        return score

    population = Population(
        params,
        fitness=fitness,
        size=size,
        tournament_size=tournament_size,
        to_string=lambda ind: "".join([x.char for x in ind.params])
    )

    mins = []
    means = []
    maxs = []

    for it in range(iters):
        min_f, mean_f, max_f, best_guess = population.evolve()

        if verbose:
            print(f'iteration {it:3}: {best_guess} ({max_f})')

        if plot:
            mins.append(min_f)
            means.append(mean_f)
            maxs.append(max_f)

        if max_f == len(word):
            break

    if verbose:
        print(f'took {it} iterations to find the word using {size} individuals with tournaments of {tournament_size}')

    if plot:
        plt.figure()
        plt.plot(mins, 'r-')
        plt.plot(means, 'b-')
        plt.plot(maxs, 'g-')
        plt.ylim(0, len(word))

        plt.title('Population Evolution')
        plt.xlabel('iteration')
        plt.ylabel('fitness')
        plt.legend(['min', 'mean', 'max'])
        plt.show()

    return it


if __name__ == '__main__':
    user_word = input('word to guess? ')
    guess_word(
        user_word,
        iters=-1,
        size=len(user_word) * 10,
        tournament_size=max(len(user_word) // 2, 5),
    )
