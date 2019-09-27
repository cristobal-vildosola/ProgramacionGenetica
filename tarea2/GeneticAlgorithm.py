import random
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import numpy
import matplotlib.pyplot as plt


class Gene(ABC):
    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Individual:
    def __init__(self, genes: List[Gene]):
        self.genes: List[Gene] = genes

    def crossover(self, other):
        index = random.randint(0, len(self.genes) - 1)
        new_params = self.genes[:index] + other.genes[index:]

        index = random.randint(0, len(self.genes) - 1)
        new_params[index] = new_params[index].mutate()

        return Individual(new_params)

    def __str__(self):
        return "".join([str(x) for x in self.genes])


class Population:
    def __init__(self,
                 genes: List[Gene],
                 fitness: Callable,
                 size: int = 100,
                 tournament_size=5,
                 elitism=True):

        self.genes: List[Gene] = genes
        self.fitness: Callable = fitness

        self.size: int = size
        self.tournament_size: int = tournament_size
        self.elitism = elitism

        self.individuals: List[Individual] = []
        self.fitnesses: List[float] = numpy.zeros(size)

        self.gen_individuals()

    def gen_individuals(self):
        self.individuals = []

        for _ in range(self.size):
            new_params = [x.mutate() for x in self.genes]
            self.individuals.append(Individual(new_params))

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

        size = (self.size - self.elitism) * 2
        for _ in range(size):
            selected.append(self.tournament())

        # generate new individuals
        new_individuals = []

        if self.elitism:
            new_individuals.append(best_ind)

        for i in range(0, len(selected), 2):
            new_individuals.append(selected[i].crossover(selected[i + 1]))

        self.individuals = new_individuals
        return min_fit, mean_fit, max_fit, best_ind

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


class GeneticAlgorith:
    def __init__(self,
                 genes: List[Gene],
                 fitness: callable,
                 size: int = 100,
                 tournament_size: int = 5,
                 elitism=True):

        self.population = Population(
            genes=genes,
            fitness=fitness,
            size=size,
            tournament_size=tournament_size,
            elitism=elitism
        )

        self.history = {
            'mins': [],
            'means': [],
            'maxs': [],
            'bests': [],
        }

    def evolve(self, iters=30, end_condition: callable = None, verbose: bool = True, to_string=None):
        if iters < 1:
            iters = 999999999

        for it in range(iters):
            min_f, mean_f, max_f, best_ind = self.population.evolve()

            self.history['mins'].append(min_f)
            self.history['means'].append(mean_f)
            self.history['maxs'].append(max_f)
            self.history['bests'].append(best_ind)

            if verbose:
                if to_string:
                    print(f'iteration {it + 1:3}: {to_string(best_ind)} ({max_f})')
                else:
                    print(f'iteration {it + 1:3}: {best_ind} ({max_f})')

            if end_condition is not None and end_condition(self):
                break

        return

    def plot_history(self, fitness_range: Tuple[int, int] = None):
        plt.figure()

        plt.plot(self.history['mins'], 'r-')
        plt.plot(self.history['means'], 'b-')
        plt.plot(self.history['maxs'], 'g-')
        plt.ylim(fitness_range)

        plt.title('Population Evolution')
        plt.xlabel('iteration')
        plt.ylabel('fitness')
        plt.legend(['min', 'mean', 'max'])

        plt.show()
        return
