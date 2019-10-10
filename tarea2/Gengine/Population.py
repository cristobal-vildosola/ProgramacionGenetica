import random
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy


class Gene(ABC):
    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class Individual:
    def __init__(self, genes: List[Gene], mutation_rate: float = 0.1):
        self.genes: List[Gene] = genes
        self.mutation_rate = mutation_rate

    def crossover(self, other: 'Individual'):
        index = random.randint(0, len(self.genes) - 1)
        new_genes = self.genes[:index] + other.genes[index:]

        for i in range(len(new_genes)):
            if random.random() < self.mutation_rate:
                new_genes[i] = new_genes[i].mutate()

        return Individual(genes=new_genes, mutation_rate=self.mutation_rate)

    def __str__(self):
        return "".join([str(x) for x in self.genes])


class Population:
    def __init__(self,
                 genes: List[Gene],
                 fitness: Callable[[Individual], float],
                 mutation_rate: float = 0.1,
                 size: int = 100,
                 tournament_size=5,
                 elitism=True):

        self.genes: List[Gene] = genes
        self.mutation_rate = mutation_rate
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
            new_genes = [x.mutate() for x in self.genes]
            self.individuals.append(
                Individual(genes=new_genes, mutation_rate=self.mutation_rate)
            )

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
        for i in range(0, len(selected), 2):
            new_individuals.append(selected[i].crossover(selected[i + 1]))

        if self.elitism:
            new_individuals.append(best_ind)

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
