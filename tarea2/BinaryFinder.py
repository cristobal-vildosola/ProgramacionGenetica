import random

from Gengine import *


class Bit(Gene):
    def __init__(self):
        self.char = random.randint(0, 1)

    def mutate(self):
        return Bit()

    def __str__(self):
        return str(self.char)


def guess_binary(word='011001011010', iters=30, size=100, tournament_size=5):
    genes = [Bit() for _ in range(len(word))]

    def fitness(ind: Individual):
        score = 0
        for i in range(len(word)):
            score += word[i] == str(ind.genes[i])

        return score

    def end_condition(gen_alg):
        return gen_alg.history['maxs'][-1] == len(word)

    genetic_alg = GeneticAlgorith(
        genes=genes,
        fitness=fitness,
        size=size,
        tournament_size=tournament_size
    )

    genetic_alg.evolve(iters=iters, verbose=True, end_condition=end_condition)
    genetic_alg.plot_history()

    return


if __name__ == '__main__':
    user_word = input('bits to guess? ')
    guess_binary(
        user_word,
        iters=100,
        size=len(user_word) * 10,
        tournament_size=max(len(user_word) // 2, 5),
    )
