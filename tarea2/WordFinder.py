import random
import string
from typing import List, Union, Tuple

from src.GeneticAlgorithm import GeneticAlgorith
from src.Population import Gene, Individual


class Char(Gene):
    def __init__(self, alphabet: Union[List[str], Tuple[str, ...]] = tuple(string.printable[:-5])):
        self.alphabet = alphabet
        self.char = random.choice(alphabet)

    def mutate(self):
        return Char(self.alphabet)

    def __str__(self):
        return self.char


def guess_word(word='askdjhas', iters=30, size=100, tournament_size=5):
    genes = [Char() for _ in range(len(word))]

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
    user_word = input('word to guess? ')
    guess_word(
        user_word,
        iters=-1,
        size=len(user_word) * 10,
        tournament_size=max(len(user_word) // 2, 5),
    )
