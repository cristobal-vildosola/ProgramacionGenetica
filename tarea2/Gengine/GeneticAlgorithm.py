from typing import List, Tuple, Callable

import matplotlib.pyplot as plt

from Gengine.Population import Population, Gene


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

    def evolve(self, iters=30, end_condition: Callable[['GeneticAlgorith'], bool] = None,
               verbose: bool = True, show_best=None):
        if iters < 1:
            iters = 999999999

        for it in range(iters):
            min_f, mean_f, max_f, best_ind = self.population.evolve()

            self.history['mins'].append(min_f)
            self.history['means'].append(mean_f)
            self.history['maxs'].append(max_f)
            self.history['bests'].append(best_ind)

            if verbose:
                if show_best:
                    print(f'iteration {it + 1:3}: {show_best(best_ind)} ({max_f})')
                else:
                    print(f'iteration {it + 1:3}: {best_ind} ({max_f})')

            if end_condition is not None and end_condition(self):
                break

        return

    def plot_history(self, fitness_range: Tuple[int, int] = None):
        plt.figure()

        plt.plot(self.history['maxs'], 'g-')
        plt.plot(self.history['means'], 'b-')
        plt.plot(self.history['mins'], 'r-')
        plt.ylim(fitness_range)

        plt.title('Population Evolution')
        plt.xlabel('iteration')
        plt.ylabel('fitness')
        plt.legend(['max', 'mean', 'min'])

        plt.show()
        return
