import copy
import random
from typing import List

import matplotlib.pyplot as plt
import numpy

from Gengine import *


class Step(Gene):
    def __init__(self):
        self.direction = random.randint(0, 3)

    def mutate(self):
        return Step()

    def __str__(self):
        return '<^>v'[self.direction]


# Maze utilities
W = -1.  # wall
S = 1.  # start
E = 2.  # end
dxs = (-1, 0, 1, 0)
dys = (0, -1, 0, 1)


def find_num(maze: List[List[float]], number: float) -> List[int]:
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == number:
                return [i, j]

    return [-1, -1]


def distance(maze: List[List[float]], start: List[int]) -> float:
    # DFS
    dist = numpy.zeros_like(maze)
    visited = numpy.zeros_like(maze)
    queue = [start]
    visited[start[0]][start[1]] = 1

    while len(queue) > 0:
        pos = queue.pop(0)
        length = dist[pos[0]][pos[1]]

        # detect end
        if maze[pos[0]][pos[1]] == E:
            return length

        # explore neighbours
        for i in range(4):
            new_pos = [pos[0] + dys[i], pos[1] + dxs[i]]
            if visited[new_pos[0]][new_pos[1]] == 0 and maze[new_pos[0]][new_pos[1]] != W:
                # mark as visited, update length and add to queue
                dist[new_pos[0]][new_pos[1]] = length + 1
                visited[new_pos[0]][new_pos[1]] = 1
                queue.append(new_pos)

    return -1.


def simulate_path(maze: List[List[float]], genes: List[Step]):
    pos = find_num(maze, S)
    length = 0

    # simulate path
    for step in genes:
        dy = dys[step.direction]
        dx = dxs[step.direction]

        # advance only when valid
        if maze[pos[0] + dy][pos[1] + dx] != W:
            pos[0] += dy
            pos[1] += dx

            length += 1

        # detect end
        if maze[pos[0]][pos[1]] == E:
            break

    dist = distance(maze, pos)
    return dist, length


def mark_path(maze: List[List[float]], genes: List[Step]) -> [List[Step], List[List[float]]]:
    maze_copy = copy.deepcopy(maze)
    pos = find_num(maze, S)
    path = []

    # simulate path
    for step in genes:
        dy = dys[step.direction]
        dx = dxs[step.direction]

        # advance only when valid
        if maze[pos[0] + dy][pos[1] + dx] != W:
            pos[0] += dy
            pos[1] += dx

            maze_copy[pos[0]][pos[1]] = max(S, maze_copy[pos[0]][pos[1]] + 0.2)
            path.append(step)

        # detect end
        if maze[pos[0]][pos[1]] == E:
            break

    # mark end of path
    maze_copy[pos[0]][pos[1]] = S + (E - S) * 3 / 4
    return path, maze_copy


def guess_path(
        maze: List[List[float]],
        iters: int = 30,
        size: int = 100,
        mutation_rate=0.5,
        tournament_size: int = 5
):
    # show maze
    img = plt.imshow(maze)
    plt.set_cmap('viridis')
    plt.axis('off')
    plt.waitforbuttonpress(.3)

    # calculate min path length
    start = find_num(maze, S)
    min_length = distance(maze, start)
    print(f'Minimum path length: {min_length}')

    # initialize GA
    genes = [Step() for _ in range(int(min_length * 2))]

    def fitness(ind: Individual):
        dist, length = simulate_path(maze, ind.genes)
        return -(dist * min_length + length)

    genetic_alg = GeneticAlgorith(
        genes=genes,
        fitness=fitness,
        mutation_rate=mutation_rate,
        size=size,
        tournament_size=tournament_size
    )

    # evolve population
    def end_condition(gen_alg: GeneticAlgorith) -> bool:
        # end if there is no change in last n iterations
        n = 40
        last_iters = gen_alg.history['maxs'][-n:]
        if len(last_iters) == n and max(last_iters) == min(last_iters):
            return True

        # end when the optimum is found
        return int(gen_alg.history['maxs'][-1]) == -min_length

    def show_best(ind: Individual) -> str:
        path, maze_copy = mark_path(maze, ind.genes)

        plt.pause(.4)
        img.set_data(maze_copy)

        return "".join([str(x) for x in path])

    genetic_alg.evolve(iters=iters, verbose=True, end_condition=end_condition, show_best=show_best)
    genetic_alg.plot_history()
    return


def main():
    maze = [
        [W, W, W, W, W, W, W, W, W, W],
        [W, S, 0, 0, 0, 0, 0, 0, 0, W],
        [W, 0, W, W, W, W, 0, W, 0, W],
        [W, 0, 0, W, 0, 0, 0, W, 0, W],
        [W, W, 0, W, 0, W, W, W, 0, W],
        [W, 0, 0, W, 0, W, 0, 0, 0, W],
        [W, 0, W, 0, 0, W, 0, W, 0, W],
        [W, 0, W, 0, W, W, W, W, W, W],
        [W, 0, W, 0, 0, 0, 0, 0, E, W],
        [W, W, W, W, W, W, W, W, W, W],
    ]

    random.seed('holi')

    guess_path(
        maze,
        iters=-1,
        size=100,
        mutation_rate=0.5,
    )


if __name__ == '__main__':
    main()
