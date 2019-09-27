import copy
import random

import matplotlib.pyplot as plt

from GeneticAlgorithm import Gene, GeneticAlgorith, Individual


class Step(Gene):
    def __init__(self):
        self.direction = random.randint(0, 3)

    def mutate(self):
        return Step()

    def __str__(self):
        return '<^>v'[self.direction]


dxs = [-1, 0, 1, 0]
dys = [0, -1, 0, 1]


def find_num(maze, number):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == number:
                return [i, j]

    return None


def simulate_path(maze, genes):
    pos = find_num(maze, 2)
    end = find_num(maze, 4)

    distance = 0
    path = []
    for step in genes:
        dy = dys[step.direction]
        dx = dxs[step.direction]

        if maze[pos[0] + dy][pos[1] + dx] != 1:
            pos[0] += dy
            pos[1] += dx

            distance += 1
            path.append(step)

        if maze[pos[0]][pos[1]] == 4:
            break

    distance_to_end = abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    return distance_to_end, distance


def show_path(maze, genes):
    pos = [0, 0]
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 2:
                pos = [i, j]

    path = []
    maze_copy = copy.deepcopy(maze)

    for step in genes:
        dy = dys[step.direction]
        dx = dxs[step.direction]

        if maze[pos[0] + dy][pos[1] + dx] != 1:
            pos[0] += dy
            pos[1] += dx

            maze_copy[pos[0]][pos[1]] = max(3, maze_copy[pos[0]][pos[1]] + 0.2)
            path.append(step)

        if maze[pos[0]][pos[1]] == 4:
            break

    return path, maze_copy


def guess_path(maze, iters=30, size=100, tournament_size=5):
    # show maze
    img = plt.imshow(maze)
    plt.set_cmap('gist_rainbow')
    plt.axis('off')
    plt.pause(.01)

    # calculate min path length
    min_length = 0

    # DFS
    start = find_num(maze, 2)
    dfs = copy.deepcopy(maze)

    dfs[start[0]][start[1]] = 0
    queue = [start]

    while len(queue) > 0:
        pos = queue.pop(0)
        length = dfs[pos[0]][pos[1]]

        # explore neighbours
        for i in range(4):
            new_pos = [pos[0] + dys[i], pos[1] + dxs[i]]

            if dfs[new_pos[0]][new_pos[1]] == 0:
                # use negative numbers to avoid collission with exit
                dfs[new_pos[0]][new_pos[1]] = length - 1
                queue.append(new_pos)

            if dfs[new_pos[0]][new_pos[1]] == 4:
                # when exit is reached set length and end search
                min_length = -length + 1.0
                queue.clear()
                break

    print(f'Minimum path length: {min_length}')

    # initialize GA
    genes = [Step() for _ in range(int(min_length) * 4)]

    def fitness(ind: Individual):
        distance_to_end, distance = simulate_path(maze, ind.genes)
        return -distance_to_end * 100 - distance

    def end_condition(gen_alg):
        # end if there is no change in last n iterations
        n = 40
        last_iters = gen_alg.history['maxs'][-n:]
        if len(last_iters) == n and max(last_iters) == min(last_iters):
            return True

        # end when the optimum is found
        return gen_alg.history['maxs'][-1] == -min_length

    genetic_alg = GeneticAlgorith(
        genes=genes,
        fitness=fitness,
        size=size,
        tournament_size=tournament_size
    )

    # evolve GA
    def to_string(ind):
        path, maze_copy = show_path(maze, ind.genes)

        plt.pause(.3)
        img.set_data(maze_copy)
        plt.title(f'best path')

        return "".join([str(x) for x in path])

    genetic_alg.evolve(iters=iters, verbose=True, end_condition=end_condition, to_string=to_string)
    genetic_alg.plot_history()

    return


def main():
    maze = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 4, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    guess_path(
        maze,
        iters=-1,
        size=100,
        tournament_size=20,
    )


if __name__ == '__main__':
    main()
