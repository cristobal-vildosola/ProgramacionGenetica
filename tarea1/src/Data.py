import csv

import numpy


def read_csv(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = numpy.array(list(reader)).astype("float")
    return data


def normalize(data, low=0, high=1):
    maximums = numpy.max(data, axis=0)
    minimums = numpy.min(data, axis=0)

    for i in range(data.shape[1] - 1):
        if minimums[i] == maximums[i]:
            data[:, i] = data[:, i] - minimums[i] + low
        else:
            data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i]) * (high - low) + low

    return data


# separa los datos en las clases dadas
def divide_by_class(data):
    classes = list(set(data[:, -1]))
    n = len(classes)
    groups = [[] for _ in range(n)]

    for row in data:
        for i in range(n):
            if row[-1] == classes[i]:
                groups[i].append(row)

    return groups


# separa los datos en 2 conjuntos manteniendo la proporcion entre las n clases
def separate_data(data, prop):
    # shuffle and divide data by class
    numpy.random.shuffle(data)
    classes_groups = divide_by_class(data)

    group_1 = []
    group_2 = []

    # divide data with the given proportion
    for c in range(len(classes_groups)):
        n = len(classes_groups[c])

        for i in range(n):
            if i < n * prop:
                group_1.append(classes_groups[c][i])
            else:
                group_2.append(classes_groups[c][i])

    # convert to numpy array and shuffle
    group_1 = numpy.array(group_1)
    group_2 = numpy.array(group_2)
    numpy.random.shuffle(group_1)
    numpy.random.shuffle(group_2)

    return [group_1, group_2]
