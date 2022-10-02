import math
import random
import types
import numpy as np


def greedy_solution(customers, distance_table, avg_length):
    # customers: list of customers' id
    dist_to_depot = distance_table[-1][customers]

    selected_customers = np.argsort(dist_to_depot)[:avg_length]

    return selected_customers


def generate_neighbor(customers: list, selected_customers: list,
                      operator_probs=(0.5, 0.25, 0.25)):
    '''
    considers three operators:
    1- swap
    2- add
    3- remove
    '''

    outsourced = list(set(customers) - set(selected_customers))

    c1 = selected_customers[int(math.floor(random.random() * len(selected_customers)))]
    c2 = outsourced[int(math.floor(random.random() * len(outsourced)))]

    rnd_operator = np.random.choice([1, 2, 3], size=1, p=operator_probs)
    if rnd_operator == 1:
        # swap c1 and c2
        selected_customers.remove(c1)
        selected_customers.append(c2)
    elif rnd_operator == 2:
        # add c2
        selected_customers.append(c2)
    else:
        # remove c1
        selected_customers.remove(c1)

    return selected_customers

