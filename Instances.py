import copy
import json
import random
import numpy as np
import collections


def generate_goodson_instances(instance_class, customers, vehicles, capacity, durationlimit,
                               stoch_types, env_config=None):
    instances = []

    print("Generate instances:")

    # customers
    c_set = load_solomon_instance("Instances/" + instance_class + "101.txt")[:customers]

    for dlp in durationlimit:
        for cp in capacity:
            for stoch_type in stoch_types:
                v_set = []
                instance_name = instance_class + "101_1_" + str(customers) + "_" + str(vehicles) + \
                                "_" + str(int(dlp)) + "_" + str(cp)
                config = copy.copy(env_config)
                config.Q = cp
                config.m = vehicles
                config.n = customers
                config.dl = dlp
                config.stoch_type = stoch_type

                # Vehicles
                for j in range(config.m):
                    v_set.append([j, config.Depot[0], config.Depot[0], config.Q * 1., 0, config.n])

                instance = {"Customers": c_set, "Vehicles": v_set, "Config": config,
                            "Name": instance_name + "_" + str(random.randint(1000, 9999))}
                instances.append(instance)

    return instances


def generate_vrpsd_instances(n_customers, n_vehicles, capacity, duration_limit,
                             stoch_types, env_config=None):
    code = str(n_customers) + "_" + str(round(duration_limit[0])) + "_" + str(capacity[0]) + "_" \
           + str(random.randint(10000, 99999))

    instances = []

    print("Generate instances:")

    # customers
    c_set = load_solomon_instance("Instances/VRPSD/r101.txt")[:n_customers]

    for dlp in duration_limit:
        for cp in capacity:
            for stoch_type in stoch_types:
                v_set = []
                instance_name = "r101_1_" + str(n_customers) + "_" + str(n_vehicles) + \
                                "_" + str(int(dlp)) + "_" + str(cp)
                config = copy.copy(env_config)
                config.Q = cp
                config.m = n_vehicles
                config.n = n_customers
                config.dl = dlp
                config.stoch_type = stoch_type

                # Vehicles
                for j in range(config.m):
                    v_set.append([j, config.Depot[0], config.Depot[0], config.Q * 1., 0, config.n])

                instance = {"Customers": c_set, "Vehicles": v_set, "Config": config,
                            "Name": instance_name + "_" + str(random.randint(1000, 9999))}
                instances.append(instance)

    return instances, code


def load_solomon_instance(fname):
    # takes and convert all customers in the instance
    file = open(fname, 'r')
    d = file.readlines()

    # line 10 to ,, are customers
    c_set = []

    max_x = 0
    min_x = 100
    max_y = 0
    min_y = 100

    for ct in d[10:]:
        # since customers in the solomon instances are starting from 1, I reduce their id by 1 to start from 0.
        cid, psi, demand = get_coordination_from_text(ct)
        # id, x, y, available, exp_dem, realized, unserved, is_realized, last_demand
        cus = [(cid-1), psi[0], psi[1], 1, demand * 1., -1.0, -1.0, 0., demand * 1.]
        c_set.append(cus)
        max_x = max(max_x, psi[0])
        min_x = min(min_x, psi[0])
        max_y = max(max_y, psi[1])
        min_y = min(min_y, psi[1])

    return c_set


def get_coordination_from_text(s):
    d = [m for m in s.split(" ") if m != '']
    return int(d[0]), [int(d[1]), int(d[2])], int(d[3])


def load_VRPVCSD_instances(fname):
    if "[" in fname:
        fname = fname.replace("[", "")
        fname = fname.replace("]", "")

    with open("Instances/VRPVCSD/" + fname, 'r') as f:
        s = json.load(f)
    random_instances = []

    for e, instance in enumerate(s):
        v_set = np.array(instance["Vehicles"])
        c_set = np.array(instance["Customers"])
        cn = instance["Config"]
        config = collections.namedtuple("ModelConfig", cn.keys())(*cn.values())
        # config = ModelConfig(Q=cn["Q"], m=cn["m"], dl=cn["dl"], xy_length=cn["xy_length"])
        random_instances.append({"Vehicles": v_set, "Customers": c_set, "Config": config, "Name": e})

    return random_instances


def generate_VRPVCSD_instances(config, density_class, capacity, duration_limit, n_vehicles, count):
    """
    Notes:
    1-since the number of realized customers is random, we define the vector of customers with a fixed size equal to
    1.2*|nbar|. In case of having fewer realized customers, we fill the vector with dummy customers.
    2- the last index in the set of customers (nodes) always refers to the depot.

    :param config: the characteristics of the environment
    :param density_class: the density level of customers (low, moderate, and high)
    :param n_vehicles: the number of vehicles
    :param duration_limit: the duration limit
    :param capacity: the max capacity of vehicles
    :param count: the number of instances to be generated
    :return: a set of instances
    """

    code = str(density_class) + "_" + str(capacity) + "_" + str(random.randint(10000, 99999))

    config.Q = capacity
    config.dl = duration_limit
    config.m = n_vehicles
    config.stoch_type = 0

    # use big heatmaps to generate instances
    heatmap = [[1, 1, 0, 1, 0],
               [1, 1, 0, 0, 1],
               [1, 0, 1, 1, 0],
               [0, 1, 0, 1, 1],
               [0, 1, 1, 1, 0]]
    instances = []

    max_c_number_rate = 1.2
    new_nbar_list = np.floor([23 * max_c_number_rate, 53 * max_c_number_rate, 83 * max_c_number_rate])
    nbar = int(new_nbar_list[density_class])

    #   the probability distribution for the number of customers at each zone
    n_probs = [0.1, 0.4, 0.4, 0.1]
    if density_class == 0:
        n_numbers = [0, 1, 2, 3]
    elif density_class == 1:
        n_numbers = [2, 3, 4, 5]
    elif density_class == 2:
        n_numbers = [4, 5, 6, 7]
    else:
        raise Exception("Unknown density class")

    #   the set of vehicles [index, l_x, l_y, q, a, occupied_node]
    v_set = []
    # vehicles index 0, 1, ..., m-1
    for j in range(config.m):
        v_set.append([j, config.Depot[0], config.Depot[0], config.Q * 1., 0, nbar])

    #   the distribution function for the expected demands (uniform)
    exp_demands = [5, 10, 15]

    il = len(heatmap)
    jl = len(heatmap[0])
    for _ in range(count):
        # Generate a set of customers (location + expected demand)
        c_set = []
        realized_pos = [(50, 50)]
        c_count = 0
        #   enumerate over partitions of the heatmap and generate a random number of customers for eah that is active.
        for i in range(il):
            #   make sure the number of realized customers does not exceed the nbar.
            if c_count >= nbar:
                break

            for j in range(jl):

                if heatmap[i][j] == 0:
                    continue
                if c_count >= nbar:
                    break

                #   generate n_z
                n_z = np.random.choice(n_numbers, 1, p=n_probs)[0]
                for c in range(n_z):
                    x_coord = random.randint(j * 20 + 1, (j + 1) * 20)
                    y_coord = random.randint(i * 20 + 1, (i + 1) * 20)

                    #   make sure no two customers request from exactly the same location
                    while (x_coord, y_coord) in realized_pos:
                        x_coord = random.randint(j * 20 + 1, (j + 1) * 20)
                        y_coord = random.randint(i * 20 + 1, (i + 1) * 20)
                    realized_pos.append((x_coord, y_coord))

                    #   randomly assign an expected demand to the realized location
                    exp_demand = random.choice(exp_demands)

                    #   construct the customers raw feature set
                    # [id, l_x, l_y, h, \bar{d}, realized demand, \hat{d}, is_realized, \tilde{d}]
                    c_set.append([c_count, x_coord, y_coord, 1, exp_demand, -1, -1, 0, exp_demand])
                    c_count += 1

                    if c_count >= nbar:
                        break

        #   fill the c_set with dummy customers
        for i in range(nbar - c_count):
            c_set.append([c_count + i, 0, 0, 0, 0, 0, 0, 0, 0])

        #   set the instance config
        config = copy.deepcopy(config)
        config.n = len(c_set)
        config.real_n = c_count
        inst_name = "I_" + str(density_class) + "_" + str(config.m) + "_" + str(config.Q) + "_" + \
                    str(random.randint(100000, 999999))
        instance = {"Vehicles": np.array(v_set), "Customers": np.array(c_set).astype(float), "Config": config,
                    "Name": inst_name}
        instances.append(instance)

    return instances, code


def generate_VRPVCSD_instances_generalized(config, density_class_list, capacity_list, count,
                                          max_c_size=None, max_v_size=None):
    """
    Notes:
    1-since the number of realized customers is random, we define the vector of customers with a fixed size equal to
    1.2*|nbar|. In case of having fewer realized customers, we fill the vector with dummy customers.
    2- the last index in the set of customers (nodes) always refers to the depot.

    :param config: the characteristics of the environment
    :param density_class: the density level of customers (low, moderate, and high)
    :param n_vehicles: the number of vehicles
    :param duration_limit: the duration limit
    :param capacity: the max capacity of vehicles
    :param count: the number of instances to be generated
    :return: a set of instances
    """
    # use big heatmaps to generate instances
    heatmap = [[1, 1, 0, 1, 0],
               [1, 1, 0, 0, 1],
               [1, 0, 1, 1, 0],
               [0, 1, 0, 1, 1],
               [0, 1, 1, 1, 0]]
    instances = []

    nbar_list = [23, 53, 83]
    m_list = [3, 7, 11]
    L_list = [221.47, 195.54, 187.29, 9999999]
    n_probs = [0.1, 0.4, 0.4, 0.1]
    n_numbers_list = [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]
    #   the distribution function for the expected demands (uniform)
    exp_demands = [5, 10, 15]

    if max_c_size is None:
        max_c_size = int(1.2 * nbar_list[int(max(density_class_list))])
    if max_v_size is None:
        max_v_size = m_list[int(max(density_class_list))]

    for _ in range(count):
        density_class = random.choice(density_class_list)
        capacity = random.choice(capacity_list)

        #   set the instance config
        config = copy.deepcopy(config)

        config.Q = capacity
        config.dl = L_list[density_class] + 0.
        config.m = int(m_list[density_class] + 0.)
        nbar = int(nbar_list[density_class])
        n_numbers = n_numbers_list[density_class]

        #   the set of vehicles [index, l_x, l_y, q, a, occupied_node]
        v_set = np.zeros([max_v_size, 6])
        # vehicles index 0, 1, ..., m-1
        for j in range(config.m):
            v_set[j] = [j, config.Depot[0], config.Depot[0], config.Q * 1., 0, max_c_size]

        il = len(heatmap)
        jl = len(heatmap[0])

        # Generate a set of customers (location + expected demand)
        c_set = np.zeros([max_c_size, 9])
        realized_pos = [(50, 50)]
        c_count = 0
        n_cust_limit = min(nbar * 1.2, max_c_size)
        #   enumerate over partitions of the heatmap and generate a random number of customers for eah that is active.
        for i in range(il):
            #   make sure the number of realized customers does not exceed the nbar.
            if c_count >= n_cust_limit:
                break

            for j in range(jl):

                if heatmap[i][j] == 0:
                    continue
                if c_count >= n_cust_limit:
                    break

                #   generate n_z
                n_z = np.random.choice(n_numbers, 1, p=n_probs)[0]
                for c in range(n_z):
                    x_coord = random.randint(j * 20 + 1, (j + 1) * 20)
                    y_coord = random.randint(i * 20 + 1, (i + 1) * 20)

                    #   make sure no two customers request from exactly the same location
                    while (x_coord, y_coord) in realized_pos:
                        x_coord = random.randint(j * 20 + 1, (j + 1) * 20)
                        y_coord = random.randint(i * 20 + 1, (i + 1) * 20)
                    realized_pos.append((x_coord, y_coord))

                    #   randomly assign an expected demand to the realized location
                    exp_demand = random.choice(exp_demands)

                    #   construct the customers raw feature set
                    # [id, l_x, l_y, h, \bar{d}, realized demand, \hat{d}, is_realized, \tilde{d}]
                    c_set[c_count] = [c_count, x_coord, y_coord, 1, exp_demand, -1, -1, 0, exp_demand]
                    c_count += 1

                    if c_count >= n_cust_limit:
                        break

        config.n = len(c_set)
        config.real_n = c_count
        inst_name = "I_" + str(density_class) + "_" + str(config.m) + "_" + str(config.Q) + "_" + \
                    str(random.randint(100000, 999999))
        instance = {"Vehicles": np.array(v_set), "Customers": np.array(c_set).astype(float), "Config": config,
                    "Name": inst_name}
        instances.append(instance)

    return instances


class ModelConfig:
    def __init__(self, Q=10, m=2, dl=100, n=10, stoch_type=0, depot=(0, 0), service_area=(100, 100)):

        self.service_area = service_area

        # Max capacity of each vehicle
        self.Q = Q
        # number of vehicles
        self.m = m
        # Vehicles Duration limit
        self.dl = dl

        # the type of stochasticity, low, moderate, high
        self.stoch_type = stoch_type

        # number of customers
        self.n = n

        self.Depot = depot
