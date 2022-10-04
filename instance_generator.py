import copy
import json
import random
import numpy as np
import collections
import Utils


def load_solomon_instance(fname):
    # takes and convert all customers in the instance
    with open(fname, 'r') as file:
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

    return np.asarray(c_set)


def generate_vrpsd_instances(n_customers, n_vehicles, capacity, duration_limit,
                             stoch_types, instance_config=None, normalize=True):
    code = str(n_customers) + "_" + str(round(duration_limit[0])) + "_" + str(capacity[0]) + "_" \
           + str(random.randint(10000, 99999))

    instances = []

    print("Generate instances:")

    # customers
    c_set = load_solomon_instance("Instances/VRPSD/r101.txt")[:n_customers]
    if normalize:
        c_set[:, [1, 2]] /= Utils.Norms.COORD
        instance_config.depot[0] /= Utils.Norms.COORD
        instance_config.depot[1] /= Utils.Norms.COORD

    for dlp in duration_limit:
        for cp in capacity:
            if normalize:
                c_set_n = np.array(c_set)
                c_set_n[:, [4, -1]] /= Utils.Norms.Q

            else:
                c_set_n = c_set

            for stoch_type in stoch_types:
                v_set = []
                instance_name = "r101_1_" + str(n_customers) + "_" + str(n_vehicles) + \
                                "_" + str(int(dlp)) + "_" + str(cp)
                config = copy.copy(instance_config)
                config.capacity = cp
                config.m = n_vehicles
                config.n = n_customers
                if normalize:
                    config.duration_limit = dlp / Utils.Norms.COORD
                else:
                    config.duration_limit = dlp
                config.stoch_type = stoch_type

                # Vehicles
                if normalize:
                    q = config.capacity / Utils.Norms.Q
                else:
                    q = config.capacity
                for j in range(config.m):
                    v_set.append([j, config.depot[0], config.depot[0], q, 0, config.n])

                instance = {"Customers": c_set_n, "Vehicles": v_set, "Config": config,
                            "Name": instance_name + "_" + str(random.randint(1000, 9999))}
                instances.append(instance)

    return instances


def get_coordination_from_text(s):
    d = [m for m in s.split(" ") if m != '']
    return int(d[0]), [int(d[1]), int(d[2])], int(d[3])


def load_vrpscd_instances(fname):
    if "[" in fname:
        fname = fname.replace("[", "")
        fname = fname.replace("]", "")

    with open("Instances/VRPSCD/" + fname, 'r') as f:
        s = json.load(f)
    random_instances = []

    for e, instance in enumerate(s):
        v_set = np.array(instance["Vehicles"])
        c_set = np.array(instance["Customers"])

        # Normalize
        v_set /= [1, Utils.Norms.COORD, Utils.Norms.COORD, Utils.Norms.Q, 1, 1]
        c_set /= [1, Utils.Norms.COORD, Utils.Norms.COORD, 1, Utils.Norms.Q, 1, 1, 1, Utils.Norms.Q]

        cn = instance["Config"]
        config = InstanceConfig(**cn)
        config.density_class = int(str.split(fname, "_")[0])

        config.depot[0] /= Utils.Norms.COORD
        config.depot[1] /= Utils.Norms.COORD
        config.duration_limit /= Utils.Norms.COORD

        c_set[config.real_n:, 0] = 0

        # config.real_n = len(c_set)
        # config.duration_limit = cn["duration_limit"]
        # config.m = cn["m"]
        # config.n = len(c_set)
        # config.capacity = cn["capacity"]
        # config = collections.namedtuple("InstanceConfig", cn.keys())(*cn.values())
        # config = ModelConfig(Q=cn["Q"], m=cn["m"], dl=cn["dl"], xy_length=cn["xy_length"])
        random_instances.append({"Vehicles": v_set, "Customers": c_set, "Config": config, "Name": e})

    return random_instances


def generate_vrpscd_instances_generalized(instance_config, density_class_list, capacity_list, count,
                                          max_c_size=None, max_v_size=None, normalize=True):
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

    nbar_list = [23, 53, 83, 10, 15]
    m_list = [3, 7, 11, 2, 2]
    L_list = [221.47, 195.54, 187.29, 143.71, 201.38]
    n_probs_list = [[0.1, 0.4, 0.4, 0.1], [0.1, 0.4, 0.4, 0.1], [0.1, 0.4, 0.4, 0.1],
                    [1 / 2., 1 / 3., 1 / 6.], [1 / 3., 1 / 3., 1 / 3.]]
    n_numbers_list = [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [0, 1, 2], [0, 1, 2]]
    #   the distribution function for the expected demands (uniform)
    exp_demands = [5, 10, 15]

    if max_c_size is None:
        max_c_size = int(1.2 * nbar_list[int(max(density_class_list))])
    if max_v_size is None:
        max_v_size = m_list[int(max(density_class_list))]

    if normalize:
        instance_config.depot[0] /= Utils.Norms.COORD
        instance_config.depot[1] /= Utils.Norms.COORD

    for _ in range(count):
        density_class = random.choice(density_class_list)
        capacity = random.choice(capacity_list)

        #   set the instance config
        config = copy.deepcopy(instance_config)

        config.capacity = capacity
        config.duration_limit = L_list[density_class] + 0.
        if normalize:
            config.duration_limit /= Utils.Norms.DL

        config.real_duration_limit = L_list[density_class] + 0.
        config.m = int(m_list[density_class] + 0.)
        nbar = int(nbar_list[density_class])
        n_numbers = n_numbers_list[density_class]
        n_probs = n_probs_list[density_class]

        #   the set of vehicles [index, l_x, l_y, q, a, occupied_node]
        v_set = np.zeros([max_v_size, 6])
        # vehicles index 0, 1, ..., m-1
        q_coef = 1. / config.capacity if normalize else 1.
        for j in range(config.m):
            v_set[j] = [j, config.depot[0], config.depot[0], config.capacity * q_coef, 0, max_c_size]

        il = len(heatmap)
        jl = len(heatmap[0])

        # Generate a set of customers (location + expected demand)
        c_set = np.zeros([max_c_size + 1, 9])
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
                    x_coord = random.randint(j * 20 + 1, (j + 1) * 20 - 1)
                    y_coord = random.randint(i * 20 + 1, (i + 1) * 20 - 1)

                    #   make sure no two customers request from exactly the same location
                    while (x_coord, y_coord) in realized_pos:
                        x_coord = random.randint(j * 20 + 1, (j + 1) * 20 - 1)
                        y_coord = random.randint(i * 20 + 1, (i + 1) * 20 - 1)
                    realized_pos.append((x_coord, y_coord))

                    #   randomly assign an expected demand to the realized location
                    exp_demand = random.choice(exp_demands)

                    if normalize:
                        x_coord /= Utils.Norms.COORD
                        y_coord /= Utils.Norms.COORD
                        exp_demand /= config.capacity

                    #   construct the customers raw feature set
                    # [id, l_x, l_y, h, \bar{d}, realized demand, \hat{d}, is_realized, \tilde{d}]
                    c_set[c_count] = [c_count, x_coord, y_coord, 1, exp_demand, -1, -1, 0, exp_demand]
                    c_count += 1

                    if c_count >= n_cust_limit:
                        break

        config.n = len(c_set) - 1
        config.real_n = c_count
        c_set[-1] = [config.n, 50 / Utils.Norms.COORD, 50 / Utils.Norms.COORD, 0, 0, 0, 0, 0, 0]
        inst_name = "I_" + str(density_class) + "_" + str(config.m) + "_" + str(config.capacity) + "_" + \
                    str(random.randint(100000, 999999))
        instance = {"Vehicles": np.array(v_set), "Customers": np.array(c_set).astype(float), "Config": config,
                    "Name": inst_name}
        instances.append(instance)

    return instances


class InstanceConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        v = vars(self)
        return ', '.join("%s: %s" % item for item in v.items())


class EnvConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        v = vars(self)
        return ', '.join("%s: %s" % item for item in v.items())


class GenConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        v = vars(self)
        return ', '.join("%s: %s" % item for item in v.items())


class RLConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __str__(self):
        v = vars(self)
        return ', '.join("%s: %s" % item for item in v.items())


