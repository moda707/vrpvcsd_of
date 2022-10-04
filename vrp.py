import copy
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
import Utils
from instance_generator import InstanceConfig


class VRPSD(object):
    def __init__(self, env_config):
        self.vehicles = None
        self.customers = None
        self.model_type = env_config.model_type
        # self.c_enc = None
        # self.v_enc = None

        self.time = 0
        self.env_config = env_config
        self.ins_config = None

        self.demand_scenario = None

        self.actions = {}
        self.TT = []
        self.final_reward = 0

        self.all_distance_tables = {}
        self.distance_table = None

        self.all_heatmap_ids = {}
        self.heatmap_ids = None

        self.demand_prob = []
        self.demand_val = []

        if self.model_type == "VRPSD":
            self.demand_realization = self.demand_realization_solomon
        else:
            self.demand_realization = self.demand_realization_gendreau

    def init_encoded_env(self):
        norm = Utils.Norms()
        n = self.ins_config.n

        #   features set for customers
        # is_realized: if the actual demand is realized, \tild{d}=\bar{d} if \hat{d}=-1 else \hat{d},
        # is_customer: indicates that the node is a customer and not a depot,
        # is_target: whether it is in the set of target customers or not
        # l_x, l_y, is_realized, \tild{d}, is_customer, is_target
        # c_set = self.customers[:, [1, 2, 7, 8, 8, 8]]
        # c_set[:, -2:] = [1., 0.]

        # l_x, l_y, h, bar{d}, hat{d}
        c_set = self.customers[:, [1, 2, 3, 4, 5]]
        # c_set[c_set[:, 0] > 0.0001, -2:] = [1., 0.]

        # add a node as the depot
        # depot = np.array([self.ins_config.depot[0], self.ins_config.depot[1],
        #                   1., 0., -1, 0])
        depot = np.array([self.ins_config.depot[0], self.ins_config.depot[1],
                          1., 0., 0])

        # add a dummy node at the end
        # dummy = np.zeros(6)
        dummy = np.zeros(5)

        c_set = np.vstack([c_set, depot, dummy])

        #   features set fo vehicles
        #   d_exp: expected demand to serve at its destination, loc_depot: whether it is located at the depot
        # For v: x, y, q, a, d_exp, loc_depot
        v_set = []
        for v in self.vehicles:
            exp_dem = 0
            loc_depot = 1
            l = int(v[-1])
            if l != n:
                exp_dem = self.customers[l, -1]
                loc_depot = 0
            v_set.append([v[1], v[2], v[3], v[4] - self.time, exp_dem, loc_depot])

        v_set = np.array(v_set)

        #   instance characteristics - used in the generalized DecDQN
        instance_chars = [self.ins_config.duration_limit, self.ins_config.capacity / norm.Q, self.ins_config.m,
                          0, 0, 0]
        if self.model_type == "VRPSD":
            instance_chars[-3 + self.ins_config.stoch_type] = 1.
        else:
            instance_chars[-3 + self.ins_config.density_class] = 1.

        return c_set, v_set, instance_chars

    def initialize_environment(self, instance):
        self.customers = np.array(instance["Customers"])
        self.vehicles = np.array(instance["Vehicles"])
        self.ins_config = instance["Config"]

        self.update_distance_table(instance["Name"])
        self.update_heatmap_ids(instance["Name"])

    def reset(self, instance=None, reset_distance=True, scenario=None):
        self.time = 0

        if instance is None:
            # don't change the customers realization
            # update availability, realized demand, unserved demand, is realized
            self.customers[:, [3, 5, 6, 7]] = [1, -1, -1, 0]
            # update the last demand to exp dem
            self.customers[:, -1] = self.customers[:, 4]

            self.vehicles[:, 1:] = [self.ins_config.depot[0], self.ins_config.depot[1],
                                    self.ins_config.capacity / Utils.Norms.Q, 0,
                                    self.ins_config.n]

        else:
            self.customers = np.array(instance["Customers"])
            self.vehicles = np.array(instance["Vehicles"])
            self.ins_config = instance["Config"]

        if reset_distance:
            # when the set of customers might be changed, the distance tables will be changed here,
            # generate them and add to the memory if it is not generated yet, otherwise, use the memory
            self.update_distance_table(instance["Name"])

        if scenario is None:
            self.demand_scenario = -np.ones(self.ins_config.n)
        else:
            self.demand_scenario = scenario

        self.actions = {}
        for m in range(self.ins_config.m):
            self.actions[m] = [-2]

        # self.c_enc, self.v_enc, self.instance_chars = self.init_encoded_env()

        if self.model_type == "VRPSD":
            if self.ins_config.stoch_type == 0:
                self.demand_prob = [0.05, 0.9, 0.05]
                self.demand_val = [0.5, 1., 1.5]
            elif self.ins_config.stoch_type == 1:
                self.demand_prob = [0.05, 0.15, 0.6, 0.15, 0.05]
                self.demand_val = [0, 0.5, 1., 1.5, 2.]
            else:
                self.demand_prob = [0.2, 0.2, 0.2, 0.2, 0.2]
                self.demand_val = [0, 0.5, 1., 1.5, 2.]
        else:
            self.demand_prob = [0.05, 0.15, 0.6, 0.15, 0.05]
            self.demand_val = [0, 0.5, 1., 1.5, 2.]

    def update_distance_table(self, instance_name):
        if instance_name in self.all_distance_tables:
            self.distance_table = squareform(self.all_distance_tables[instance_name])
        else:
            # generate distance table
            pos_list = list(self.customers[:, 1:3])
            # pos_list.append(list(self.ins_config.depot))
            distance_table = pdist(np.array(pos_list))
            self.all_distance_tables[instance_name] = distance_table
            self.distance_table = squareform(distance_table)

    def update_heatmap_ids(self, instance_name):
        if instance_name in self.all_heatmap_ids:
            self.heatmap_ids = self.all_heatmap_ids[instance_name]
        else:
            # generate heatmap ids
            # hm_slice is the length of each partition, here we descretize the locations
            slices = np.floor((self.customers[:self.ins_config.real_n, 1:3] - [0.001, 0.001]) / self.env_config.hm_slice)
            # make an id for each pair of descretized xy
            ids = [int(i[0] / self.env_config.hm_slice[0] + i[1]) for i in slices]
            self.all_heatmap_ids[instance_name] = ids
            self.heatmap_ids = ids

    def demand_realization_solomon(self, cid):
        realized_demand = self.demand_scenario[cid]

        # if demand_scenario[cid] == -1, generate a new value, otherwise, return that

        if realized_demand >= 0:
            return realized_demand

        current_c = self.customers[cid]
        exp_demand = current_c[4]

        if self.ins_config.stoch_type == -1:
            return exp_demand

        op = [[0.5, 1.0, 1.5], [0, 0.5, 1.0, 1.5, 2.0], [0, 0.5, 1.0, 1.5, 2.0]]
        pr = [[0.05, 0.9, 0.05], [0.05, 0.15, 0.6, 0.15, 0.05], [0.2, 0.2, 0.2, 0.2, 0.2]]

        realized_demand = np.random.choice(op[self.ins_config.stoch_type],
                                           p=pr[self.ins_config.stoch_type]) * exp_demand

        return realized_demand

    def demand_realization_gendreau(self, cid):
        realized_demand = self.demand_scenario[cid]

        if realized_demand >= 0:
            return realized_demand

        current_c = self.customers[cid]
        if current_c[-1] == 0:
            return 0
        exp_demand = current_c[4]
        return self.generate_dem_real_gendreau(exp_demand)

    def generate_dem_real_gendreau(self, exp_demand):
        if exp_demand == 0:
            return 0
        if exp_demand == 5 / Utils.Norms.Q:
            return exp_demand + random.choice(list(range(-4, 5))) / Utils.Norms.Q
        else:
            return exp_demand + random.choice(list(range(-5, 6))) / Utils.Norms.Q

    def post_decision(self, x, k):
        # in this function, the current state transits to the post decision state.
        # it means, action x, only blocks customer x temporary to not be served by any other vehicles and
        # updates the position and the arrival time of the vehicle k to respectively x and get_distance(l_k, x)
        depot = self.ins_config.depot
        n = self.ins_config.n
        # norm = Utils.Norms()
        v_k = self.vehicles[k]
        q = v_k[3]
        # psi = v_k[1:3]
        # at = v_k[4]
        loc_id = int(v_k[5])

        # exp_serve_dem = 0
        # x = depot
        if x == n:
            # if the vehicle is located at the depot
            if loc_id == n:
                # If it is located at the depot and again selects depot, it is the terminal
                # set the arrival time
                at = self.ins_config.duration_limit
            else:
                travel_time = self.distance_table[loc_id][n]
                at = self.time + travel_time

            psi = depot
            loc_id = n

        else:
            c = self.customers[x]
            psi_x = c[1:3]

            travel_time = self.distance_table[loc_id][x]
            at = self.time + travel_time

            psi = psi_x
            c[3] = 0
            loc_id = x

        # Update the V_k in Global state
        v_k[3] = q
        v_k[4] = at
        v_k[1:3] = psi
        v_k[5] = loc_id

        return at

    def state_transition(self, k):
        n = self.ins_config.n
        v_k = self.vehicles[k]
        served_demand = 0
        loc_id = int(v_k[5])

        if loc_id == n:
            v_k[3] = 1.
        else:
            # loc_id
            cur_cus = self.customers[loc_id]

            # if the demand is not realized yet, get a realization
            if cur_cus[6] == -1:
                w = self.demand_realization(cid=loc_id)
                self.demand_scenario[loc_id] = w
                # realize the actual demand and flag it as realized [7]=1
                cur_cus[5] = w
                cur_cus[6] = w
                cur_cus[7] = 1
                cur_cus[8] = w

            served_demand = min(cur_cus[5], self.vehicles[k][3])
            cur_cus[5] -= served_demand
            cur_cus[8] = cur_cus[5] + 0.
            v_k[3] -= served_demand

            cur_cus[3] = cur_cus[5] > 1e-5

        return served_demand

    def get_available_customers(self, k):
        v_k = self.vehicles[k]
        loc_id = int(v_k[5])

        is_terminal = 0

        if v_k[3] == 0:
            target_customers = []
        else:
            # distance i to j then j to depot
            # distances has two rows, first-> to customers, second, to depot
            distances = self.distance_table[[loc_id, self.ins_config.n], :]
            rr_dist = np.sum(distances, axis=0)

            remaining_time = self.ins_config.duration_limit - self.time
            avail_customers_cond = self.customers[:, 3] == 1
            # set the depot to False
            avail_customers_cond[-1] = False

            feas_cond = np.logical_and(rr_dist <= remaining_time, avail_customers_cond)
            c_set = self.customers[feas_cond]
            target_customers = [int(m[0]) for m in c_set]

        if loc_id != self.ins_config.n:
            target_customers.append(self.ins_config.n)

        if loc_id == self.ins_config.n and len(target_customers) == 0:
            is_terminal = 1
        return target_customers, is_terminal

