import copy
import random
import numpy as np
from scipy.spatial.distance import pdist, squareform
import Utils
from instance_generator import InstanceConfig


class VRP(object):
    def __init__(self, env_config):
        self.vehicles = None
        self.customers = None
        self.model_type = env_config.model_type
        self.c_enc = None
        self.v_enc = None

        self.time = 0
        self.env_config = env_config
        self.ins_config = None

        self.demand_scenario = None

        self.actions = {}
        self.TT = []
        self.final_reward = 0

        self.all_distance_tables = {}
        self.distance_table = None

        self.post_decision = None
        self.state_transition = None
        self.get_available_customers = None

        self.demand_prob = []
        self.demand_val = []

    def init_encoded_env(self):
        pass

    def initialize_environment(self, instance):
        self.customers = np.array(instance["Customers"])
        self.vehicles = np.array(instance["Vehicles"])
        self.ins_config = instance["Config"]

        self.update_distance_table(instance["Name"])

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

        self.c_enc, self.v_enc, self.instance_chars = self.init_encoded_env()

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
            pos_list.append(list(self.ins_config.depot))
            distance_table = pdist(np.array(pos_list))
            self.all_distance_tables[instance_name] = distance_table
            self.distance_table = squareform(distance_table)

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

    def post_decision_without_p(self, x, k, p):
        pass

    def state_transition_without_p(self, k, is_test=False, time_diff=0):
        pass

    def get_available_customers_without_p(self, k):
        pass

    def post_decision_with_p(self, x, k, preemptive):
        pass

    def state_transition_with_p(self, k, is_test=False, time_diff=0):
        pass

    def get_available_customers_with_p(self, k):
        pass


class VRPSD(VRP):
    def __init__(self, env_config, preempt_action=True):
        super().__init__(env_config)

        if preempt_action:
            self.post_decision = self.post_decision_with_p
            self.state_transition = self.state_transition_with_p
            self.get_available_customers = self.get_available_customers_with_p
        else:
            self.post_decision = self.post_decision_without_p
            self.state_transition = self.state_transition_without_p
            self.get_available_customers = self.get_available_customers_without_p

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

        self.c_enc, self.v_enc, self.instance_chars = self.init_encoded_env()

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

    def post_decision_without_p(self, x, k, p):
        # in this function, the current state transits to the post decision state.
        # it means, action x, only blocks customer x temporary to not be served by any other vehicles and
        # updates the position and the arrival time of the vehicle k to respectively x and get_distance(l_k, x)
        depot = self.ins_config.depot
        n = self.ins_config.n
        norm = Utils.Norms()
        v_k = self.vehicles[k]
        q = v_k[3]
        psi = v_k[1:3]
        at = v_k[4]
        loc_id = int(v_k[5])

        exp_serve_dem = 0
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
            # self.available_C.remove(x)
            # self.active_C[x] = False
            exp_serve_dem = c[-1]

        # Update the V_k in Global state
        v_k[3] = q
        v_k[4] = at
        v_k[1:3] = psi
        v_k[5] = loc_id
        loc_depot = 1 if loc_id == n else 0

        # encoded array:
        if at == self.ins_config.duration_limit:
            self.v_enc[k] = [psi[0], psi[1], q, -1,
                             exp_serve_dem, loc_depot]
        else:
            self.v_enc[k] = [psi[0], psi[1], q, (at - self.time),
                             exp_serve_dem, loc_depot]
        # self.vehicles_pos[k].pop(0)
        # self.vehicles_pos[k].append(psi)

        return at

    def state_transition_without_p(self, k, is_test=False, time_diff=0):
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
            # if cur_cus[3] == 0:
            #     self.active_C[loc_id] = False
            #
            # else:
            #     self.available_C.append(loc_id)
            #     self.active_C[loc_id] = True

            # update encoded customers
            self.v_enc[loc_id, 2:4] = [1, cur_cus[-1]]

        # update encoded vehicles
        self.v_enc[self.v_enc[:, 3] > 0, 3] -= time_diff
        self.v_enc[k, [2, 4]] = [v_k[3], 0]

        return served_demand

    def get_available_customers_without_p(self, k):
        v_k = self.vehicles[k]
        loc_id = int(v_k[5])

        is_terminal = 0

        if v_k[3] == 0:
            target_customers = []
        else:
            # distance i to j then j to depot
            # distances has two rows, first-> to customers, second, to depot
            distances = self.distance_table[[loc_id, self.ins_config.n], :]
            rr_dist = np.sum(distances, axis=0)[:self.ins_config.n]

            remaining_time = self.ins_config.duration_limit - self.time
            avail_customers_cond = self.customers[:, 3] == 1
            feas_cond = np.logical_and(rr_dist <= remaining_time, avail_customers_cond)
            c_set = self.customers[feas_cond]
            target_customers = [int(m[0]) for m in c_set]

        if loc_id != self.ins_config.n:
            target_customers.append(self.ins_config.n)

        if loc_id == self.ins_config.n and len(target_customers) == 0:
            is_terminal = 1
        return target_customers, is_terminal

    def post_decision_with_p(self, x, k, preemptive):
        # in this function, the current state transits to the post decision state.
        # it means, action x, only blocks customer x temporary to not be served by any other vehicles and

        # updates the position and the arrival time of the vehicle k
        depot = self.ins_config.depot
        n = self.ins_config.n
        v_k = self.vehicles[k]
        q = v_k[3]
        loc_id = int(v_k[5])
        exp_serve_dem = 0

        if x == n:
            if v_k[3] == 0:
                at = self.time + self.distance_table[loc_id][n]
            else:
                # it is the terminal. no other choice is available
                at = self.ins_config.duration_limit
            psi = depot
            loc_id = n
        else:
            c = self.customers[x]
            psi_x = c[1:3]

            if preemptive:
                travel_time = self.distance_table[loc_id][n] + self.distance_table[n][x]

                # restock
                q = self.ins_config.capacity / Utils.Norms.Q
            else:
                travel_time = self.distance_table[loc_id][x]
            at = self.time + travel_time

            psi = psi_x
            c[3] = 0
            loc_id = x
            # self.available_C.remove(x)
            # self.active_C[x] = False
            exp_serve_dem = c[-1]

        # Update the V_k in Global state
        v_k[3] = q
        v_k[4] = at
        v_k[1:3] = psi
        v_k[5] = loc_id
        loc_depot = 1 if loc_id == n else 0

        # encoded array:
        if at == self.ins_config.duration_limit:
            self.v_enc[k] = [psi[0], psi[1], q, -1,
                             exp_serve_dem, loc_depot]
        else:
            self.v_enc[k] = [psi[0], psi[1], q, (at - self.time),
                             exp_serve_dem, loc_depot]
        # self.vehicles_pos[k].pop(0)
        # self.vehicles_pos[k].append(psi)

        return at

    def state_transition_with_p(self, k, is_test=False, time_diff=0):
        n = self.ins_config.n
        norm = Utils.Norms()
        v_k = self.vehicles[k]
        served_demand = 0
        loc_id = int(v_k[5])

        if loc_id == n:
            v_k[3] = self.ins_config.capacity / norm.Q
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
            # if cur_cus[3] == 0:
            #     self.active_C[loc_id] = False
            #
            # else:
            #     self.available_C.append(loc_id)
            #     self.active_C[loc_id] = True

            # update encoded customers
            self.c_enc[loc_id, 2:4] = [1, cur_cus[-1]]

        # update encoded vehicles
        self.v_enc[self.v_enc[:, 3] > 0, 3] -= time_diff
        self.v_enc[k, [2, 4]] = [v_k[3], 0]

        return served_demand

    def get_available_customers_with_p(self, k):
        v_k = self.vehicles[k]
        loc_id = int(v_k[5])

        is_terminal = 0

        if v_k[3] == 0:
            target_customers_dir = []
            target_customers_indir = []
        else:
            # distance i to j then j to depot
            # distances has two rows, first-> to customers, second, to depot
            distances_l = self.distance_table[loc_id, :]
            distances_d = self.distance_table[self.ins_config.n, :]
            rr_dist_dir = (distances_l + distances_d)[:self.ins_config.n]
            remaining_time = self.ins_config.duration_limit - self.time
            avail_customers_cond = self.customers[:, 3] == 1
            feas_cond_dir = np.logical_and(rr_dist_dir <= remaining_time, avail_customers_cond)
            target_customers_dir = self.customers[feas_cond_dir, 0].astype(int)

            if loc_id != self.ins_config.n:
                #   loc -> depot -> customer -> depot
                rr_dist_indir = (distances_l + 2 * distances_d)[:self.ins_config.n]
                feas_cond_indir = np.logical_and(rr_dist_indir <= remaining_time, avail_customers_cond)
                target_customers_indir = self.customers[feas_cond_indir, 0].astype(int)
            else:
                #   if located at the depot the preemptive action is not defined
                target_customers_indir = []

        if loc_id == self.ins_config.n and len(target_customers_dir) == 0:
            is_terminal = 1
        return target_customers_dir, target_customers_indir, is_terminal


class DVRPSD(VRP):
    def __init__(self, env_config):
        super().__init__(env_config)

        self.post_decision = self.post_decision_with_p
        self.state_transition = self.state_transition_with_p
        self.get_available_customers = self.get_available_customers_with_p
        self.demand_realization = self.demand_realization_gendreau

        self.vehicles_traveled_time = []
        self.vehicles_traveled_time2 = []
        self.p_l = 5.

        self.vehicles_terminated = None

    def init_encoded_env(self):
        norm = Utils.Norms()
        n = self.ins_config.n

        #   features set for customers
        # is_realized: if the actual demand is realized, \tild{d}=\bar{d} if \hat{d}=-1 else \hat{d},
        # is_customer: indicates that the node is a customer and not a depot,
        # is available: not being served by others
        # is_target: whether it is in the set of target customers or not
        # l_x, l_y, is_realized, \tild{d}, is_available, is_customer, is_target
        c_set = self.customers[:, [1, 2, 7, 8, 8, 8, 8]]
        # c_set[c_set[:, 0] > 0.0001, -2:] = [1., 0.]
        c_set[:, -3:] = [1., 1., 0.]

        # add a node as the depot
        depot = np.array([self.ins_config.depot[0], self.ins_config.depot[1],
                          1., 0., 1, -1, 0])

        # add a dummy node at the end
        dummy = np.zeros(7)

        c_set = np.vstack([c_set, depot, dummy])

        #   features set fo vehicles
        #   d_exp: expected demand to serve at its destination, loc_depot: whether it is located at the depot
        # For v: x, y, q, a, d_exp, loc_depot, is_terminated
        v_set = []
        for v in self.vehicles:
            exp_dem = 0
            loc_depot = 1
            l = int(v[-1])
            if l != n:
                exp_dem = self.customers[l, -1]
                loc_depot = 0
            v_set.append([v[1], v[2], v[3], v[4] - self.time, exp_dem, loc_depot, 0])

        v_set = np.array(v_set)

        #   instance characteristics - used in the generalized DecDQN
        instance_chars = [self.ins_config.duration_limit, self.ins_config.capacity / norm.Q, self.ins_config.m,
                          0, 0, 0]
        instance_chars[-3 + self.ins_config.density_class] = 1.

        return c_set, v_set, instance_chars

    def initialize_environment(self, instance):
        self.customers = np.array(instance["Customers"])
        self.vehicles = np.array(instance["Vehicles"])
        self.ins_config = instance["Config"]

        self.update_distance_table(instance["Name"])

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
            self.ins_config = copy.deepcopy(instance["Config"])

        if reset_distance:
            # when the set of customers might be changed, the distance tables will be changed here,
            # generate them and add to the memory if it is not generated yet, otherwise, use the memory
            self.update_distance_table(instance["Name"])

        if scenario is None:
            self.demand_scenario = -np.ones(self.ins_config.n)
        else:
            self.demand_scenario = scenario

        self.actions = {}
        self.vehicles_traveled_time = []
        self.vehicles_traveled_time2 = []
        for m in range(self.ins_config.m):
            self.actions[m] = [-2]
            self.vehicles_traveled_time.append(0.)
            self.vehicles_traveled_time2.append(0.)

        self.c_enc, self.v_enc, self.instance_chars = self.init_encoded_env()

        self.vehicles_terminated = np.zeros(self.ins_config.m)

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

    def post_decision_with_p(self, x, k, preemptive):
        # in this function, the current state transits to the post decision state.
        # it means, action x, only blocks customer x temporary to not be served by any other vehicles and

        # updates the position and the arrival time of the vehicle k
        depot = self.ins_config.depot
        n = self.ins_config.n
        v_k = self.vehicles[k]
        q = v_k[3]
        loc_id = int(v_k[5])
        exp_serve_dem = 0
        is_terminated = 0
        if x == n:
            # it is route failure
            at = self.time + self.distance_table[loc_id][n]
            self.vehicles_traveled_time[k] += self.distance_table[loc_id][n]
            # if the vehicle chooses to travel directly to the depot with a positive capacity, it means terminal
            if q > 0.0001:
                # it is the terminal
                self.vehicles_terminated[k] = 1
                is_terminated = 1

            psi = depot
            loc_id = n
        else:
            c = self.customers[x]
            psi_x = c[1:3]

            if preemptive:
                travel_time = self.distance_table[loc_id][n] + self.distance_table[n][x]

                # restock
                q = self.ins_config.capacity / Utils.Norms.Q
            else:
                travel_time = self.distance_table[loc_id][x]
            at = self.time + travel_time
            self.vehicles_traveled_time[k] += travel_time

            psi = psi_x
            c[3] = 0
            loc_id = x
            # self.available_C.remove(x)
            # self.active_C[x] = False
            exp_serve_dem = c[-1]

            # make the customer unavailable for others
            self.c_enc[x, 4] = 0

        # Update the V_k in Global state
        v_k[3] = q
        v_k[4] = at
        v_k[1:3] = psi
        v_k[5] = loc_id
        loc_depot = 1 if loc_id == n else 0

        # encoded array:
        self.v_enc[k] = [psi[0], psi[1], q, at,
                         exp_serve_dem, loc_depot, is_terminated]

        return at

    def state_transition_with_p(self, k, is_test=False, time_diff=0):
        n = self.ins_config.n
        norm = Utils.Norms()
        v_k = self.vehicles[k]
        served_demand = 0
        loc_id = int(v_k[5])

        if loc_id == n:
            v_k[3] = self.ins_config.capacity / norm.Q
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
            if not cur_cus[3]:
                self.c_enc[loc_id] = np.zeros(7)
            else:
                # update the realized, tiled{d}, available
                self.c_enc[loc_id, 2:5] = [1, cur_cus[-1], 1]

        # update encoded vehicles, q and exp_dem
        self.v_enc[k, [2, 4]] = [v_k[3], 0]

        return served_demand

    def get_available_customers_with_p(self, k):
        v_k = self.vehicles[k]
        loc_id = int(v_k[5])

        is_terminal = 0

        if v_k[3] == 0:
            target_customers_dir = []
            target_customers_indir = []
        else:
            # distance i to j then j to depot
            # distances has two rows, first-> to customers, second, to depot
            avail_customers_cond = self.customers[:, 3] == 1
            # feas_cond_dir = np.logical_and(rr_dist_dir <= remaining_time, avail_customers_cond)
            target_customers_dir = list(self.customers[avail_customers_cond, 0].astype(int))

            # if there is another active vehicle besides k, allow k to terminate
            if self.ins_config.m - sum(self.vehicles_terminated) > 1.01:
                target_customers_dir.append(self.ins_config.n)

            if loc_id != self.ins_config.n:
                #   loc -> depot -> customer -> depot
                target_customers_indir = self.customers[avail_customers_cond, 0].astype(int)
            else:
                #   if located at the depot the preemptive action is not defined
                target_customers_indir = []

        if loc_id == self.ins_config.n and len(target_customers_dir) == 0:
            is_terminal = 1
        return target_customers_dir, target_customers_indir, is_terminal

