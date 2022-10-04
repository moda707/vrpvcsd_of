import math
import random
import numpy as np
import tensorflow as tf
import Utils
import qnetwork
import vrp


class QLearning(object):
    def __init__(self, env: vrp.VRPSD, rl_config, sess, transfer_learning=False, feat_size_c=8, feat_size_v=6):
        # set the environment
        self.env = env
        # self.ins_config = self.env.ins_config
        self.env_config = self.env.env_config

        # rl configs
        self.rl_config = rl_config
        self.lr = float(rl_config.lr)
        self.test_every = int(rl_config.test_every)
        self.ep_max = int(rl_config.ep_max)
        self.update_freq = int(rl_config.update_freq)
        self.replace_target_iter = int(rl_config.replace_target_iter)
        self.batch_size = int(self.rl_config.batch_size)
        self.memory_size = int(self.rl_config.memory_size)
        self.tl = transfer_learning
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v

        self.memory = Memory(self.memory_size)
        self.replay_start = int(self.memory_size)
        self.dqn = None
        self.nb = rl_config.nb

        # loging
        self.depot_stay_count = []
        self.TrainTime = 0
        self.DecisionEpochs = 0
        self.epsilon = 1.
        self.update_counter = 0
        self.depot_stay_count = []
        self.zero_q = 0

        # epsilon decaying parameters
        self.main_variables = None
        self.build_net()
        self.sess = sess
        # self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(var_list=self.main_variables)

        self.Max_trials = rl_config.trials
        if self.rl_config.use_obs:
            self.get_state = self.get_state_observed
            self.choose_action = self.choose_action_with_obs
        else:
            self.get_state = self.get_state_real
            self.choose_action = self.choose_action_without_obs

    def build_net(self):
        if self.rl_config.use_obs:
            # Note that 100 = 1 / (hm_slice[0] * hm_slice[1])
            state_size = 2 * 100 + \
                         self.env.ins_config.m * 6 + 8 * self.nb + 2
            action_size = self.nb + 1
        else:
            state_size = self.env.ins_config.n * 5 + self.env.ins_config.m * 4 + 1
            action_size = self.env.ins_config.n + 1

        dqn_args = {"init_lr": 0.001, "state_size": state_size, "action_size": action_size}
        self.dqn = qnetwork.DQN(**dqn_args)

    def get_state_observed(self, k):
        hm_slice = self.env_config.hm_slice
        # hm_size = int(1 / (hm_slice[0] * hm_slice[1]))
        hm_size = 10 * 10
        # state_size = 2 * hm_size + self.env.ins_config.m * 7 + self.nb * 8 + 6
        norms = Utils.Norms()

        v_k = self.env.vehicles[k]
        v_loc = int(v_k[-1])

        ins_config = self.env.ins_config

        # customers: heatmaps
        hm_count = np.zeros(hm_size)
        hm_dem = np.zeros(hm_size)
        for e, i in enumerate(self.env.heatmap_ids):
            hm_count[i] += 0.2
            hm_dem[i] += self.env.customers[e, -1]

        # vehicles: x, y, q, a, xd
        v_set = self.env.vehicles[:, [1, 2, 3, 4, 4]]
        v_set[:, -2] -= self.env.time
        v_set[:, -1] = self.env.customers[self.env.vehicles[:, -1].astype(int), -1]
        v_set = np.reshape(v_set, [-1])

        # time
        rt = (ins_config.duration_limit - self.env.time) / norms.DL

        # active vehicle
        dist_to_depot = self.env.distance_table[v_loc][ins_config.n]
        active_vehicles = np.zeros(ins_config.m)
        active_vehicles[k] = 1.

        # target customers
        # take all feasible customers
        avail_customers_ids, is_terminal = self.env.get_available_customers(k)
        depot_is_action = False
        if ins_config.n in avail_customers_ids:
            avail_customers_ids.remove(ins_config.n)
            depot_is_action = True

        avail_customers = self.env.customers[avail_customers_ids]
        distances_to_v = self.env.distance_table[avail_customers_ids, v_loc]
        distances_to_depot = self.env.distance_table[avail_customers_ids, ins_config.n]

        # choose based on min(q, dem) / dist_cv
        measure = [-min(v_k[3], t[-1]) / distances_to_v[e] for e, t in enumerate(avail_customers)]
        measure_sorted = np.argsort(measure)
        target_customers_ids = list(np.array(avail_customers_ids)[measure_sorted[:self.nb]])
        target_customers = self.env.customers[target_customers_ids]

        target_customers_features = np.array(
            [[t[1], t[2], t[-1], min(v_k[3], t[-1]), t[7], t[-1] / distances_to_v[e],
              distances_to_v[e], distances_to_depot[e]]
             for e, t in enumerate(target_customers)])
        target_customers_features = np.reshape(target_customers_features, [-1])
        target_customers_features_padded = np.zeros(8*self.nb)
        target_customers_features_padded[:len(target_customers_features)] = target_customers_features

        target_customers_onehot = np.zeros(self.nb + 1)
        target_customers_onehot[:len(target_customers_ids)] = 1.
        if depot_is_action:
            target_customers_ids.append(ins_config.n)
            target_customers_onehot[-1] = 1.

        # state_size = 10*10+10*10+ m*5 + m + 8 * nb + 2
        state_observed = np.concatenate([hm_count, hm_dem, v_set, active_vehicles, target_customers_features_padded,
                                         [rt, dist_to_depot]])

        return state_observed, target_customers_ids, target_customers_onehot, is_terminal

    def get_state_real(self, k):
        c_set = np.reshape(self.env.customers[:self.env.ins_config.n, [1, 2, 3, 4, 5]], [-1])
        v_set = self.env.vehicles[:, [1, 2, 3, 4]]
        v_set[:, -1] -= self.env.time
        v_set = np.reshape(v_set, [-1])
        state = np.concatenate([c_set, v_set, [self.env.ins_config.duration_limit - self.env.time]])
        target_customers, is_terminal = self.env.get_available_customers(k)
        target_customers_onehot = np.zeros(self.env.ins_config.n + 1)
        target_customers_onehot[target_customers] = 1

        return state, target_customers, target_customers_onehot, is_terminal

    def choose_action_without_obs(self, k, trials, train=True):
        if train:
            epsilon = 1. - 0.9 * trials / (self.rl_config.max_trials * .3)
            epsilon = max(0.1, epsilon)
        else:
            epsilon = 0.

        state, target_customers, target_customers_onehot, is_terminal = self.get_state(k)
        n_actions = len(target_customers)

        if n_actions == 0:
            best_action = self.env.ins_config.n
        else:
            if random.random() <= epsilon:
                # explore
                sind = math.floor(random.random() * n_actions)
                best_action = target_customers[sind]
            else:
                q_values = self.dqn.value(state=np.expand_dims(state, axis=0), sess=self.sess)[0]
                q_values[q_values < 0.] = 0.
                q_values += (1 - target_customers_onehot) * (-10e9)

                if np.max(q_values) <= 0.01:
                    self.zero_q += 1
                else:
                    self.zero_q = 0

                best_action = np.argmax(q_values)
        # if self.env.vehicles[k][-1] == self.env.ins_config.n and best_action == self.env.ins_config.n:
        #     print("asd")
        return best_action, best_action, state, target_customers_onehot, is_terminal

    def choose_action_with_obs(self, k, trials, train=True):
        if train:
            epsilon = 1. - 0.9 * trials / (self.rl_config.max_trials * .3)
            epsilon = max(0.1, epsilon)
        else:
            epsilon = 0.

        state, target_customers, target_customers_onehot, is_terminal = self.get_state(k)
        n_actions = len(target_customers)

        selected_target_id = self.nb
        if n_actions == 0:
            best_action = self.env.ins_config.n
        else:
            if random.random() <= epsilon:
                # explore
                selected_target_id = math.floor(random.random() * n_actions)
                best_action = target_customers[selected_target_id]
            else:
                q_values = self.dqn.value(state=np.expand_dims(state, axis=0), sess=self.sess)[0]
                q_values[q_values < 0.] = 0.
                q_values += (1 - target_customers_onehot) * (-10e9)

                if np.max(q_values) <= 0.01:
                    self.zero_q += 1
                else:
                    self.zero_q = 0

                selected_target_id = np.argmax(q_values)
                if selected_target_id == self.nb:
                    best_action = self.env.ins_config.n
                else:
                    best_action = target_customers[selected_target_id]

        return best_action, selected_target_id, state, target_customers_onehot, is_terminal

    def learn(self, trials):
        if trials % self.replace_target_iter == 0:
            self.sess.run(self.dqn.replace_target_op)

        # batch contains: state, action, available_targets, reward, is_terminal, next_state
        batch = self.memory.sample(self.batch_size)

        batch_range = range(self.batch_size)

        state = [val[0] for val in batch]
        available_targets = [val[1] for val in batch]
        selected_targets = [val[2] for val in batch]
        selected_targets = np.array([[i, selected_targets[i]] for i in batch_range])

        rewards = [val[3] for val in batch]
        is_terminal = [val[4] for val in batch]
        next_state = [val[5] for val in batch]
        next_available_targets = [val[6] for val in batch]

        # batch x nb
        next_q_values = self.dqn.value(state=next_state, sess=self.sess)
        # for not available targets, set the q value to a big negative number
        next_q_values -= 1000 * (1 - np.array(next_available_targets))

        # if self.rl_config.use_double_qn:
        best_action = np.argmax(next_q_values, axis=1).reshape([-1, 1])

        target_q_values = self.dqn.value_(state=next_state, sess=self.sess)

        max_next_q_values = [target_q_values[i][best_action[i]] for i in batch_range]

        max_next_q_values = np.array(max_next_q_values).reshape(-1)
        max_next_q_values[max_next_q_values < 0] = 0.

        discount_factor = self.rl_config.gama * (1 - np.array(is_terminal))

        q_target = rewards + discount_factor * max_next_q_values

        loss, avg_gradient = self.dqn.optimize(states=state, selected_targets=selected_targets,
                                               target_values=q_target, sess=self.sess)
        return loss, avg_gradient

    def compute_actual_reward(self, k, x):
        n = self.env.ins_config.n

        if x == n:
            return 0
        else:
            selected_customer = self.env.customers[x]
            q = self.env.vehicles[k][3]

            if self.env_config.model_type == "VRPSD":
                kk = sum(min(q, x * selected_customer[-1]) * y
                         for x, y in zip(self.env.demand_val, self.env.demand_prob))
            else:
                exp_dem = selected_customer[-1]
                rng = 5 / Utils.Norms.Q if exp_dem > 5 / Utils.Norms.Q else 4 / Utils.Norms.Q
                l = selected_customer[-1] - rng
                u = selected_customer[-1] + rng
                if q >= u:
                    return (l + u) / 2.
                elif q <= l:
                    return q
                else:
                    e = (q - l) / (u - l)
                    return e * (l + q) / 2 + (1 - e) * q

            return kk

    def test_model(self, test_instance, scenario, need_reset=True):
        sum_exp_served = 0
        n = self.env.ins_config.n
        m = self.env.ins_config.m

        self.env.actions = {}
        reset_distance = need_reset
        # if need_reset:
        if test_instance is not None:
            self.env.reset(test_instance, scenario=scenario,
                           reset_distance=reset_distance)
        else:
            self.env.reset(scenario=scenario)
        self.env.time_table = []
        for j in range(m):
            self.env.time_table.append((j, 0))

        # self.env.demand_scenario = scenario

        # time scheduler
        agent_reward = [0] * m
        agents_record = {}
        for j in range(m):
            self.env.actions[j] = [n]
            agents_record[j] = []

        final_reward = 0

        last_actions = {}
        n_routes = 0
        n_visits = 0
        n_preemptives = 0
        avg_terminal_time = 0
        max_travel_time = 0

        prev_time = 0

        while len(self.env.time_table) > 0:
            self.env.time_table.sort(key=lambda x: x[1])
            k, time = self.env.time_table.pop(0)
            self.env.time = time

            # transit from s^x_{k-1} to s_k
            served_demand = self.env.state_transition(k)

            prev_time = time

            # active vehicle k takes action
            x_k, _, _, _, is_terminal = self.choose_action(k, trials=0, train=False)

            agent_reward[k] += served_demand
            agents_record[k].append(served_demand)

            if x_k != n:
                sum_exp_served += self.env.customers[x_k][-1]

            self.env.actions[k].append(x_k)

            if x_k == n:
                if self.env.vehicles[k][-1] != n:
                    n_routes += 1
                    if self.env.vehicles[k][3] > 0 and not is_terminal:
                        n_preemptives += 1
            else:
                n_visits += 1

            if is_terminal == 1:
                avg_terminal_time += time
                max_travel_time = time

            last_actions[k] = x_k

            # transit from s_k to s^x_k
            t_k = self.env.post_decision(x_k, k)

            # schedule the next event for vehicle k if it still has time
            if t_k < self.env.ins_config.duration_limit and is_terminal == 0:
                self.env.time_table.append((k, t_k))

            final_reward += served_demand

        final_reward *= Utils.Norms.Q
        avg_terminal_time /= m
        n_served = len([c for c in self.env.customers if c[3] == 0])
        results = Utils.TestResults(final_reward=final_reward, actions=self.env.actions, n_routes=n_routes,
                                    n_served=n_served, avg_travel_time=avg_terminal_time, n_visits=n_visits,
                                    max_travel_time=max_travel_time, agents_reward=agent_reward)
        results.n_preemptives = n_preemptives
        if scenario is not None:
            results.tot_realized_demand = sum(scenario)
        results.service_rate = results.final_reward / results.tot_realized_demand
        results.agent_record = agents_record

        return results

    def save_network(self, base_address, code, write_meta=True):
        # saver = tf.compat.v1.train.Saver()
        dir_name = base_address
        self.saver.save(self.sess, dir_name + "/" + code, write_meta_graph=write_meta)

    def load_network(self, network_address):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(network_address))
        self.sess.run(self.dqn.replace_target_op)


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []
        self._weights = []

    def add_sample(self, sample, weight=1):
        self._samples.append(sample)

        # self._weights.append(weight)

        if len(self._samples) > self._max_memory:
            self._samples.pop(0)
            # self._weights.pop(0)

    def sample(self, no_samples):
        # if no_samples > len(self._samples):
        #     no_samples = len(self._samples)
        ll = self.get_size()
        out = [self._samples[int(random.random() * ll)] for _ in range(no_samples)]
        return out

    def sample_last(self, no_samples):
        if no_samples > len(self._samples):
            no_samples = len(self._samples)

        out = self._samples[-no_samples:]
        return out

    def get_size(self):
        return len(self._samples)
