import math
import random
import numpy as np
import tensorflow as tf
import Utils
import qnetwork
import vrp
from qnetwork import DQN, DQNPTransferred, DQNP2, DQNBono, DQNP3, DQNP4


class QLearning(object):
    def __init__(self, env: vrp.VRP, rl_config, sess, transfer_learning=False, feat_size_c=8, feat_size_v=6):
        # set the environment
        self.env = env
        # self.ins_config = self.env.ins_config
        self.env_config = self.env.env_config

        # active fine-tuning?
        fine_tune = False

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
        self.build_net(transfer_learning, fine_tune=fine_tune)
        self.sess = sess
        # self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver(var_list=self.main_variables)

        # epsilon decaying strategy - parameters
        self.eps_A = 0.15
        self.eps_B = 0.1
        self.eps_C = 0.05
        self.Max_trials = rl_config.trials

    def build_net(self, transfer_learning=False, fine_tune=False):
        dqn_args = {"init_lr": 0.001, "emb_size": 128, "feat_size_c": self.feat_size_c,
                    "feat_size_v": self.feat_size_v, "nb": self.env.ins_config.n + 1}
        if self.rl_config.preempt_action:
            if transfer_learning:
                dqn_args["fine_tune"] = fine_tune
                model = DQNPTransferred(**dqn_args)
            else:
                # model = DQNP2(**dqn_args)
                # model = DQNBono(**dqn_args)
                # model = DQNP3(**dqn_args)
                model = DQNP4(**dqn_args)
        else:
            model = DQN(**dqn_args)
        self.dqn = model

    def observation_function_without_p(self, k):
        pass

    def observation_function_with_p(self, k):
        pass

    def choose_action_without_p(self, k, trials, train=True):
        pass

    def choose_action_with_p(self, k, trials, train=True):
        pass

    def choose_action_centralized(self, k, trials, train=True):
        pass

    def learn(self, trials):
        pass

    def compute_actual_reward(self, k, x, is_preepmtive, is_terminal=False):
        pass

    def test_model(self, test_instance, scenario, need_reset=True):
        pass

    def save_network(self, base_address, code, write_meta=True):
        # saver = tf.compat.v1.train.Saver()
        dir_name = base_address
        self.saver.save(self.sess, dir_name + "/" + code, write_meta_graph=write_meta)

    def epsilon_calculator(self, trial):
        standardized_time = (trial - self.eps_A * self.Max_trials) / (self.eps_B * self.Max_trials)
        cosh = np.cosh(math.exp(-standardized_time))
        epsilon = (1.05 + self.eps_C) - (1 / cosh + (trial * self.eps_C / self.Max_trials))

        # In transfer learning, it makes the simulation to exploit more.
        # epsilon /= 2.
        return epsilon

    def load_network(self, network_address):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(network_address))
        self.sess.run(self.dqn.replace_target_op)


class QLearningMax(QLearning):
    def __init__(self, env: vrp.VRPSD, rl_config, sess, transfer_learning=False, feat_size_c=8, feat_size_v=6):
        super().__init__(env, rl_config, sess, transfer_learning)

        if self.rl_config.preempt_action:
            self.observation_function = self.observation_function_with_p
            self.choose_action = self.choose_action_with_p
        else:
            self.observation_function = self.observation_function_without_p
            self.choose_action = self.choose_action_without_p

    # IMPORTANT: this function generates a two layered NN for the centralized formulation.
    # comment it for the other cases.
    def build_net(self, transfer_learning=False, fine_tune=False):
        dqn_args = {"init_lr": 0.001, "n": self.env.ins_config.n, "m": self.env.ins_config.m,
                    "feat_size_c": 5, "feat_size_v": 4}

        self.dqn = qnetwork.CentralizedDQN(**dqn_args)

    def observation_function_without_p(self, k):
        n = self.env.ins_config.n
        v_k = self.env.vehicles[k]
        loc_id = int(v_k[-1])

        # take all feasible customers
        target_customers, is_terminal = self.env.get_available_customers(k)

        if loc_id != n or len(target_customers) == 0:
            # if len(target_customers) == 0:
            target_customers.append(n)

        c_set = np.array(self.env.c_enc)
        v_set = np.array(self.env.v_enc)

        # vehicles_pos_history = np.array(self.Env.vehicles_pos[k]) / norm.COORD
        rem_time = (self.env.ins_config.duration_limit - self.env.time)

        available_customers = np.zeros(n + 2)
        available_customers[:n] = self.env.customers[:, 3] > 0

        obs = {"customers": c_set, "vehicles": v_set, "active_vehicle": k + 0,
               "time": [rem_time], "available_customers": available_customers,
               "instance_chars": self.env.instance_chars}

        return obs, target_customers, is_terminal

    def observation_function_with_p(self, k):
        n = self.env.ins_config.n

        # take all feasible customers
        target_customers_dir, target_customers_indir, is_terminal = self.env.get_available_customers(k)
        target_customers = (target_customers_dir, target_customers_indir)

        c_set = np.array(self.env.c_enc)
        v_set = np.array(self.env.v_enc)

        # vehicles_pos_history = np.array(self.Env.vehicles_pos[k]) / norm.COORD
        # Normalized dl
        rem_time = (self.env.ins_config.duration_limit / Utils.Norms.COORD - self.env.time)

        available_customers = np.zeros(n + 2)
        available_customers[:n] = self.env.customers[:, 3] > 0

        obs = {"customers": c_set, "vehicles": v_set, "active_vehicle": k + 0,
               "time": [rem_time], "available_customers": available_customers,
               "instance_chars": self.env.instance_chars}

        return obs, target_customers, is_terminal

    def observation_function_centralized(self):
        c_set = np.reshape(self.env.customers[:self.env.ins_config.n, [1, 2, 3, 4, 5]], [-1])
        v_set = self.env.vehicles[:, [1, 2, 3, 4]]
        v_set[:, -1] -= self.env.time
        v_set = np.reshape(v_set, [-1])
        state = np.concatenate([c_set, v_set, [self.env.ins_config.duration_limit - self.env.time]])
        return state

    def observation_function_heatmap(self):
        c_set = np.reshape(self.env.customers[:self.env.ins_config.n, [1, 2, 3, 4, 5]], [-1])

        v_set = self.env.vehicles[:, [1, 2, 3, 4]]
        v_set[:, -1] -= self.env.time
        v_set = np.reshape(v_set, [-1])
        state = np.concatenate([c_set, v_set, [self.env.ins_config.duration_limit - self.env.time]])
        return state

    def choose_action_without_p(self, k, trials, train=True):
        nb = self.nb
        n = self.env.ins_config.n

        if train:
            # epsilon = 1 - (0.9 * self.Trials) / self.ep_max
            # epsilon = max(epsilon, 0.1)
            epsilon = self.epsilon_calculator(trials)
        else:
            epsilon = 0.
        # epsilon = 0.
        loc_id = int(self.env.vehicles[k][-1])

        obs, target_customers, is_terminal = self.observation_function(k)

        n_actions = len(target_customers)

        o_target_customers = [target_customers[i] if i < n_actions else n + 1
                              for i in range(nb)]
        obs["target_customers"] = o_target_customers
        available_targets = np.zeros(nb)
        available_targets[:n_actions] = 1.
        obs["available_targets"] = available_targets

        # target_customers always has at least one item
        if n_actions == 1:
            # it means there is only one choice, so take it.
            selected_index = 0
            best_action = target_customers[0]
        else:
            if n_actions >= nb - 1:
                # compute the moda value and take 15
                tt = np.array(target_customers)
                if n in tt:
                    tt = tt[:-1]

                vv = np.minimum(self.env.customers[tt, -1], self.env.vehicles[k][3])

                vv /= self.env.distance_table[loc_id][tt]
                tt = list(tt[np.argsort(-vv)])[:nb - 1]
                if n in target_customers:
                    tt.append(n)
                target_customers = tt
                n_actions = len(target_customers)

                o_target_customers = [target_customers[i] if i < n_actions else n + 1
                                      for i in range(nb)]
                obs["target_customers"] = o_target_customers
                available_targets = np.zeros(nb)
                available_targets[:n_actions] = 1.
                obs["available_targets"] = available_targets
            obs["customers"][target_customers, -1] = 1.
            if random.random() <= epsilon:
                # explore
                selected_index = math.floor(random.random() * n_actions)

                best_action = target_customers[selected_index]

            else:
                obs_mk = {}
                for e1, e2 in obs.items():
                    obs_mk[e1] = np.expand_dims(e2, axis=0)
                obs_mk["active_vehicle"] = [[0, obs["active_vehicle"]]]

                q_values = self.dqn.value(obs=obs_mk, sess=self.sess)[0]
                q_values[q_values < 0.] = 0.
                q_values += (1 - available_targets) * (-10e9)

                if np.max(q_values) <= 0.01:
                    self.zero_q += 1
                else:
                    self.zero_q = 0
                if self.zero_q > 5000:
                    print("5000 consecutive zero q values")
                    self.zero_q = 0
                    if train:
                        self.stop = True

                selected_index = np.argmax(q_values)

                best_action = target_customers[selected_index]
                # best_action = target_customers[0]

        # return observation, x_k, preemptive, terminal, target customers, index of x_k
        return obs, int(best_action), None, is_terminal, selected_index

    def choose_action_with_p(self, k, trials, train=True):
        nb = self.nb
        n = self.env.ins_config.n

        exp_coef = 16 / self.rl_config.max_trials
        if train:
            # epsilon = 1 - (0.9 * trials) / self.ep_max
            # epsilon = max(epsilon, 0.1)
            epsilon = np.exp(-trials * exp_coef) + 0.05 + 0.03 * (1 - trials / self.rl_config.max_trials)
            if self.tl:
                epsilon /= 2.
        else:
            epsilon = 0.

        loc_id = int(self.env.vehicles[k][-1])

        obs, (target_customers_dir, target_customers_indir), is_terminal = self.observation_function(k)
        n_actions_dir = len(target_customers_dir)

        # target_customers always has at least one item
        if n_actions_dir == 0:
            # depot is the only choice to travel
            # it means there is only one choice, so take it.
            best_action = n
            preemptive = False
            selected_index = 0
            obs["target_customers"] = np.array([n + 1] * nb)
            obs["target_customers"][0] = n
            obs["available_targets"] = np.zeros(2 * nb)
            obs["available_targets"][0] = 1

            # is_terminal = True
        else:
            #   to reduce the size of the action set, we restrict the set of customers to a set of target customers
            if n_actions_dir > nb:
            # if n_actions_dir > 0:
                tt = np.array(target_customers_dir)

                #   make sure the depot is out of the list
                if n in tt:
                    tt = tt[:-1]

                # compute the \rho value and take the top nb customers
                vv = np.minimum(self.env.customers[tt, -1], self.env.vehicles[k][3])
                vv /= self.env.distance_table[loc_id][tt]
                tt = tt[np.argsort(-vv)][:nb]

                #   update the modified list of target direct and direct actions
                target_customers_dir = tt
                target_customers_indir = np.intersect1d(target_customers_dir, target_customers_indir)

            n_actions_dir = len(target_customers_dir)

            #   the set of target customers can be served directly, if they are less than nb, fill the list with
            #   the index of the dummy customer (n + 1)
            o_target_customers = [target_customers_dir[i] if i < n_actions_dir else n + 1
                                  for i in range(nb)]
            obs["target_customers"] = o_target_customers

            # Not all the direct target customers can be served indirectly, so flag their indirect service to zero.
            # this flag for dummy customers is always zero.
            available_targets_dir = np.zeros(nb)
            available_targets_dir[:n_actions_dir] = 1.
            available_targets_indir = np.zeros(nb)
            available_targets_indir[:n_actions_dir] = [m in target_customers_indir for m in target_customers_dir]

            #   combine them together - size: 2*nb
            available_targets = np.array([[available_targets_dir[m], available_targets_indir[m]]
                                          for m in range(nb)]).reshape(-1)

            obs["available_targets"] = available_targets

            obs["customers"][target_customers_dir, -1] = 1.
            if len(target_customers_indir) > 0:
                obs["customers"][target_customers_indir, -1] = 2.

            if random.random() <= epsilon:
                # explore
                sind = math.floor(random.random() * n_actions_dir)
                best_action = target_customers_dir[sind]

                if target_customers_dir[sind] in target_customers_indir:
                    if random.random() < 0.5:
                        selected_index = sind * 2
                        preemptive = False
                    else:
                        selected_index = sind * 2 + 1
                        preemptive = True
                else:
                    selected_index = sind * 2
                    preemptive = False

            else:
                obs_mk = {}
                for e1, e2 in obs.items():
                    obs_mk[e1] = np.expand_dims(e2, axis=0)
                obs_mk["active_vehicle"] = [[0, obs["active_vehicle"]]]

                q_values = self.dqn.value(obs=obs_mk, sess=self.sess)[0]
                q_values[q_values < 0.] = 0.
                q_values += (1 - available_targets) * (-10e9)

                # temp: justify the q values with vv
                # vrate = (np.sort((vv - np.min(vv))/(np.max(vv) - np.min(vv))/200. + 1.)[::-1][:nb]).reshape(-1, 1)
                # vrate = np.concatenate([vrate, vrate], axis=1).reshape(-1)
                # vrate_z = np.zeros(nb*2)
                # vrate_z[:len(vrate)] = vrate
                # modified_q = q_values * vrate_z

                # if np.max(q_values) <= 0.01:
                #     self.zero_q_value += 1
                # else:
                #     self.zero_q_value = 0
                # if self.zero_q_value > 5000:
                #     print("5000 consecutive zero q values")
                #     self.zero_q_value = 0
                #     if train:
                #         self.stop = True
                # if train:
                #     # softmax
                #     q_values[np.invert(available_targets.astype(bool))] = -np.inf
                #     sm_q_values = Utils.softmax(q_values)
                #     selected_index = np.random.choice(range(len(q_values)), 1, p=sm_q_values)[0]
                #     if available_targets[selected_index] == 0:
                #         selected_index = 0
                # else:
                selected_index = np.argmax(q_values)
                # selected_index = np.argmax(modified_q)

                best_action = target_customers_dir[math.floor(selected_index / 2.)]
                if selected_index % 2. == 1:
                    preemptive = True
                else:
                    preemptive = False

        return obs, int(best_action), preemptive, is_terminal, selected_index

    def choose_action_centralized(self, k, trials, train=True):
        # exp_coef = 16 / self.rl_config.max_trials
        if train:
            epsilon = 1. - 0.9 * trials / (self.rl_config.max_trials * .3)
            epsilon = max(0.1, epsilon)
            # epsilon = np.exp(-trials * exp_coef) + 0.05 + 0.03 * (1 - trials / self.rl_config.max_trials)
            # if self.tl:
            #     epsilon /= 2.
        else:
            epsilon = 0.

        target_customers, is_terminal = self.env.get_available_customers_without_p(k)
        n_actions = len(target_customers)
        state = self.observation_function_centralized()
        target_customers_onehot = np.zeros(self.env.ins_config.n + 1)
        if n_actions == 0:
            best_action = self.env.ins_config.n
            target_customers_onehot[-1] = 1
        else:
            target_customers_onehot[target_customers] = 1

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
        return best_action, state, target_customers_onehot, is_terminal

    def learn(self, trials):
        """
        each batch contains:
        1- obs: C, V, time, vb, act veh position, target_customers, available_targets, available customers, inst chars
        2- selected_target (action, an index in available targets): ind -> batch,ind
        3- reward
        4- is_terminal
        5- obs_next: same items as obs

        Procedure:
        - a batch of experiences is sampled from the memory
        - with double_qn, in order to compute the max_{x} Q(s_{k+1}, x), we use the primary network to decide the best,
        but evaluate the Q value of that action with the target network.
        - the future rewards are discounted considering the n-step q learning.

        """

        if trials % self.replace_target_iter == 0:
            self.sess.run(self.dqn.replace_target_op)

        batch = self.memory.sample(self.batch_size)

        def make_up_dict(o, axis=None):
            key_list = ["customers", "vehicles", "active_vehicle", "time", "target_customers",
                        "available_targets", "available_customers", "instance_chars"]
            if axis is None:
                new_dict = dict((k, np.stack([v[k] for v in o])) for k in key_list)
            else:
                new_dict = dict((k, np.concatenate([v[k] for v in o], axis=axis)) for k in key_list)

            return new_dict

        batch_range = range(self.batch_size)

        obs = [val[0] for val in batch]
        obs = make_up_dict(obs)

        selected_targets = [val[1] for val in batch]
        rewards = [val[2] for val in batch]
        is_terminal = [val[3] for val in batch]

        # DQN is off-policy, so it needs to compute the max_a of the next state
        obs_next = [val[4] for val in batch]
        obs_next = make_up_dict(obs_next)

        obs["active_vehicle"] = np.array([[i, obs["active_vehicle"][i]] for i in batch_range])
        obs["selected_target"] = np.array([[i, selected_targets[i]] for i in batch_range])

        obs_next["active_vehicle"] = np.array([[i, obs_next["active_vehicle"][i]] for i in batch_range])

        # batch x nb
        next_q_values = self.dqn.value(obs=obs_next, sess=self.sess)
        # for not available targets, set the q value to a big negative number
        next_q_values -= 1000 * (1 - obs_next["available_targets"])

        # if self.rl_config.use_double_qn:
        best_action = np.argmax(next_q_values, axis=1).reshape([-1, 1])

        target_q_values = self.dqn.value_(obs=obs_next, sess=self.sess)

        max_next_q_values = [target_q_values[i][best_action[i]] for i in batch_range]
        # else:
        #     max_next_q_values = np.max(next_q_values, axis=1)

        max_next_q_values = np.array(max_next_q_values).reshape(-1)
        max_next_q_values[max_next_q_values < 0] = 0.

        discount_factor = np.array([0. if m else self.rl_config.gama ** self.rl_config.q_steps for m in is_terminal])

        q_target = rewards + discount_factor * max_next_q_values

        loss, avg_gradient = self.dqn.optimize(obs=obs, target=q_target, sess=self.sess)
        return loss, avg_gradient

    def learn_centralized(self, trials):
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

    def compute_actual_reward(self, k, x, is_preepmtive):
        n = self.env.ins_config.n

        if x == n:
            return 0
        else:
            selected_customer = self.env.customers[x]
            q = self.env.ins_config.capacity / Utils.Norms.Q if is_preepmtive else self.env.vehicles[k][3]

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
            # if selected_customer[5] == -1:
            #     # the demand is not realized yet
            #     w = self.env.demand_realization(x)
            #     self.env.demand_scenario[x] = w
            #     return min(w, q)
            # else:
            #     # the demand is realized and partially served before
            #     return min(selected_customer[5], q)
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
            served_demand = self.env.state_transition(k, is_test=True, time_diff=time - prev_time)

            prev_time = time

            # active vehicle k takes action
            _, x_k, preemptive, is_terminal, _ = self.choose_action(k, None, False)

            agent_reward[k] += served_demand
            agents_record[k].append(served_demand)

            if x_k != n:
                sum_exp_served += self.env.customers[x_k][-1]

            if preemptive:
                self.env.actions[k].append(-2)
                self.env.actions[k].append(x_k)
            else:
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
            t_k = self.env.post_decision(x_k, k, preemptive)

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

    def test_model_centralized(self, test_instance, scenario, need_reset=True):
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
            served_demand = self.env.state_transition(k, is_test=True, time_diff=time - prev_time)

            prev_time = time

            # active vehicle k takes action
            x_k, _, _, is_terminal = self.choose_action_centralized(k, None, False)

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
            t_k = self.env.post_decision(x_k, k, False)

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

class QLearningMin(QLearning):
    def __init__(self, env: vrp.DVRPSD, rl_config, sess, transfer_learning=False, feat_size_c=8, feat_size_v=6):
        super().__init__(env, rl_config, sess, transfer_learning, feat_size_c, feat_size_v)

        self.observation_function = self.observation_function_with_p
        self.choose_action = self.choose_action_with_p

    def observation_function_with_p(self, k):
        n = self.env.ins_config.n

        # take all feasible customers
        target_customers_dir, target_customers_indir, is_terminal = self.env.get_available_customers(k)
        target_customers = (target_customers_dir, target_customers_indir)

        c_set = np.array(self.env.c_enc)
        v_set = np.array(self.env.v_enc)

        # vehicles_pos_history = np.array(self.Env.vehicles_pos[k]) / norm.COORD
        # Normalized dl
        # rem_time = (self.env.ins_config.duration_limit / Utils.Norms.COORD - self.env.time)

        available_customers = np.zeros(n + 2)
        available_customers[:n] = self.env.customers[:, 3] > 0

        obs = {"customers": c_set, "vehicles": v_set, "active_vehicle": k + 0,
               "time": [self.env.time + 0.], "available_customers": available_customers,
               "instance_chars": self.env.instance_chars}

        return obs, target_customers, is_terminal

    def choose_action_with_p(self, k, trials, train=True):
        nb = self.env.ins_config.n + 1
        n = self.env.ins_config.n

        exp_coef = 16 / self.rl_config.max_trials
        if train:
            # epsilon = 1 - (0.9 * trials) / self.ep_max
            # epsilon = max(epsilon, 0.1)
            epsilon = np.exp(-trials * exp_coef) + 0.05 + 0.03 * (1 - trials / self.rl_config.max_trials)
            if self.tl:
                epsilon /= 2.
        else:
            epsilon = 0.

        loc_id = int(self.env.vehicles[k][-1])

        obs, (target_customers_dir, target_customers_indir), is_terminal = self.observation_function(k)
        n_actions_dir = len(target_customers_dir)

        # target_customers always has at least one item
        if n_actions_dir == 0:
            # depot is the only choice to travel
            # it means there is only one choice, so take it.
            best_action = n
            preemptive = False
            selected_index = 0
            obs["target_customers"] = np.array([n + 1] * nb)
            obs["target_customers"][0] = n
            obs["available_targets"] = np.zeros(2 * nb)
            obs["available_targets"][0] = 1

            # is_terminal = True
        else:
            #   to reduce the size of the action set, we restrict the set of customers to a set of target customers
            # if n_actions_dir > nb:
            #     # if n_actions_dir > 0:
            #     tt = np.array(target_customers_dir)
            #
            #     #   make sure the depot is out of the list
            #     if n in tt:
            #         tt = tt[:-1]
            #
            #     # compute the \rho value and take the top nb customers
            #     vv = np.minimum(self.env.customers[tt, -1], self.env.vehicles[k][3])
            #     vv /= self.env.distance_table[loc_id][tt]
            #     tt = tt[np.argsort(-vv)][:nb]
            #
            #     #   update the modified list of target direct and direct actions
            #     target_customers_dir = tt
            #     target_customers_indir = np.intersect1d(target_customers_dir, target_customers_indir)

            n_actions_dir = len(target_customers_dir)

            #   the set of target customers can be served directly, if they are less than nb, fill the list with
            #   the index of the dummy customer (n + 1)
            o_target_customers = [target_customers_dir[i] if i < n_actions_dir else n + 1
                                  for i in range(nb)]
            obs["target_customers"] = o_target_customers

            # Not all the direct target customers can be served indirectly, so flag their indirect service to zero.
            # this flag for dummy customers is always zero.
            available_targets_dir = np.zeros(nb)
            available_targets_dir[:n_actions_dir] = 1.
            available_targets_indir = np.zeros(nb)
            available_targets_indir[:n_actions_dir] = [m in target_customers_indir for m in target_customers_dir]

            #   combine them together - size: 2*nb
            available_targets = np.array([[available_targets_dir[m], available_targets_indir[m]]
                                          for m in range(nb)]).reshape(-1)

            obs["available_targets"] = available_targets

            obs["customers"][target_customers_dir, -1] = 1.
            if len(target_customers_indir) > 0:
                obs["customers"][target_customers_indir, -1] = 2.

            if random.random() <= epsilon:
                # explore
                sind = math.floor(random.random() * n_actions_dir)
                best_action = target_customers_dir[sind]

                if target_customers_dir[sind] in target_customers_indir:
                    if random.random() < 0.5:
                        selected_index = sind * 2
                        preemptive = False
                    else:
                        selected_index = sind * 2 + 1
                        preemptive = True
                else:
                    selected_index = sind * 2
                    preemptive = False

            else:
                obs_mk = {}
                for e1, e2 in obs.items():
                    obs_mk[e1] = np.expand_dims(e2, axis=0)
                obs_mk["active_vehicle"] = [[0, obs["active_vehicle"]]]

                q_values = self.dqn.value(obs=obs_mk, sess=self.sess)[0]
                q_values[q_values < 0.] = 0.
                q_values += (1 - available_targets) * (10e9)

                # temp: justify the q values with vv
                # vrate = (np.sort((vv - np.min(vv))/(np.max(vv) - np.min(vv))/200. + 1.)[::-1][:nb]).reshape(-1, 1)
                # vrate = np.concatenate([vrate, vrate], axis=1).reshape(-1)
                # vrate_z = np.zeros(nb*2)
                # vrate_z[:len(vrate)] = vrate
                # modified_q = q_values * vrate_z

                # if np.max(q_values) <= 0.01:
                #     self.zero_q_value += 1
                # else:
                #     self.zero_q_value = 0
                # if self.zero_q_value > 5000:
                #     print("5000 consecutive zero q values")
                #     self.zero_q_value = 0
                #     if train:
                #         self.stop = True
                # if train:
                #     # softmax
                #     q_values[np.invert(available_targets.astype(bool))] = -np.inf
                #     sm_q_values = Utils.softmax(q_values)
                #     selected_index = np.random.choice(range(len(q_values)), 1, p=sm_q_values)[0]
                #     if available_targets[selected_index] == 0:
                #         selected_index = 0
                # else:
                selected_index = np.argmin(q_values)
                # selected_index = np.argmax(modified_q)

                best_action = target_customers_dir[math.floor(selected_index / 2.)]
                if selected_index % 2. == 1:
                    preemptive = True
                else:
                    preemptive = False

        if best_action == n and self.env.vehicles[k][3] > 0.0001:
            is_terminal = 1

        return obs, int(best_action), preemptive, is_terminal, selected_index

    def learn(self, trials):
        """
        each batch contains:
        1- obs: C, V, time, vb, act veh position, target_customers, available_targets, available customers, inst chars
        2- selected_target (action, an index in available targets): ind -> batch,ind
        3- reward
        4- is_terminal
        5- obs_next: same items as obs

        Procedure:
        - a batch of experiences is sampled from the memory
        - with double_qn, in order to compute the max_{x} Q(s_{k+1}, x), we use the primary network to decide the best,
        but evaluate the Q value of that action with the target network.
        - the future rewards are discounted considering the n-step q learning.

        """

        if trials % self.replace_target_iter == 0:
            self.sess.run(self.dqn.replace_target_op)

        batch = self.memory.sample(self.batch_size)

        def make_up_dict(o, axis=None):
            key_list = ["customers", "vehicles", "active_vehicle", "time", "target_customers",
                        "available_targets", "available_customers", "instance_chars"]
            if axis is None:
                new_dict = dict((k, np.stack([v[k] for v in o])) for k in key_list)
            else:
                new_dict = dict((k, np.concatenate([v[k] for v in o], axis=axis)) for k in key_list)

            return new_dict

        batch_range = range(self.batch_size)

        obs = [val[0] for val in batch]
        obs = make_up_dict(obs)

        selected_targets = [val[1] for val in batch]
        rewards = [val[2] for val in batch]
        is_terminal = [val[3] for val in batch]

        # DQN is off-policy, so it needs to compute the max_a of the next state
        obs_next = [val[4] for val in batch]
        obs_next = make_up_dict(obs_next)

        obs["active_vehicle"] = np.array([[i, obs["active_vehicle"][i]] for i in batch_range])
        obs["selected_target"] = np.array([[i, selected_targets[i]] for i in batch_range])

        obs_next["active_vehicle"] = np.array([[i, obs_next["active_vehicle"][i]] for i in batch_range])

        # batch x nb
        next_q_values = self.dqn.value(obs=obs_next, sess=self.sess)
        # for not available targets, set the q value to a big positive number
        next_q_values += 1000 * (1 - obs_next["available_targets"])

        # if self.rl_config.use_double_qn:
        best_action = np.argmin(next_q_values, axis=1).reshape([-1, 1])

        target_q_values = self.dqn.value_(obs=obs_next, sess=self.sess)

        min_next_q_values = [target_q_values[i][best_action[i]] for i in batch_range]
        # else:
        #     max_next_q_values = np.max(next_q_values, axis=1)

        min_next_q_values = np.array(min_next_q_values).reshape(-1)
        min_next_q_values[min_next_q_values < 0] = 0.

        discount_factor = np.array([0. if m else self.rl_config.gama ** self.rl_config.q_steps for m in is_terminal])

        q_target = rewards + discount_factor * min_next_q_values

        loss, avg_gradient = self.dqn.optimize(obs=obs, target=q_target, sess=self.sess)
        return loss, avg_gradient

    def compute_actual_reward(self, k, x, is_preepmtive, is_terminal=False):
        # the objective is to min costs= travel time + over time costs
        # if time >= L: cost = tt * p_l
        # else:
        # if time + tt <= L: cost = tt
        # else: cost = (L - time) + (TT - L) * p_l

        time = self.env.time
        L = self.env.ins_config.duration_limit
        n = self.env.ins_config.n
        loc = int(self.env.vehicles[k][-1])
        if is_preepmtive:
            dist = self.env.distance_table[loc][n] + self.env.distance_table[n][x]
        else:
            dist = self.env.distance_table[loc][x]
        if time >= L:
            cost = dist * self.env.p_l
        else:
            if time + dist <= L:
                cost = dist
            else:
                cost = (L - time) + (time + dist - L) * self.env.p_l

        # if is_terminal:
        #     # termination of the trip
        #     assert x == n
        #
        #     # since the vehicle still has dist to go, it should be added to the final TT
        #     overtime = self.env.p_l * max(0,
        #                                   self.env.vehicles_traveled_time[k] + dist -
        #                                   self.env.ins_config.duration_limit)
        #     reward = dist + overtime
        # elif is_preepmtive:
        #     reward = self.env.distance_table[loc][-1] + self.env.distance_table[-1][x]
        #     # self.env.vehicles_traveled_time2[k] += reward
        # else:
        #     reward = self.env.distance_table[loc][x]
        #     # self.env.vehicles_traveled_time2[k] += reward

        return cost

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
            self.env.reset(instance=test_instance, scenario=scenario)

        self.customer_set_selection(int(50 * self.env.ins_config.real_n / 100.))
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
            self.env.state_transition(k, is_test=True, time_diff=time - prev_time)

            prev_time = time

            # active vehicle k takes action
            _, x_k, preemptive, is_terminal, _ = self.choose_action(k, None, False)

            reward = self.compute_actual_reward(k, x_k, preemptive)
            agent_reward[k] += reward
            agents_record[k].append(reward)

            if x_k != n:
                sum_exp_served += self.env.customers[x_k][-1]

            if preemptive:
                self.env.actions[k].append(-2)
                self.env.actions[k].append(x_k)
            else:
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
            t_k = self.env.post_decision(x_k, k, preemptive)

            # schedule the next event for vehicle k if it still has time
            if not is_terminal:
                self.env.time_table.append((k, t_k))

            final_reward += reward

        # final_reward *= Utils.Norms.Q
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

    def customer_set_selection(self, n=None):
        if n is None:
            # randomly remove 0% to 40% of customers
            n = int((random.random() * 40) * self.env.ins_config.real_n / 100.)
        m = 0
        while m < n:
            cn = int(random.random() * self.env.ins_config.real_n)
            if not (self.env.customers[cn][1] == 0 and self.env.customers[cn][2] == 0):
                m += 1
                self.env.customers[cn] = np.zeros(9)
                self.env.c_enc[cn] = np.zeros(7)

        self.env.ins_config.real_n -= m


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
