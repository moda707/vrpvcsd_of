import json
import math
import os
import random
import numpy as np
import tensorflow as tf

import instance_generator
import simulated_annealing
import vrp
import Utils
import rl


class Learner:
    def __init__(self, env, instances, test_instance, rl_config, gen_config, sess):
        """

        Args:
            env: Environment
            instances: set of instances to train on them (this set will be updated with new instances
                        during the learning)
            test_instance: One instance as validation instance
        """
        self.instances = instances
        self.env = env
        self.env.initialize_environment(instances[0])
        self.test_instance = test_instance
        self.max_trials = gen_config.trials
        self.gen_config = gen_config
        self.trials = 0
        rl_config.max_trials = self.max_trials

        self.model = rl.QLearning(env, rl_config, sess)

        self.code = self.gen_config.code

    def save_model(self):
        """
        Saves the trained model
        """
        self.model.save_network(self.gen_config.base_address + self.gen_config.code + "/final",
                                self.gen_config.code)

    def load_model(self):
        """
        Loads a trained model with a given code
        """
        new_saver = tf.compat.v1.train.Saver()
        # model_dir = f"{self.gen_config.base_address}{self.gen_config.code}"
        # model_dir = f"{self.gen_config.base_address}{self.gen_config.code}/final"
        model_dir = f"{self.gen_config.base_address}{self.gen_config.code}/final/{self.gen_config.code}"

        new_saver.restore(self.model.sess, tf.train.latest_checkpoint(model_dir))

    def train(self):
        rl_config = self.model.rl_config
        update_prob = rl_config.update_prob

        fr_list = []
        learning_loss = [0]
        n_period_result = self.gen_config.report_every

        counter = 0
        self.model.stop = False
        update_counter = 0
        gradient_record = []

        # generate a decaying strategy for learning rate and hubber loss parameter
        lr_decay_params = rl_config.lr_decay
        lr_decay = Utils.LinearSchedule(init_t=self.gen_config.trials // 10,
                                        end_t=2 * self.gen_config.trials // 3,
                                        init_val=lr_decay_params[2], end_val=lr_decay_params[3],
                                        update_every_t=lr_decay_params[4])
        new_lr = lr_decay.init_val
        hp_decay = Utils.LinearSchedule(init_t=self.gen_config.trials // 5, end_t=2 * self.gen_config.trials // 3,
                                        init_val=1., end_val=.25, update_every_t=30000)

        self.model.dqn.set_opt_param(new_lr=new_lr, new_hp=hp_decay.init_val, sess=self.model.sess)

        prev_avg_rewards = 0

        # for VRPSD, load test scenarios, otherwise (i.e., VRPVCSD), generate randomly when needed.
        if self.model.env.model_type == "VRPSD":
            # Be careful! This is only for VRPSD. test demand scenarios
            with open(f"Instances/VRPSD/realizations/realization_r101_1_100_"
                      f"{self.instances[0]['Config'].stoch_type}_500", 'r') as f:
                test_scenarios = json.load(f)
            test_scenarios = test_scenarios.values()
            test_scenarios = np.asarray([list(t.values()) for t in test_scenarios]) / Utils.Norms.Q
        else:
            test_scenarios = None

        # save the default models
        self.model.save_network(self.gen_config.base_address + self.code, self.code)
        self.model.save_network(self.gen_config.base_address + self.code + "/final", self.code)

        experience_buffer = []
        reset_distance_table = self.model.env.model_type == "VRPVCSD"
        time_tracker = Utils.TimeTracker()
        while not self.model.stop:
            # update the pool of instances
            if self.env.model_type == "VRPVCSD":
                if self.trials % 5000 == 0:
                    # generate 50 new instances and replace them in instances
                    new_instances = instance_generator.generate_VRPVCSD_instances_generalized(
                        instance_config=self.env.ins_config,
                        density_class_list=[self.env.ins_config.density_class],
                        capacity_list=[self.env.ins_config.capacity],
                        count=50)

                    # remove old ones
                    old_codes = [m["Name"] for m in self.instances[:50]]
                    del self.instances[:50]
                    self.instances.extend(new_instances)
                    for o in old_codes:
                        self.env.all_distance_tables.pop(o, None)

            # choose one instance to simulate
            if len(self.instances) == 1:
                instance = None
            else:
                ins_id = math.floor(random.random() * len(self.instances))
                instance = self.instances[ins_id]

            self.model.env.reset(instance=instance, reset_distance=reset_distance_table, scenario=None)

            # time scheduler
            for j in range(self.env.ins_config.m):
                self.model.env.TT.append((j, 0))

            self.model.env.final_reward = 0

            decision_epoch_counter = 0
            state, available_targets, selected_target_id = None, None, None

            # Simulation
            while len(self.model.env.TT) > 0:
                self.model.env.TT.sort(key=lambda x: x[1])
                v, stime = self.model.env.TT.pop(0)
                self.model.env.time = stime

                # transition from s^x_{k-1} to s_k. For k=0 it does nothing.
                served_demand = self.model.env.state_transition(v)

                # select action
                x_k, selected_target_id, state, available_targets, is_terminal = \
                    self.model.choose_action(v, self.trials, True)

                # compute the expected reward based on the vehicle's capacity, the customer's expected demand and
                # demand's distribution function
                # Note: I observed that this reward reduces the volatility and helps to converge faster
                reward_to_learn = self.model.compute_actual_reward(v, x_k)

                # record the action
                self.model.env.actions[v].append(x_k)

                # compute the n-step q value and store it in the experience replay
                if len(experience_buffer) > rl_config.q_steps - 1:
                    tmpr = sum([r[3] * (rl_config.gama ** ii) for ii, r in enumerate(experience_buffer)])
                    tmp_exp = experience_buffer.pop(0)
                    tmp_exp[3] = tmpr
                    tmp_exp.append(state)
                    tmp_exp.append(available_targets)
                    self.model.memory.add_sample(tmp_exp)
                # Experience: state, available_targets, action, reward, is_terminal, next_state, next_available_targets
                experience_buffer.append([state, available_targets, selected_target_id, reward_to_learn, is_terminal])

                t_k = self.model.env.post_decision(x_k, v)

                # schedule the next event for vehicle v if it still has time
                k_still_continue = t_k < self.env.ins_config.duration_limit and is_terminal == 0

                if k_still_continue:
                    self.model.env.TT.append((v, t_k))

                self.model.env.final_reward += served_demand
                decision_epoch_counter += 1

                # with update_prob probability at each decision epoch k, call learn function
                if random.random() < update_prob and self.model.memory.get_size() > 1000:
                    loss, gradient = self.model.learn(self.trials)

                    learning_loss.append(loss)
                    gradient_record.append(gradient)
                    update_counter += 1

                    if len(learning_loss) > 500:
                        learning_loss.pop(0)
                        gradient_record.pop(0)

            self.model.DecisionEpochs += decision_epoch_counter

            # end of trial
            self.model.env.time = self.env.ins_config.duration_limit
            experience_buffer.append([state, available_targets, selected_target_id, 0, 1])

            while len(experience_buffer) > 1:
                tmpr = sum([r[3] for r in experience_buffer])
                tmp_exp = experience_buffer.pop(0)
                tmp_exp[3] = tmpr
                tmp_exp.append(state)
                tmp_exp.append(available_targets)
                if tmp_exp[0] is not None and state is not None:
                    self.model.memory.add_sample(tmp_exp)

            fr_list.append(self.model.env.final_reward * Utils.Norms.Q)

            self.model.TrainTime += time_tracker.timeit()

            # update learning rate
            if lr_decay.update_time(self.trials):
                new_lr = lr_decay.val(self.trials)
                self.model.dqn.set_opt_param(new_lr=new_lr, sess=self.model.sess)

            # update huber parameter
            if hp_decay.update_time(self.trials):
                new_hp = hp_decay.val(self.trials)
                self.model.dqn.set_opt_param(new_hp=new_hp, sess=self.model.sess)

            self.trials += 1

            if self.trials % n_period_result == 0:
                avg_final_reward = np.mean(fr_list)
                print(f"{self.trials}\t{update_counter}\t{avg_final_reward:.2f}\t{self.model.TrainTime:.1f}\t"
                      f"{np.mean(learning_loss):.6f}")
                fr_list = []
                self.model.zero_q = 0

            # validation test
            if self.trials % self.model.rl_config.test_every == 0:
                if test_scenarios is None:
                    if self.model.env_config.model_type == "VRPVCSD":
                        # generate test scenarios
                        test_scenarios = [[self.model.env.generate_dem_real_gendreau(c[-1])
                                          for c in self.test_instance["Customers"]]
                                          for _ in range(500)]
                    else:
                        test_scenarios = []
                        print("test scenarios for VRPSD are not provided.")

                c_res = Utils.TestResults()

                for test_scenario in test_scenarios:
                    res = self.model.test_model(self.test_instance, test_scenario)
                    c_res.accumulate_results(res)
                c_res.print_avgs_full()
                avg_final_reward = c_res.get_avg_final_reward()
                if avg_final_reward > prev_avg_rewards:
                    prev_avg_rewards = avg_final_reward
                    self.model.save_network(self.gen_config.base_address + self.gen_config.code, self.gen_config.code,
                                            False)
                self.model.zero_q = 0

            # save the model every
            if self.trials % 1000 == 0:
                if self.model.tl:
                    saving_code = self.code
                else:
                    saving_code = self.gen_config.code + "/final"

                self.model.save_network(self.gen_config.base_address + saving_code,
                                        saving_code, False)

            counter += 1
            # stoppage criteria
            if self.trials > self.max_trials:
                self.model.stop = True

        return None

    def test(self, instance, visualize=False):
        """
        Test the trained model for a given instance
        Args:
            instance: test instance
            visualize: to visualize the result or not

        """
        c_res = Utils.TestResults()

        # load scenarios
        if self.env.model_type == "VRPSD":
            with open(f"Instances/VRPSD/realizations/realization_r101_1_100_{instance['Config'].stoch_type}_500", 'r') as f:
                scenarios = json.load(f)
            scenarios = scenarios.values()
            scenarios = np.asarray([list(t.values()) for t in scenarios]) / Utils.Norms.Q
        else:
            scenarios = None

        # to visualize, set a small subset of scenarios to solve and visualize them
        if visualize:
            if scenarios is None:
                for _ in range(3):
                    res = self.model.test_model(instance, None)
                    Utils.plot_environment(c=self.model.env.customers,
                                           v=res.actions,
                                           depot=[.50, .50],
                                           service_area_length=[1.00, 1.00],
                                           detail_level=3,
                                           animate=False)
                    res.print_full()
                    print(res.agent_record)
                    res.print_actions()
            else:
                for scenario in scenarios[:3]:
                    res = self.model.test_model(instance, scenario)
                    Utils.plot_environment(c=self.model.env.customers,
                                           v=res.actions,
                                           depot=[.35, .35],
                                           service_area_length=[0.8, 0.8],
                                           detail_level=3,
                                           animate=False)
                    res.print_full()
                    print(res.agent_record)
                    res.print_actions()
        else:
            # generate scenarios if not provided
            if scenarios is None:
                for _ in range(500):
                    res = self.model.test_model(instance, None)
                    c_res.accumulate_results(res)
            else:
                for scenario in scenarios:
                    res = self.model.test_model(instance, scenario)

                    c_res.accumulate_results(res)
            # c_res.print_avgs_full()
            return c_res.get_avg_final_reward()
