import json
import math
import os
import time

import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import (MultipleLocator)


def decentralize_p(value, min_val, max_val, n_step):
    value = max(min_val, value)
    value = min(value, max_val)
    if n_step == -1:
        r_val = value
    else:
        r_val = math.floor(value * n_step) / n_step

    return r_val


def get_zone_bound_by_id(zone_id, steps, service_area_length):
    zn = service_area_length / steps
    zone_x = math.floor(zone_id / steps)
    zone_y = zone_id - zone_x * steps
    x_bounds = [zone_x * zn, (zone_x + 1) * zn]
    y_bounds = [zone_y * zn, (zone_y + 1) * zn]
    return x_bounds, y_bounds


def get_zone_id_by_psi(psi, xy_offset=(0, 0), xy_length=None, xy_steps=None):
    step_length_x = xy_length[0] / xy_steps[0]
    step_length_y = xy_length[1] / xy_steps[1]

    psi_x = math.floor((psi[0] - xy_offset[0]) / step_length_x)
    if psi[0] == xy_length[0] + xy_offset[0]:
        psi_x -= 1

    psi_y = math.floor((psi[1] - xy_offset[1]) / step_length_y)
    if psi[1] == xy_length[1] + xy_offset[1]:
        psi_y -= 1

    return psi_x * xy_steps[1] + psi_y


def get_zone_tuple_by_psi(psi, xy_offset=(0, 0), xy_length=None, xy_steps=None):
    step_length_x = xy_length[0] / xy_steps[0]
    step_length_y = xy_length[1] / xy_steps[1]

    psi_x = math.floor((psi[0] - xy_offset[0]) / step_length_x)
    if psi[0] == xy_length[0] + xy_offset[0]:
        psi_x -= 1

    psi_y = math.floor((psi[1] - xy_offset[1]) / step_length_y)
    if psi[1] == xy_length[1] + xy_offset[1]:
        psi_y -= 1
    # return tuple([psi_x, psi_y])
    return psi_x, psi_y


def get_zone_id_by_tuple(zone, steps, xy_steps=None):
    if xy_steps is None:
        xy_steps = (steps, steps)

    # zone_id = x_id * y_steps + y_id
    return zone[0] * xy_steps[1] + zone[1]


def softmax(x):
    m = np.max(x)
    exp = np.exp(x - m)
    return exp / np.sum(exp)


def print_to_file(fname, result, is_new=False):
    if is_new:
        ftype = "w+"
    else:
        ftype = "a+"

    f = open(fname, ftype)
    f.write("\n" + result)
    f.close()


class LinearSchedule(object):
    """
    For scheduling the annealing of training hyperparameters.
    Args:
       init_t: step at which the annealing starts
       end_t : step at which the annealing end
       init_val: value of hyperparameter at init_t and before
       end_val: value of hyperparameter at end_t and after
       updated_every_t: updates happen only every update_every_t steps
    """

    def __init__(self, init_t, end_t, init_val, end_val, update_every_t):
        self.init_t = init_t
        self.end_t = end_t
        self.init_val = init_val
        self.end_val = end_val
        self.update_every_t = update_every_t

    def val(self, t):
        if t < self.init_t:
            return self.init_val
        if t > self.end_t:
            return self.end_val
        return ((t - self.init_t) * self.end_val + (self.end_t - t) * self.init_val) / float(self.end_t - self.init_t)

    def update_time(self, t):
        return t % self.update_every_t == 0 and self.init_t < t < self.end_t


class PWLinearSchedule(object):
    def __init__(self, time_seq, value_seq, update_every_t):
        self.time_seq = time_seq
        self.value_seq = value_seq
        self.update_every_t = update_every_t

    def val(self, t):

        tz = 0
        for i in self.time_seq:
            if t < i:
                break
            tz += 1
        if tz >= len(self.time_seq):
            tz = len(self.time_seq) - 1
            t = self.time_seq[tz]

        init_t = self.time_seq[tz - 1]
        end_t = self.time_seq[tz]
        init_val = self.value_seq[tz - 1]
        end_val = self.value_seq[tz]
        return ((t - init_t) * end_val + (end_t - t) * init_val) / float(end_t - init_t)

    def update_time(self, t):
        return t % self.update_every_t == 0 and t < self.time_seq[-1]


class Norms:
    Q = 50.
    DL = 100.
    COORD = 100.


def plot_environment(c, v, depot=(50, 50), c_id_modifier=0, service_area_length=(100, 100),
                     zone_step=(10, 10), offsets=(0, 0), detail_level=0,
                     title="", animate=False, plot_name="myplot"):
    fig = plt.figure(figsize=[13, 9.8])

    grid_space = service_area_length[0] / zone_step[0]
    ax = plt.axes(xlim=(offsets[0], offsets[0] + service_area_length[0]),
                  ylim=(offsets[1], offsets[1] + service_area_length[1]))
    v_x, v_y = {}, {}
    for i in range(len(v)):
        v_x[i] = []
        v_y[i] = []
        v[i] = [-2] + v[i]

        # v[i] = [depot[1]]

    c_x = [m[1] for m in c]
    c_y = [m[2] for m in c]

    vehicle = []
    pos_depot = [-2, len(c), -3]
    jet = cm = plt.get_cmap('tab10')
    cNorm = colors.Normalize(vmin=0, vmax=len(v))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for j in range(len(v)):
        colorVal = scalarMap.to_rgba(j)
        tmp_v, = ax.plot([], [], lw=1, color=colorVal)
        vehicle.append(tmp_v)

    # plot customers
    cusp, = ax.plot(c_x, c_y, 'o', color='black', linewidth=1)

    # Grids
    ax.xaxis.set_major_locator(MultipleLocator(grid_space))
    ax.yaxis.set_major_locator(MultipleLocator(grid_space))
    # plt.plot(c_x, c_y, 'ro', v_x, v_y, 'bs')
    # plt.axis([0, max_x, 0, max_y])
    plt.grid(True)
    plt.title = title

    # annotate customers
    for n in c:
        if detail_level == 0:
            ax.text(n[1], n[2], str(int(n[0])))
        elif detail_level == 1:
            ax.text(n[1], n[2], str(int(n[0])) + "(" + str(int(n[4])) + ")")
        elif detail_level == 2:
            ax.text(n[1], n[2], str(int(n[0])) + " (" + str(n[4]) + "," + str(n[6]) + ")")
        elif detail_level == 3:
            ax.text(n[1], n[2],
                    str(int(n[0])) + " (" + str(n[4]) + ", " + str(round(n[6], 2)) + ", " + str(round(n[6] - n[5], 2)) + ")")
        else:
            print("Unexpected detail_level")

    if animate:
        def init():
            for r in vehicle:
                r.set_data([], [])
            return vehicle

        def animate_func(t):
            for ev, r in enumerate(vehicle):
                if t < len(v[ev]):
                    new_point = v[ev][t]
                    v_x[ev].append(depot[0] if new_point in pos_depot else c[new_point][1])
                    v_y[ev].append(depot[1] if new_point in pos_depot else c[new_point][2])
                    r.set_data(v_x[ev], v_y[ev])
            # vehicle.set_data(nnn[0, :t], nnn[1, :t])
            return vehicle

        num_frames = max([len(m) for m in v.values()])
        anim = FuncAnimation(fig, func=animate_func, init_func=init,
                             frames=num_frames, interval=3000, blit=True, repeat=False)
        anim.save(plot_name + ".gif", writer='imagemagick')
    else:
        for i in range(len(v)):
            v_x[i] = [depot[0] if m in pos_depot else c[abs(m)][1] for m in v[i]]
            v_y[i] = [depot[1] if m in pos_depot else c[abs(m)][2] for m in v[i]]

            colorVal = scalarMap.to_rgba(i)
            v_1 = ax.plot(v_x[i], v_y[i], linewidth=1, color=colorVal)
            vehicle.append(v_1)

    plt.show()


def plot_customers(c, service_area=80, zone_step=5):
    fig = plt.figure()
    v_x = {}
    v_y = {}
    grid_space = service_area / zone_step
    ax = [fig.add_subplot(111)]
    c_x = [m[1] for m in c]
    c_y = [m[2] for m in c]
    cusp, = ax[0].plot(c_x, c_y, 'o', color='black', linewidth=1)
    dep, = ax[0].plot(service_area / 2, service_area / 2, "gs", color='green', linewidth=3)
    max_x = service_area
    max_y = service_area
    plt.axis([0, max_x, 0, max_y])
    # plt.grid(which='both')
    # Change major ticks to show every 20.
    ax[0].xaxis.set_major_locator(MultipleLocator(grid_space))
    ax[0].yaxis.set_major_locator(MultipleLocator(grid_space))

    # # Change minor ticks to show every 5. (20/4 = 5)
    # ax[0].xaxis.set_minor_locator(AutoMinorLocator(4))
    # ax[0].yaxis.set_minor_locator(AutoMinorLocator(4))

    plt.grid(which='major')
    # plt.grid(which='minor')
    for n in c:
        # ax[0].text(n[1], n[2], str(n[0]) + " (" + str(n[4]) + ", " + str(n[6]) + ", " + str(n[6] - n[5]) + ")")
        ax[0].text(n[1], n[2], str(n[0]))

    plt.show()


class TestResults:
    def __init__(self, final_reward=0, actions=[], n_routes=0, n_served=0, n_visits=0, avg_travel_time=0,
                 expected_demand=0, max_travel_time=0, agents_reward=[]):
        self.final_reward = final_reward
        self.actions = actions
        self.n_routes = n_routes
        self.n_served = n_served
        self.n_visits = n_visits
        self.avg_travel_time = avg_travel_time
        self.expected_demand = expected_demand
        self.max_travel_time = max_travel_time
        self.n_preemptives = 0
        self.results_count = 0
        self.agent_reward = agents_reward
        self.tot_realized_demand = 0.
        self.service_rate = 0.

    def print_full(self, title=False):
        if title:
            print("FR", "#routes", "#served", "#visited", "#avg_tt", "#max_tt", "#preemp", "exp_dem")
        print(self.final_reward, self.n_routes, self.n_served, self.n_visits, self.avg_travel_time,
              self.max_travel_time, self.n_preemptives, self.expected_demand, self.agent_reward,
              self.service_rate)

    def print_actions(self):
        for m in self.actions.items():
            print(m[0], ":", m[1])

    def accumulate_results(self, results):
        self.final_reward += results.final_reward
        self.n_routes += results.n_routes
        self.n_served += results.n_served
        self.n_visits += results.n_visits
        self.avg_travel_time += results.avg_travel_time
        self.expected_demand += results.expected_demand
        self.max_travel_time += results.max_travel_time
        self.n_preemptives += results.n_preemptives
        self.agent_reward = [self.agent_reward[i] + results.agent_reward[i] for i in range(len(self.agent_reward))]
        self.results_count += 1
        self.tot_realized_demand += results.tot_realized_demand
        self.service_rate += results.service_rate

    def print_avgs_full(self, title=False):
        if title:
            print("FR", "#routes", "#served", "#visited", "#avg_tt", "#max_tt", "#preemp", "exp_dem")
        print(round(self.final_reward / self.results_count, 2),
              round(self.n_routes / self.results_count, 2),
              round(self.n_served / self.results_count, 2),
              round(self.n_visits / self.results_count, 2),
              round(self.avg_travel_time / self.results_count, 2),
              round(self.max_travel_time / self.results_count, 2),
              round(self.n_preemptives / self.results_count, 2),
              round(self.expected_demand / self.results_count, 2),
              np.round(np.array(self.agent_reward) / self.results_count, 2))

    def get_avg_final_reward(self):
        return round(self.final_reward / self.results_count, 2)

    def get_avg_tot_realized_demand(self):
        return round(self.tot_realized_demand / self.results_count, 2)

    def get_avg_service_rate(self):
        return round(self.service_rate / self.results_count, 2)


class TimeTracker:
    def __init__(self):
        self.prev_time = {0: time.time()}

    def initiate_timer(self, tid):
        if tid in self.prev_time:
            print("timer id exists")
        self.prev_time[tid] = time.time()

    def timeit(self, tid=0):
        diff = time.time() - self.prev_time[tid]
        self.prev_time[tid] = time.time()
        return diff


def str_to_arr(str, ptype=float):
    str = str.replace(']', '').replace('[', '')
    l = str.replace('"', '').split(",")
    l = [ptype(m) for m in l]
    return l


class TrainConfig(object):
    def __init__(self, note, code, n_customers, n_vehicles, capacity, duration_limit, stoch_type, env, trials,
                 base_address, configfile=None):
        self.note = note
        self.code = code
        self.n_customers = n_customers
        self.n_vehicles = n_vehicles
        self.capacity = capacity
        self.dl = duration_limit
        self.stoch_type = stoch_type
        self.env = env
        self.trials = trials
        self.base_address = base_address

        self.instance_class = None
        self.instance_count = None
        self.zone_step = 10
        self.service_area = None
        self.nb = 10

        self.memory_size = 5000
        self.batch_size = 32
        self.ep_max = 500000
        self.update_prob = 0.04
        self.lr_decay = [50000, 1200000, 0.0005, 0.00005, 5000]

        if configfile is not None:
            self.set_config(configfile)

    def set_config(self, configfile):
        env_config = configfile["Environment"]
        self.instance_class = env_config["instance_class"]
        self.instance_count = env_config["instance_count"]
        self.zone_step = env_config["zone_step"]
        self.service_area = env_config["service_area"]
        self.nb = env_config["nb"]

        rl_config = configfile["RL"]
        self.memory_size = rl_config["memory_size"]
        self.batch_size = rl_config["batch_size"]
        self.ep_max = rl_config["ep_max"]
        self.update_prob = rl_config["update_prob"]
        self.lr_decay = rl_config["rl_decay"]

    def print(self):
        os.makedirs(self.base_address + self.code)
        with open(self.base_address + self.code + "/configFile.txt", 'w') as file:
            file.write(json.dumps(self.__dict__, indent=1))


def expected_demand(c, stoch_type, q):
    dem = c[-1]
    if c[5] > -1:
        if dem > q:
            return q
        else:
            return dem

    coefs = [0, 0.5, 1., 1.5, 2.]
    probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    if stoch_type == 1:
        coefs = [0, 0.5, 1., 1.5, 2.]
        probs = [0.05, 0.15, 0.6, 0.15, 0.05]
    elif stoch_type == 0:
        coefs = [0.5, 1., 1.5]
        probs = [0.05, 0.9, 0.05]

    return np.matmul(np.clip([m * dem for m in coefs], a_min=None, a_max=q), probs)


def condense_index(i, j, n):
    if i == j:
        return 0
    if i < j:
        i, j = j, i
    return n * j - j * (j + 1) / 2 + i - 1 - j
