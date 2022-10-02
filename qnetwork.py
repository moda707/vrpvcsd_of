import numpy as np
import tensorflow as tf
from tensorflow import keras


def fancy_clip(grad, low, high):  # for gradient clipping without error
    if grad is None:
        return grad

    return tf.clip_by_value(grad, low, high)


class DQN(object):
    """

    Inputs: Customers, Vehicles, Previous_route, time, target_customers
    Output: Q value for a location
    The size of the target customers should be action_dim, if there are less customers, refer to customer n as a dummy.
    Operations:
    1- Embed customers and vehicles,
    2- Attention on customers to find their importance to each other
    3- Attention on vehicles to find their importance to customers
    4- Concat two attentions as h_i
    5- Encode the previous route by RNN (hidden state) as p_j
    6- Average h_i and p_j over n and m to find H and P
    7- make the context as C=[H, P, V', time] where V' is the embedding of the active vehicle
    8- Attention to find the the importance of customers for the context C.
    9- The output is Q values


    Notes:
        1- Each batch represents a vector of q values for all nodes in target_customers.
        2- customer set C has n+2 nodes, 0 to n-1 nodes are referring to customers, n is the depot and n+1 is a dummy
        node with values zero()
        3- The q value for dummy target customers is masked to zero.

        4- c -> x -> self-attention -> attention respect to vbar
    """

    def __init__(self, init_lr, emb_size, feat_size_c, feat_size_v, pos_hist=3, nb=15, use_path=True):
        self.emb_size = emb_size
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v
        self.nb = nb

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # in placeholders, the first axis is the batch size
        # batch x n_customers (height) x features_size (width)
        self.customers = tf.placeholder(tf.float32, shape=[None, None, feat_size_c], name="raw_c")

        # batch x n_vehicles x feature_size_vehicle
        self.vehicles = tf.placeholder(tf.float32, shape=[None, None, feat_size_v], name="raw_v")

        # batch x nb
        self.target_customers = tf.placeholder(tf.int32, shape=[None, nb], name="target_customers_ind")

        # batch x nb: real targets=1, dummy targets=0.
        self.available_targets = tf.placeholder(tf.float32, shape=[None, nb], name="available_targets")

        # batch x n_customers: [1, 1, 0, 1, ...] => |C| + 1 + 1
        self.available_customers = tf.placeholder(tf.float32, shape=[None, None], name="available_customers")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        # batch x 2 (batch_id, vehicle_id)
        self.active_vehicle = tf.placeholder(tf.int32, [None, 2], name='active_vehicle_ind')

        # batch x pos_hist (3) x position size (x, y)
        self.vehicles_pos = tf.placeholder(tf.float32, [None, pos_hist, 2], name='vehicles_pos')

        self.time = tf.placeholder(tf.float32, [None, 1], name='time')

        # instance characteristics: DL, Q, SV
        self.instance_chars = tf.placeholder(tf.float32, [None, 5], name='instance_chars')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_embedding(customers_set, vehicles_set, available_customers, w_init, class_name, heads=4):
            #   Transformer Encoder for customers: multi head self attention mechanism
            wq = tf.get_variable('wq_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # Query: batch x n x e
            hbar_c = tf.matmul(customers_set, wq)

            # e'
            head_size = int(self.emb_size / heads)

            # Query: batch x n x e
            q = hbar_c

            # Key: batch x n x e
            k = tf.matmul(customers_set, wk)

            # Value: batch x n x e
            v = tf.matmul(customers_set, wv)

            # reshape to split
            batch_size, n_customers = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x n x e'
            q = tf.transpose(tf.reshape(q, [batch_size, n_customers, heads, head_size]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_customers, heads, head_size]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_customers, heads, head_size]), [0, 2, 1, 3])

            ll = head_size * 1.

            # compatibility weights: batch x h x n x e' * batch x h x e' x n: batch x h x n x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # eliminate the unavailable customers: batch x n x n ** batch x n
            a += tf.broadcast_to(tf.expand_dims(tf.expand_dims((1. - available_customers) * (-1e9), axis=1), axis=1),
                                 [batch_size, heads, n_customers, n_customers])

            # softmax: batch x h x n x n
            a = tf.nn.softmax(a, axis=-1)

            # embedding: batch x h x n x n * batch x h x n x e' -> batch x h x n x e' ->
            # batch x n x h x e' -> batch x n x e
            h_c = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, n_customers, self.emb_size])

            # Feed-forward:
            h_c += hbar_c

            # vehicle embedding
            wv2 = tf.get_variable('wv_emb', [self.feat_size_v, self.emb_size], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)
            h_v = tf.matmul(vehicles_set, wv2, name="vehicles_emb")
            # x_v = None
            return h_c, h_v

        def build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name):

            w1 = tf.get_variable('w1_att_g', [h_z_size + self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            w2 = tf.get_variable('w2_att_g', [self.emb_size, 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            batch_size, n_customers = tf.shape(h_c)[0], tf.shape(h_c)[1]
            # batch x e' -> batch x 1 x e' -> batch x n x e'
            context_extended = tf.broadcast_to(tf.expand_dims(h_z, axis=1),
                                               [batch_size, n_customers, h_z_size])

            # batch x n x (e + e')
            mu = tf.concat([context_extended, h_c], axis=-1)

            # batch x n
            weights = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(mu, w1)), w2), [batch_size, n_customers])

            # Masking
            weights += (1. - available_customers) * (-1e9)
            # batch x n: batch x 1 x n
            weights = tf.expand_dims(tf.nn.softmax(weights, axis=-1), axis=1)

            # batch x 1 x n * batch x n x e -> batch x 1 x e -> batch x e
            h_g = tf.reshape(tf.matmul(weights, h_c), [batch_size, self.emb_size])

            return h_g

        def build_fc_q_network(obs, obs_size, h_c, target_customers, w_init, b_init, class_name):
            batch_size = tf.shape(obs)[0]

            batch_range = tf.expand_dims(
                tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=1), [batch_size, self.nb]),
                axis=2)
            # batch x nb x 2
            indices = tf.concat([batch_range, tf.expand_dims(target_customers, axis=2)], axis=2)

            # gather the embedding of target customers for each batch: batch x nb x e
            h_c_t = tf.gather_nd(h_c, indices, name="gather_emb")

            # batch x nb x state_size
            bstate = tf.broadcast_to(tf.expand_dims(obs, axis=1), [batch_size, self.nb, obs_size])

            # batch x nb x (state_size + emb_size)
            obs_action = tf.concat([bstate, h_c_t], axis=2)
            oa_size = obs_size + self.emb_size
            h1 = int(oa_size / 2.)

            w1 = tf.get_variable('w_fcd_1', [oa_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name)

            q_value = tf.nn.relu(tf.matmul(obs_action, w1) + b1)

            w2 = tf.get_variable('w_fcd_2', [h1, 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b2 = tf.get_variable('b_fcd_2', [1, 1], initializer=b_init, collections=class_name)

            # batch x nb x 1
            q_value = tf.matmul(q_value, w2) + b2

            # batch x nb
            q_value = tf.reshape(q_value, [batch_size, self.nb])

            return q_value

        def build_position_history_rnn(vehicles_pos, w_init, class_name):
            w_p = tf.get_variable('w_pos', [2, 8], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)
            # batch x m x hist x 8
            x_p = tf.matmul(vehicles_pos, w_p)

            w = tf.get_variable('w_pos_rnn', [8, self.emb_size], initializer=w_init,
                                collections=class_name, dtype=tf.float32)
            w2 = tf.get_variable('w2_pos_rnn', [self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            u = tf.get_variable('u_pos_rnn', [self.emb_size, self.emb_size], initializer=w_init,
                                collections=class_name, dtype=tf.float32)

            p = None
            # pos_hist = 1
            for i in range(pos_hist):
                # batch x m x 8
                l = x_p[:, i, :]

                if i == 0:
                    p = tf.matmul(tf.nn.tanh(tf.matmul(l, w)), w2)
                else:
                    p = tf.matmul(tf.nn.tanh(tf.matmul(l, w) + tf.matmul(p, u)), w2)

            return p

        def build_network(customers_set, vehicles_set, active_vehicle, target_customers, available_targets,
                          available_customers, vehicles_pos, time, instance_chars,
                          w_init, b_init, class_name):
            # embed the raw data: batch x n x e, batch x m x e
            h_c, h_v = build_embedding(customers_set, vehicles_set, available_customers, w_init, class_name, heads=4)

            # embedding of the active vehicle: batch x e
            h_v_vbar = tf.gather_nd(h_v, active_vehicle, name="active_vehicle_emb")

            # pooling for vehicles
            hbar_v = tf.reduce_sum(h_v, axis=1, name="x_v_pooling")

            if use_path:
                # build past route rnn: batch x m x e
                p_k = build_position_history_rnn(vehicles_pos, w_init, class_name)

                # batch x (3e + 6)
                h_z = tf.concat([h_v_vbar, p_k, hbar_v, time, instance_chars], axis=1)
                h_z_size = 3 * self.emb_size + 6

            else:
                h_z = tf.concat([h_v_vbar, hbar_v, time, instance_chars], axis=1)
                h_z_size = 2 * self.emb_size + 6

            h_g = build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name)

            obs = tf.concat([h_g, h_z], axis=1, name="concat_state")
            obs_size = h_z_size + self.emb_size

            # we use x_c_e as the target customer
            # batch x nb
            q_values = build_fc_q_network(obs, obs_size, h_c, target_customers, w_init, b_init, class_name)

            return q_values, obs

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval, self.state = build_network(self.customers, self.vehicles,
                                                    self.active_vehicle, self.target_customers, self.available_targets,
                                                    self.available_customers, self.vehicles_pos, self.time,
                                                    self.instance_chars,
                                                    w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            # self.td_error = self.q_target - selected_q
            # self.loss = tf.reduce_mean(tf.square(self.td_error))
            huber = tf.keras.losses.Huber(delta=self.huber_p)
            self.loss = tf.reduce_mean(huber(self.q_target, selected_q))
            # self.loss = tf.reduce_sum(tf.math.log(tf.math.cosh(self.td_error)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
            self.gradient = self.optimizer.compute_gradients(self.loss, var_list=var_list)
            # self.gradient = [(tf.clip_by_norm(grad, 5), var) for grad, var in self.gradinet]
            self.opt = self.optimizer.apply_gradients(self.gradient)
            # self.opt = self.optimizer.minimize(self.loss, var_list=var_list)

        # ------------------ build target_net ------------------

        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_network(self.customers, self.vehicles,
                                        self.active_vehicle, self.target_customers, self.available_targets,
                                        self.available_customers, self.vehicles_pos, self.time,
                                        self.instance_chars,
                                        w_initializer, b_initializer, c_names)[0]

        # ------------------ replace network ------------------
        t_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def optimize(self, obs, target, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.available_customers: obs["available_customers"],
                     self.vehicles_pos: obs["vehicles_pos"],
                     self.time: obs["time"],
                     self.selected_target: obs["selected_target"],
                     self.instance_chars: obs["instance_chars"],
                     self.q_target: target
                     }
        opt, loss, gradient = sess.run([self.opt, self.loss, self.gradient], feed_dict=feed_dict)
        avg_gradient = np.mean([item for g in gradient for item in g[0].reshape(-1)])
        return loss, avg_gradient

    def value(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.vehicles_pos: obs["vehicles_pos"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_eval, feed_dict=feed_dict)
        # values, mm, ss = sess.run([self.q_eval, self.embeddings, self.state], feed_dict=feed_dict)

        # batch x n_target_customers
        return values

    def value_(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.vehicles_pos: obs["vehicles_pos"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_next, feed_dict=feed_dict)

        return values

    def set_opt_param(self, sess, new_lr=None, new_hp=None):
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_hp is not None:
            sess.run(tf.assign(self.huber_p, new_hp))

    def get_learning_params(self, sess):
        # learning rate and huber param
        return sess.run([self.lr, self.huber_p])

    @staticmethod
    def expand_dim(a):
        return np.expand_dims(a, axis=0)


# preemptive q value for each action
class DQNP(object):
    """

    Inputs: Customers, Vehicles, Previous_route, time, target_customers
    Output: Q value for a location
    The size of the target customers should be action_dim, if there are less customers, refer to customer n as a dummy.
    Operations:
    1- Embed customers and vehicles,
    2- Attention on customers to find their importance to each other
    3- Attention on vehicles to find their importance to customers
    4- Concat two attentions as h_i
    5- Encode the previous route by RNN (hidden state) as p_j
    6- Average h_i and p_j over n and m to find H and P
    7- make the context as C=[H, P, V', time] where V' is the embedding of the active vehicle
    8- Attention to find the the importance of customers for the context C.
    9- The output is Q values


    Notes:
        1- Each batch represents a vector of q values for all nodes in target_customers.
        2- customer set C has n+2 nodes, 0 to n-1 nodes are referring to customers, n is the depot and n+1 is a dummy
        node with values zero()
        3- The q value for dummy target customers is masked to zero.

        4- c -> x -> self-attention -> attention respect to vbar
    """

    def __init__(self, init_lr, emb_size, feat_size_c, feat_size_v, pos_hist=3, nb=15, use_path=True):
        self.emb_size = emb_size
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v
        self.nb = nb

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # in placeholders, the first axis is the batch size
        # batch x n_customers (height) x features_size (width)
        self.customers = tf.placeholder(tf.float32, shape=[None, None, feat_size_c], name="raw_c")

        # batch x n_vehicles x feature_size_vehicle
        self.vehicles = tf.placeholder(tf.float32, shape=[None, None, feat_size_v], name="raw_v")

        # batch x nb
        self.target_customers = tf.placeholder(tf.int32, shape=[None, nb], name="target_customers_ind")

        # batch x nb: real targets=1, dummy targets=0.
        self.available_targets = tf.placeholder(tf.float32, shape=[None, 2*nb], name="available_targets")

        # batch x n_customers: [1, 1, 0, 1, ...] => |C| + 1 + 1
        self.available_customers = tf.placeholder(tf.float32, shape=[None, None], name="available_customers")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        # batch x 2 (batch_id, vehicle_id)
        self.active_vehicle = tf.placeholder(tf.int32, [None, 2], name='active_vehicle_ind')

        # distance table: batch x n_customers x n_customers
        self.distance_table = tf.placeholder(tf.float32, [None, None, None], name='distance_table')

        # batch x n_vehicles x pos_hist (3) x position size (x, y)
        # self.vehicles_pos = tf.placeholder(tf.float32, [None, None, pos_hist, 2], name='vehicles_pos')
        self.vehicles_pos = tf.placeholder(tf.float32, [None, pos_hist, 2], name='vehicles_pos')

        self.time = tf.placeholder(tf.float32, [None, 1], name='time')

        # instance characteristics: DL, Q, SV
        self.instance_chars = tf.placeholder(tf.float32, [None, 5], name='instance_chars')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_embedding(customers_set, vehicles_set, available_customers, w_init, class_name,
                            heads=4):
            # customers, node embedding
            wq = tf.get_variable('wq_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # Query: batch x n x e
            x_c = tf.matmul(customers_set, wq)

            # e'
            hsize = int(self.emb_size / heads)

            # Query: batch x n x e
            q = x_c

            # Key: batch x n x e
            k = tf.matmul(customers_set, wk)

            # Value: batch x n x e
            v = tf.matmul(customers_set, wv)

            # reshape to split
            batch_size, n_customers = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x n x e'
            q = tf.transpose(tf.reshape(q, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x n x e' * batch x h x e' x n: batch x h x n x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # involve the distance table
            # ww = tf.get_variable('wd1_att_cc', [1, heads], initializer=w_init, collections=class_name, dtype=tf.float32)
            # distance_table = tf.reshape(tf.matmul(tf.expand_dims(distance_table, axis=-1), ww),
            #                             [batch_size, heads, n_customers, n_customers, 1])
            # a = tf.concat([tf.expand_dims(a, axis=-1), distance_table], axis=-1)
            # wd = tf.get_variable('wd_att_cc', [2, 1], initializer=w_init, collections=class_name, dtype=tf.float32)
            # # batch x h x n x n
            # a = tf.reshape(tf.matmul(a, wd), [batch_size, heads, n_customers, n_customers])

            # eliminate the unavailable customers: batch x n x n ** batch x n

            a += tf.broadcast_to(tf.expand_dims(tf.expand_dims((1. - available_customers) * (-1e9), axis=1), axis=1),
                                 [batch_size, heads, n_customers, n_customers])

            # softmax: batch x h x n x n
            a = tf.nn.softmax(a, axis=-1)

            # graph embedding: batch x h x n x n * batch x h x n x e' -> batch x h x n x e' ->
            # batch x n x h x e' -> batch x n x e
            x_c_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, n_customers, self.emb_size])

            # Feed-forward:
            x_c_e += x_c

            # vehicle embedding
            wv2 = tf.get_variable('wv_emb', [self.feat_size_v, self.emb_size], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)
            x_v = tf.matmul(vehicles_set, wv2, name="vehicles_emb")
            # x_v = None
            return x_c_e, x_v

        def build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name):

            w1 = tf.get_variable('w1_att_g', [h_z_size + self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            w2 = tf.get_variable('w2_att_g', [self.emb_size, 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            batch_size, n_customers = tf.shape(h_c)[0], tf.shape(h_c)[1]
            # batch x e' -> batch x 1 x e' -> batch x n x e'
            context_extended = tf.broadcast_to(tf.expand_dims(h_z, axis=1),
                                               [batch_size, n_customers, h_z_size])

            # batch x n x (e + e')
            mu = tf.concat([context_extended, h_c], axis=-1)

            # batch x n
            weights = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(mu, w1)), w2), [batch_size, n_customers])

            # Masking
            weights += (1. - available_customers) * (-1e9)
            # batch x n: batch x 1 x n
            weights = tf.expand_dims(tf.nn.softmax(weights, axis=-1), axis=1)

            # batch x 1 x n * batch x n x e -> batch x 1 x e -> batch x e
            h_g = tf.reshape(tf.matmul(weights, h_c), [batch_size, self.emb_size])

            return h_g

        def build_fc_q_network(state, state_size, x_c, target_customers,
                               w_init, b_init, class_name):
            batch_size = tf.shape(state)[0]

            brange = tf.expand_dims(
                tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=1), [batch_size, self.nb]),
                axis=2)
            # batch x nb x 2
            indices = tf.concat([brange, tf.expand_dims(target_customers, axis=2)], axis=2)

            # gather the embedding of target customers for each batch: batch x nb x e
            x_t = tf.gather_nd(x_c, indices, name="gather_emb")

            # batch x nb x state_size
            bstate = tf.broadcast_to(tf.expand_dims(state, axis=1), [batch_size, self.nb, state_size])

            # batch x nb x (state_size + emb_size)
            state_actions = tf.concat([bstate, x_t], axis=2)
            state_size += self.emb_size
            h1 = int(state_size / 2.)

            w1 = tf.get_variable('w_fcd_1', [state_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name)

            # q_value = tf.nn.leaky_relu(tf.matmul(state_actions, w1) + b1)
            q_value = tf.nn.relu(tf.matmul(state_actions, w1) + b1)

            w2 = tf.get_variable('w_fcd_2', [h1, 2], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b2 = tf.get_variable('b_fcd_2', [1, 2], initializer=b_init, collections=class_name)

            # batch x nb x 1
            # q_value = tf.nn.leaky_relu(tf.matmul(q_value, w2) + b2)
            q_value = tf.matmul(q_value, w2) + b2

            # batch x nb*2
            q_value = tf.reshape(q_value, [batch_size, 2*self.nb])

            # mask out dummy targets
            # q_value = q_value * available_targets
            return q_value

        def build_position_history_rnn(vehicles_pos, w_init, class_name):
            w_p = tf.get_variable('w_pos', [2, 8], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)
            # batch x m x hist x 8
            x_p = tf.matmul(vehicles_pos, w_p)

            w = tf.get_variable('w_pos_rnn', [8, self.emb_size], initializer=w_init,
                                collections=class_name, dtype=tf.float32)
            w2 = tf.get_variable('w2_pos_rnn', [self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            u = tf.get_variable('u_pos_rnn', [self.emb_size, self.emb_size], initializer=w_init,
                                collections=class_name, dtype=tf.float32)

            p = None
            # pos_hist = 1
            for i in range(pos_hist):
                # batch x m x 8
                l = x_p[:, i, :]

                if i == 0:
                    p = tf.matmul(tf.nn.tanh(tf.matmul(l, w)), w2)
                else:
                    p = tf.matmul(tf.nn.tanh(tf.matmul(l, w) + tf.matmul(p, u)), w2)

            return p

        def build_network(customers_set, vehicles_set, active_vehicle, target_customers,
                          available_customers, vehicles_pos, time, instance_chars,
                          w_init, b_init, class_name):
            # There is a mistake in the old trained policies. The order of items in h_z when passing to the GNN
            # is [h_v_vbar, p_k, hbar_v, time, instance_chars],
            # while in the obs is [hbar_v, p_k, h_v_bar, time, instance_chars], It is fixed in the new models
            new_version = True

            # embed the raw data: batch x n x e, batch x m x e
            h_c, h_v = build_embedding(customers_set, vehicles_set, available_customers,
                                       w_init, class_name, heads=4)

            # embedding of the active vehicle: batch x e
            h_v_vbar = tf.gather_nd(h_v, active_vehicle, name="active_vehicle_emb")

            # pooling for vehicles
            hbar_v = tf.reduce_sum(h_v, axis=1, name="x_v_pooling")

            p_k = None
            if use_path:
                # build past route rnn: batch x m x e
                p_k = build_position_history_rnn(vehicles_pos, w_init, class_name)

                # batch x (3e + 6)
                h_z = tf.concat([h_v_vbar, p_k, hbar_v, time, instance_chars], axis=1)
                h_z_size = 3 * self.emb_size + 6

            else:
                h_z = tf.concat([h_v_vbar, hbar_v, time, instance_chars], axis=1)
                h_z_size = 2 * self.emb_size + 6

            h_g = build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name)

            if new_version:
                obs = tf.concat([h_g, h_z], axis=1, name="concat_state")
            else:
                # In old version, the order of items is different from h_z
                obs = tf.concat([h_g, hbar_v, p_k, h_v_vbar, time, instance_chars], axis=1, name="concat_state")

            obs_size = h_z_size + self.emb_size

            # we use x_c_e as the target customer
            # batch x nb x 2
            q_values = build_fc_q_network(obs, obs_size, h_c, target_customers, w_init, b_init, class_name)

            return q_values, obs

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval, self.state = build_network(self.customers, self.vehicles,
                                                    self.active_vehicle, self.target_customers,
                                                    self.available_customers, self.vehicles_pos, self.time,
                                                    self.instance_chars,
                                                    w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            # self.td_error = self.q_target - selected_q
            # self.loss = tf.reduce_mean(tf.square(self.td_error))
            huber = tf.keras.losses.Huber(delta=self.huber_p)
            self.loss = tf.reduce_mean(huber(self.q_target, selected_q))
            # self.loss = tf.reduce_sum(tf.math.log(tf.math.cosh(self.td_error)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
            self.gradient = self.optimizer.compute_gradients(self.loss, var_list=var_list)
            # self.gradinet = [(tf.clip_by_norm(grad, 5), var) for grad, var in self.gradinet]
            self.opt = self.optimizer.apply_gradients(self.gradient)
            # self.opt = self.optimizer.minimize(self.loss, var_list=var_list)

        # ------------------ build target_net ------------------

        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_network(self.customers, self.vehicles,
                                        self.active_vehicle, self.target_customers,
                                        self.available_customers, self.vehicles_pos, self.time,
                                        self.instance_chars,
                                        w_initializer, b_initializer, c_names)[0]

        # ------------------ replace network ------------------
        t_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def optimize(self, obs, target, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.available_customers: obs["available_customers"],
                     self.vehicles_pos: obs["vehicles_pos"],
                     self.time: obs["time"],
                     self.selected_target: obs["selected_target"],
                     self.instance_chars: obs["instance_chars"],
                     self.q_target: target
                     }
        opt, loss, gradient = sess.run([self.opt, self.loss, self.gradient], feed_dict=feed_dict)
        avg_gradient = np.mean([item for g in gradient for item in g[0].reshape(-1)])
        return loss, avg_gradient

    def value(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.vehicles_pos: obs["vehicles_pos"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_eval, feed_dict=feed_dict)
        # values, mm, ss = sess.run([self.q_eval, self.embeddings, self.state], feed_dict=feed_dict)

        # batch x n_target_customers
        return values

    def value_(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.vehicles_pos: obs["vehicles_pos"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_next, feed_dict=feed_dict)

        return values

    def set_opt_param(self, sess, new_lr=None, new_hp=None):
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_hp is not None:
            sess.run(tf.assign(self.huber_p, new_hp))

    def get_learning_params(self, sess):
        # learning rate and huber param
        return sess.run([self.lr, self.huber_p])

    @staticmethod
    def expand_dim(a):
        return np.expand_dims(a, axis=0)


# preemptive q value for each action
class DQNP2(object):
    """
    Diff with DQNP: vehicles embedding, no instance chars for VRPSCD. Q=1, all demands are normalized by Q
    Inputs: Customers, Vehicles, Previous_route, time, target_customers
    Output: Q value for a location
    The size of the target customers should be action_dim, if there are less customers, refer to customer n as a dummy.
    Operations:
    1- Embed customers and vehicles,
    2- Attention on customers to find their importance to each other
    3- Attention on vehicles to find their importance to customers
    4- Concat two attentions as h_i
    5- Encode the previous route by RNN (hidden state) as p_j
    6- Average h_i and p_j over n and m to find H and P
    7- make the context as C=[H, P, V', time] where V' is the embedding of the active vehicle
    8- Attention to find the the importance of customers for the context C.
    9- The output is Q values


    Notes:
        1- Each batch represents a vector of q values for all nodes in target_customers.
        2- customer set C has n+2 nodes, 0 to n-1 nodes are referring to customers, n is the depot and n+1 is a dummy
        node with values zero()
        3- The q value for dummy target customers is masked to zero.

        4- c -> x -> self-attention -> attention respect to vbar
    """

    def __init__(self, init_lr, emb_size, feat_size_c, feat_size_v, pos_hist=3, nb=15, use_path=True):
        self.emb_size = emb_size
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v
        self.nb = nb

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # in placeholders, the first axis is the batch size
        # batch x n_customers (height) x features_size (width)
        self.customers = tf.placeholder(tf.float32, shape=[None, None, feat_size_c], name="raw_c")

        # batch x n_vehicles x feature_size_vehicle
        self.vehicles = tf.placeholder(tf.float32, shape=[None, None, feat_size_v], name="raw_v")

        # batch x nb
        self.target_customers = tf.placeholder(tf.int32, shape=[None, nb], name="target_customers_ind")

        # batch x nb: real targets=1, dummy targets=0.
        self.available_targets = tf.placeholder(tf.float32, shape=[None, 2*nb], name="available_targets")

        # batch x n_customers: [1, 1, 0, 1, ...] => |C| + 1 + 1
        self.available_customers = tf.placeholder(tf.float32, shape=[None, None], name="available_customers")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        # batch x 2 (batch_id, vehicle_id)
        self.active_vehicle = tf.placeholder(tf.int32, [None, 2], name='active_vehicle_ind')

        # distance table: batch x n_customers x n_customers
        self.distance_table = tf.placeholder(tf.float32, [None, None, None], name='distance_table')

        # batch x n_vehicles x pos_hist (3) x position size (x, y)
        # self.vehicles_pos = tf.placeholder(tf.float32, [None, None, pos_hist, 2], name='vehicles_pos')
        self.vehicles_pos = tf.placeholder(tf.float32, [None, pos_hist, 2], name='vehicles_pos')

        self.time = tf.placeholder(tf.float32, [None, 1], name='time')

        # instance characteristics: DL, Q, SV
        self.instance_chars = tf.placeholder(tf.float32, [None, 5], name='instance_chars')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_customer_embedding(customers_set, available_customers, w_init, class_name,
                                     heads=4):
            # customers, node embedding
            wq = tf.get_variable('wq_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # Query: batch x n x e
            x_c = tf.matmul(customers_set, wq)

            # e'
            hsize = int(self.emb_size / heads)

            # Query: batch x n x e
            q = x_c

            # Key: batch x n x e
            k = tf.matmul(customers_set, wk)

            # Value: batch x n x e
            v = tf.matmul(customers_set, wv)

            # reshape to split
            batch_size, n_customers = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x n x e'
            q = tf.transpose(tf.reshape(q, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x n x e' * batch x h x e' x n: batch x h x n x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # involve the distance table
            # ww = tf.get_variable('wd1_att_cc', [1, heads], initializer=w_init, collections=class_name, dtype=tf.float32)
            # distance_table = tf.reshape(tf.matmul(tf.expand_dims(distance_table, axis=-1), ww),
            #                             [batch_size, heads, n_customers, n_customers, 1])
            # a = tf.concat([tf.expand_dims(a, axis=-1), distance_table], axis=-1)
            # wd = tf.get_variable('wd_att_cc', [2, 1], initializer=w_init, collections=class_name, dtype=tf.float32)
            # # batch x h x n x n
            # a = tf.reshape(tf.matmul(a, wd), [batch_size, heads, n_customers, n_customers])

            # eliminate the unavailable customers: batch x n x n ** batch x n

            a += tf.broadcast_to(tf.expand_dims(tf.expand_dims((1. - available_customers) * (-1e9), axis=1), axis=1),
                                 [batch_size, heads, n_customers, n_customers])

            # softmax: batch x h x n x n
            a = tf.nn.softmax(a, axis=-1)

            # graph embedding: batch x h x n x n * batch x h x n x e' -> batch x h x n x e' ->
            # batch x n x h x e' -> batch x n x e
            x_c_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, n_customers, self.emb_size])

            # Feed-forward:
            x_c_e += x_c

            return x_c_e

        def build_vehicle_embedding(vehicles_set, active_vehicle, w_init, class_name, heads=4):
            wq = tf.get_variable('wq_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # e'
            hsize = int(self.emb_size / heads)

            # embedding of the active vehicle: batch x e
            x_v_vbar = tf.matmul(tf.gather_nd(vehicles_set, active_vehicle, name="active_vehicle_emb"), wq)

            # Query: batch x 1 x e
            q = tf.expand_dims(x_v_vbar, 1)

            # Key: batch x n x e
            k = tf.matmul(vehicles_set, wk)

            # Value: batch x n x e
            v = tf.matmul(vehicles_set, wv)

            # reshape to split
            batch_size, n_vehicles = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x 1 x e'
            q = tf.transpose(tf.reshape(q, [batch_size, 1, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_vehicles, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_vehicles, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x 1 x e' * batch x h x e' x n: batch x h x 1 x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # softmax: batch x h x 1 x n
            a = tf.nn.softmax(a, axis=-1)

            # vehicles embedding: batch x h x 1 x n * batch x h x n x e' -> batch x h x 1 x e' ->
            # batch x 1 x h x e' -> batch x 1 x e -> batch x e
            h_v_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, self.emb_size])

            # Feed-forward:
            h_v_e += x_v_vbar
            return h_v_e

        def build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name):

            w1 = tf.get_variable('w1_att_g', [h_z_size + self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            w2 = tf.get_variable('w2_att_g', [self.emb_size, 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            batch_size, n_customers = tf.shape(h_c)[0], tf.shape(h_c)[1]
            # batch x e' -> batch x 1 x e' -> batch x n x e'
            context_extended = tf.broadcast_to(tf.expand_dims(h_z, axis=1),
                                               [batch_size, n_customers, h_z_size])

            # batch x n x (e + e')
            mu = tf.concat([context_extended, h_c], axis=-1)

            # batch x n
            weights = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(mu, w1)), w2), [batch_size, n_customers])

            # Masking
            weights += (1. - available_customers) * (-1e9)
            # batch x n: batch x 1 x n
            weights = tf.expand_dims(tf.nn.softmax(weights, axis=-1), axis=1)

            # batch x 1 x n * batch x n x e -> batch x 1 x e -> batch x e
            h_g = tf.reshape(tf.matmul(weights, h_c), [batch_size, self.emb_size])

            return h_g

        def build_fc_q_network(state, state_size, x_c, target_customers,
                               w_init, b_init, class_name):
            batch_size = tf.shape(state)[0]

            brange = tf.expand_dims(
                tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=1), [batch_size, self.nb]),
                axis=2)
            # batch x nb x 2
            indices = tf.concat([brange, tf.expand_dims(target_customers, axis=2)], axis=2)

            # gather the embedding of target customers for each batch: batch x nb x e
            x_t = tf.gather_nd(x_c, indices, name="gather_emb")

            # batch x nb x state_size
            bstate = tf.broadcast_to(tf.expand_dims(state, axis=1), [batch_size, self.nb, state_size])

            # batch x nb x (state_size + emb_size)
            state_actions = tf.concat([bstate, x_t], axis=2)
            state_size += self.emb_size
            h1 = int(state_size / 2.)

            w1 = tf.get_variable('w_fcd_1', [state_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name)

            # q_value = tf.nn.leaky_relu(tf.matmul(state_actions, w1) + b1)
            q_value = tf.nn.relu(tf.matmul(state_actions, w1) + b1)

            w2 = tf.get_variable('w_fcd_2', [h1, 2], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b2 = tf.get_variable('b_fcd_2', [1, 2], initializer=b_init, collections=class_name)

            # batch x nb x 1
            # q_value = tf.nn.leaky_relu(tf.matmul(q_value, w2) + b2)
            q_value = tf.matmul(q_value, w2) + b2

            # batch x nb*2
            q_value = tf.reshape(q_value, [batch_size, 2*self.nb])

            # mask out dummy targets
            # q_value = q_value * available_targets
            return q_value

        def build_network(customers_set, vehicles_set, active_vehicle, target_customers,
                          available_customers, time,
                          w_init, b_init, class_name):

            # embed the raw data: batch x n x e, batch x m x e
            h_c = build_customer_embedding(customers_set, available_customers,
                                           w_init, class_name, heads=4)
            h_v = build_vehicle_embedding(vehicles_set, active_vehicle, w_init, class_name, heads=4)

            h_z = tf.concat([h_v, time], axis=1)
            h_z_size = self.emb_size + 1

            h_g = build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name)
            obs = tf.concat([h_g, h_z], axis=1, name="concat_state")

            obs_size = h_z_size + self.emb_size

            # we use x_c_e as the target customer
            # batch x nb x 2
            q_values = build_fc_q_network(obs, obs_size, h_c, target_customers, w_init, b_init, class_name)

            return q_values, obs

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval, self.state = build_network(self.customers, self.vehicles,
                                                    self.active_vehicle, self.target_customers,
                                                    self.available_customers, self.time,
                                                    w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            # self.td_error = self.q_target - selected_q
            # self.loss = tf.reduce_mean(tf.square(self.td_error))
            huber = tf.keras.losses.Huber(delta=self.huber_p)
            self.loss = tf.reduce_mean(huber(self.q_target, selected_q))
            # self.loss = tf.reduce_sum(tf.math.log(tf.math.cosh(self.td_error)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
            # self.gradient = self.optimizer.compute_gradients(self.loss, var_list=var_list)
            # self.gradinet = [(tf.clip_by_norm(grad, 5), var) for grad, var in self.gradinet]
            # self.opt = self.optimizer.apply_gradients(self.gradient)
            self.opt = self.optimizer.minimize(self.loss, var_list=var_list)

        # ------------------ build target_net ------------------

        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_network(self.customers, self.vehicles,
                                        self.active_vehicle, self.target_customers,
                                        self.available_customers, self.time,
                                        w_initializer, b_initializer, c_names)[0]

        # ------------------ replace network ------------------
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def optimize(self, obs, target, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.available_customers: obs["available_customers"],
                     self.time: obs["time"],
                     self.selected_target: obs["selected_target"],
                     self.instance_chars: obs["instance_chars"],
                     self.q_target: target
                     }
        # opt, loss, gradient = sess.run([self.opt, self.loss, self.gradient], feed_dict=feed_dict)
        opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)
        # avg_gradient = np.mean([abs(item) for g in gradient for item in g[0].reshape(-1)])
        return loss, 0

    def value(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_eval, feed_dict=feed_dict)
        # values, mm, ss = sess.run([self.q_eval, self.embeddings, self.state], feed_dict=feed_dict)

        # batch x n_target_customers
        return values

    def value_(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_next, feed_dict=feed_dict)

        return values

    def set_opt_param(self, sess, new_lr=None, new_hp=None):
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_hp is not None:
            sess.run(tf.assign(self.huber_p, new_hp))

    def get_learning_params(self, sess):
        # learning rate and huber param
        return sess.run([self.lr, self.huber_p])

    @staticmethod
    def expand_dim(a):
        return np.expand_dims(a, axis=0)


# preemptive q value for each action - includes instance parameters
class DQNP3(object):
    """
    Diff with DQNP: vehicles embedding, no instance chars for VRPSCD. Q=1, all demands are normalized by Q
    Inputs: Customers, Vehicles, Previous_route, time, target_customers
    Output: Q value for a location
    The size of the target customers should be action_dim, if there are less customers, refer to customer n as a dummy.
    Operations:
    1- Embed customers and vehicles,
    2- Attention on customers to find their importance to each other
    3- Attention on vehicles to find their importance to customers
    4- Concat two attentions as h_i
    5- Encode the previous route by RNN (hidden state) as p_j
    6- Average h_i and p_j over n and m to find H and P
    7- make the context as C=[H, P, V', time] where V' is the embedding of the active vehicle
    8- Attention to find the the importance of customers for the context C.
    9- The output is Q values


    Notes:
        1- Each batch represents a vector of q values for all nodes in target_customers.
        2- customer set C has n+2 nodes, 0 to n-1 nodes are referring to customers, n is the depot and n+1 is a dummy
        node with values zero()
        3- The q value for dummy target customers is masked to zero.

        4- c -> x -> self-attention -> attention respect to vbar
    """

    def __init__(self, init_lr, emb_size, feat_size_c, feat_size_v, pos_hist=3, nb=15, use_path=True):
        self.emb_size = emb_size
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v
        self.nb = nb

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # in placeholders, the first axis is the batch size
        # batch x n_customers (height) x features_size (width)
        self.customers = tf.placeholder(tf.float32, shape=[None, None, feat_size_c], name="raw_c")

        # batch x n_vehicles x feature_size_vehicle
        self.vehicles = tf.placeholder(tf.float32, shape=[None, None, feat_size_v], name="raw_v")

        # batch x nb
        self.target_customers = tf.placeholder(tf.int32, shape=[None, nb], name="target_customers_ind")

        # batch x nb: real targets=1, dummy targets=0.
        self.available_targets = tf.placeholder(tf.float32, shape=[None, 2*nb], name="available_targets")

        # batch x n_customers: [1, 1, 0, 1, ...] => |C| + 1 + 1
        self.available_customers = tf.placeholder(tf.float32, shape=[None, None], name="available_customers")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        # batch x 2 (batch_id, vehicle_id)
        self.active_vehicle = tf.placeholder(tf.int32, [None, 2], name='active_vehicle_ind')

        # distance table: batch x n_customers x n_customers
        self.distance_table = tf.placeholder(tf.float32, [None, None, None], name='distance_table')

        # batch x n_vehicles x pos_hist (3) x position size (x, y)
        # self.vehicles_pos = tf.placeholder(tf.float32, [None, None, pos_hist, 2], name='vehicles_pos')
        self.vehicles_pos = tf.placeholder(tf.float32, [None, pos_hist, 2], name='vehicles_pos')

        self.time = tf.placeholder(tf.float32, [None, 1], name='time')

        # instance characteristics: DL, Q, SV
        self.instance_chars = tf.placeholder(tf.float32, [None, 6], name='instance_chars')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_customer_embedding(customers_set, available_customers, w_init, class_name,
                                     heads=4):
            # customers, node embedding
            wq = tf.get_variable('wq_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # Query: batch x n x e
            x_c = tf.matmul(customers_set, wq)

            # e'
            hsize = int(self.emb_size / heads)

            # Query: batch x n x e
            q = x_c

            # Key: batch x n x e
            k = tf.matmul(customers_set, wk)

            # Value: batch x n x e
            v = tf.matmul(customers_set, wv)

            # reshape to split
            batch_size, n_customers = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x n x e'
            q = tf.transpose(tf.reshape(q, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x n x e' * batch x h x e' x n: batch x h x n x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # involve the distance table
            # ww = tf.get_variable('wd1_att_cc', [1, heads], initializer=w_init, collections=class_name, dtype=tf.float32)
            # distance_table = tf.reshape(tf.matmul(tf.expand_dims(distance_table, axis=-1), ww),
            #                             [batch_size, heads, n_customers, n_customers, 1])
            # a = tf.concat([tf.expand_dims(a, axis=-1), distance_table], axis=-1)
            # wd = tf.get_variable('wd_att_cc', [2, 1], initializer=w_init, collections=class_name, dtype=tf.float32)
            # # batch x h x n x n
            # a = tf.reshape(tf.matmul(a, wd), [batch_size, heads, n_customers, n_customers])

            # eliminate the unavailable customers: batch x n x n ** batch x n

            a += tf.broadcast_to(tf.expand_dims(tf.expand_dims((1. - available_customers) * (-1e9), axis=1), axis=1),
                                 [batch_size, heads, n_customers, n_customers])

            # softmax: batch x h x n x n
            a = tf.nn.softmax(a, axis=-1)

            # graph embedding: batch x h x n x n * batch x h x n x e' -> batch x h x n x e' ->
            # batch x n x h x e' -> batch x n x e
            x_c_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, n_customers, self.emb_size])

            # Feed-forward:
            x_c_e += x_c

            return x_c_e

        def build_vehicle_embedding(vehicles_set, active_vehicle, w_init, class_name, heads=4):
            wq = tf.get_variable('wq_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # e'
            hsize = int(self.emb_size / heads)

            # embedding of the active vehicle: batch x e
            x_v_vbar = tf.matmul(tf.gather_nd(vehicles_set, active_vehicle, name="active_vehicle_emb"), wq)

            # Query: batch x 1 x e
            q = tf.expand_dims(x_v_vbar, 1)

            # Key: batch x n x e
            k = tf.matmul(vehicles_set, wk)

            # Value: batch x n x e
            v = tf.matmul(vehicles_set, wv)

            # reshape to split
            batch_size, n_vehicles = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x 1 x e'
            q = tf.transpose(tf.reshape(q, [batch_size, 1, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_vehicles, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_vehicles, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x 1 x e' * batch x h x e' x n: batch x h x 1 x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # softmax: batch x h x 1 x n
            a = tf.nn.softmax(a, axis=-1)

            # vehicles embedding: batch x h x 1 x n * batch x h x n x e' -> batch x h x 1 x e' ->
            # batch x 1 x h x e' -> batch x 1 x e -> batch x e
            h_v_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, self.emb_size])

            # Feed-forward:
            h_v_e += x_v_vbar
            return h_v_e

        def build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name):

            w1 = tf.get_variable('w1_att_g', [h_z_size + self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            w2 = tf.get_variable('w2_att_g', [self.emb_size, 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            batch_size, n_customers = tf.shape(h_c)[0], tf.shape(h_c)[1]
            # batch x e' -> batch x 1 x e' -> batch x n x e'
            context_extended = tf.broadcast_to(tf.expand_dims(h_z, axis=1),
                                               [batch_size, n_customers, h_z_size])

            # batch x n x (e + e')
            mu = tf.concat([context_extended, h_c], axis=-1)

            # batch x n
            weights = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(mu, w1)), w2), [batch_size, n_customers])

            # Masking
            weights += (1. - available_customers) * (-1e9)
            # batch x n: batch x 1 x n
            weights = tf.expand_dims(tf.nn.softmax(weights, axis=-1), axis=1)

            # batch x 1 x n * batch x n x e -> batch x 1 x e -> batch x e
            h_g = tf.reshape(tf.matmul(weights, h_c), [batch_size, self.emb_size])

            return h_g

        def build_fc_q_network(state, state_size, x_c, target_customers,
                               w_init, b_init, class_name):
            batch_size = tf.shape(state)[0]

            brange = tf.expand_dims(
                tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=1), [batch_size, self.nb]),
                axis=2)
            # batch x nb x 2
            indices = tf.concat([brange, tf.expand_dims(target_customers, axis=2)], axis=2)

            # gather the embedding of target customers for each batch: batch x nb x e
            x_t = tf.gather_nd(x_c, indices, name="gather_emb")

            # batch x nb x state_size
            bstate = tf.broadcast_to(tf.expand_dims(state, axis=1), [batch_size, self.nb, state_size])

            # batch x nb x (state_size + emb_size)
            state_actions = tf.concat([bstate, x_t], axis=2)
            state_size += self.emb_size
            h1 = int(state_size / 2.)

            w1 = tf.get_variable('w_fcd_1', [state_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name)

            w1tl = tf.get_variable('w_fcd_1_tl', [state_size, h1], initializer=w_init,
                                   collections=class_name, dtype=tf.float32)
            b1tl = tf.get_variable('b_fcd_1_tl', [1, h1], initializer=b_init, collections=class_name)

            # q_value = tf.nn.leaky_relu(tf.matmul(state_actions, w1) + b1)
            q_value = tf.nn.relu(tf.matmul(state_actions, w1) + b1 + tf.matmul(state_actions, w1tl) + b1tl)

            w2 = tf.get_variable('w_fcd_2', [h1, 2], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b2 = tf.get_variable('b_fcd_2', [1, 2], initializer=b_init, collections=class_name)

            w2tl = tf.get_variable('w_fcd_2_tl', [h1, 2], initializer=w_init,
                                   collections=class_name, dtype=tf.float32)
            b2tl = tf.get_variable('b_fcd_2_tl', [1, 2], initializer=b_init, collections=class_name)

            # batch x nb x 1
            q_value = tf.matmul(q_value, w2) + b2 + tf.matmul(q_value, w2tl) + b2tl

            # batch x nb*2
            q_value = tf.reshape(q_value, [batch_size, 2 * self.nb])

            # mask out dummy targets
            # q_value = q_value * available_targets
            return q_value

        def build_network(customers_set, vehicles_set, active_vehicle, target_customers,
                          available_customers, time, instance_chars,
                          w_init, b_init, class_name):

            # embed the raw data: batch x n x e, batch x m x e
            h_c = build_customer_embedding(customers_set, available_customers,
                                           w_init, class_name, heads=4)
            h_v = build_vehicle_embedding(vehicles_set, active_vehicle, w_init, class_name, heads=4)

            h_z = tf.concat([h_v, time], axis=1)
            h_z_size = self.emb_size + 1

            h_g = build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name)
            obs = tf.concat([h_g, h_z, instance_chars], axis=1, name="concat_state")

            obs_size = h_z_size + self.emb_size + 6

            # we use x_c_e as the target customer
            # batch x nb x 2
            q_values = build_fc_q_network(obs, obs_size, h_c, target_customers, w_init, b_init, class_name)

            return q_values, obs

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval, self.state = build_network(self.customers, self.vehicles,
                                                    self.active_vehicle, self.target_customers,
                                                    self.available_customers, self.time, self.instance_chars,
                                                    w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            # self.td_error = self.q_target - selected_q
            # self.loss = tf.reduce_mean(tf.square(self.td_error))
            huber = tf.keras.losses.Huber(delta=self.huber_p)
            self.loss = tf.reduce_mean(huber(self.q_target, selected_q))
            # self.loss = tf.reduce_sum(tf.math.log(tf.math.cosh(self.td_error)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
            # self.gradient = self.optimizer.compute_gradients(self.loss, var_list=var_list)
            # self.gradinet = [(tf.clip_by_norm(grad, 5), var) for grad, var in self.gradinet]
            # self.opt = self.optimizer.apply_gradients(self.gradient)
            self.opt = self.optimizer.minimize(self.loss, var_list=var_list)

        # ------------------ build target_net ------------------

        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_network(self.customers, self.vehicles,
                                        self.active_vehicle, self.target_customers,
                                        self.available_customers, self.time, self.instance_chars,
                                        w_initializer, b_initializer, c_names)[0]

        # ------------------ replace network ------------------
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def optimize(self, obs, target, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.available_customers: obs["available_customers"],
                     self.time: obs["time"],
                     self.selected_target: obs["selected_target"],
                     self.instance_chars: obs["instance_chars"],
                     self.q_target: target
                     }
        # opt, loss, gradient = sess.run([self.opt, self.loss, self.gradient], feed_dict=feed_dict)
        opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)
        # avg_gradient = np.mean([abs(item) for g in gradient for item in g[0].reshape(-1)])
        return loss, 0

    def value(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_eval, feed_dict=feed_dict)
        # values, mm, ss = sess.run([self.q_eval, self.embeddings, self.state], feed_dict=feed_dict)

        # batch x n_target_customers
        return values

    def value_(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_next, feed_dict=feed_dict)

        return values

    def set_opt_param(self, sess, new_lr=None, new_hp=None):
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_hp is not None:
            sess.run(tf.assign(self.huber_p, new_hp))

    def get_learning_params(self, sess):
        # learning rate and huber param
        return sess.run([self.lr, self.huber_p])

    @staticmethod
    def expand_dim(a):
        return np.expand_dims(a, axis=0)


# DQNP3 + attention for customers on vehicles
class DQNP4(object):
    """
    Diff with DQNP: vehicles embedding, no instance chars for VRPSCD. Q=1, all demands are normalized by Q
    Inputs: Customers, Vehicles, Previous_route, time, target_customers
    Output: Q value for a location
    The size of the target customers should be action_dim, if there are less customers, refer to customer n as a dummy.
    Operations:
    1- Embed customers and vehicles,
    2- Attention on customers to find their importance to each other
    3- Attention on vehicles to find their importance to customers
    4- Concat two attentions as h_i
    5- Encode the previous route by RNN (hidden state) as p_j
    6- Average h_i and p_j over n and m to find H and P
    7- make the context as C=[H, P, V', time] where V' is the embedding of the active vehicle
    8- Attention to find the the importance of customers for the context C.
    9- The output is Q values


    Notes:
        1- Each batch represents a vector of q values for all nodes in target_customers.
        2- customer set C has n+2 nodes, 0 to n-1 nodes are referring to customers, n is the depot and n+1 is a dummy
        node with values zero()
        3- The q value for dummy target customers is masked to zero.

        4- c -> x -> self-attention -> attention respect to vbar
    """

    def __init__(self, init_lr, emb_size, feat_size_c, feat_size_v, pos_hist=3, nb=15, use_path=True):
        self.emb_size = emb_size
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v
        self.nb = nb

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # in placeholders, the first axis is the batch size
        # batch x n_customers (height) x features_size (width)
        self.customers = tf.placeholder(tf.float32, shape=[None, None, feat_size_c], name="raw_c")

        # batch x n_vehicles x feature_size_vehicle
        self.vehicles = tf.placeholder(tf.float32, shape=[None, None, feat_size_v], name="raw_v")

        # batch x nb
        self.target_customers = tf.placeholder(tf.int32, shape=[None, nb], name="target_customers_ind")

        # batch x nb: real targets=1, dummy targets=0.
        # self.available_targets = tf.placeholder(tf.float32, shape=[None, 2*nb], name="available_targets")

        # batch x n_customers: [1, 1, 0, 1, ...] => |C| + 1 + 1
        self.available_customers = tf.placeholder(tf.float32, shape=[None, None], name="available_customers")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        # batch x 2 (batch_id, vehicle_id)
        self.active_vehicle = tf.placeholder(tf.int32, [None, 2], name='active_vehicle_ind')

        # distance table: batch x n_customers x n_customers
        self.distance_table = tf.placeholder(tf.float32, [None, None, None], name='distance_table')

        # batch x n_vehicles x pos_hist (3) x position size (x, y)
        # self.vehicles_pos = tf.placeholder(tf.float32, [None, None, pos_hist, 2], name='vehicles_pos')
        self.vehicles_pos = tf.placeholder(tf.float32, [None, pos_hist, 2], name='vehicles_pos')

        self.time = tf.placeholder(tf.float32, [None, 1], name='time')

        # instance characteristics: DL, Q, SV
        self.instance_chars = tf.placeholder(tf.float32, [None, 6], name='instance_chars')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_customer_embedding(customers_set, available_customers, vehicles_set, w_init, class_name,
                                     heads=4):
            # customers, node embedding
            wq = tf.get_variable('wq_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # Query: batch x n x e
            x_c = tf.matmul(customers_set, wq)

            # e'
            hsize = int(self.emb_size / heads)

            # Query: batch x n x e
            q = x_c

            # Key: batch x n x e
            k = tf.matmul(customers_set, wk)

            # Value: batch x n x e
            v = tf.matmul(customers_set, wv)

            # reshape to split
            batch_size, n_customers = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x n x e'
            q = tf.transpose(tf.reshape(q, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x n x e' * batch x h x e' x n: batch x h x n x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            a += tf.broadcast_to(tf.expand_dims(tf.expand_dims((1. - available_customers) * (-1e9), axis=1), axis=1),
                                 [batch_size, heads, n_customers, n_customers])

            # softmax: batch x h x n x n
            a = tf.nn.softmax(a, axis=-1)

            # graph embedding: batch x h x n x n * batch x h x n x e' -> batch x h x n x e' ->
            # batch x n x h x e' -> batch x n x e
            x_c_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, n_customers, self.emb_size])

            # Feed-forward:
            x_c_e += x_c

            # attention of customers on vehicles
            wqv = tf.get_variable('wq_att_cv', [self.feat_size_c, self.emb_size], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)
            wkv = tf.get_variable('wk_att_cv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)
            wvv = tf.get_variable('wv_att_cv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)
            # Query: batch x n x e
            x_cv = tf.matmul(customers_set, wqv)

            # Key: batch x m x e
            kv = tf.matmul(vehicles_set, wkv)

            # Value: batch x m x e
            vv = tf.matmul(vehicles_set, wvv)

            # batch x n x m
            av = tf.matmul(x_cv, tf.transpose(kv, [0, 2, 1])) / tf.math.sqrt(self.emb_size * 1.)
            # softmax: batch x n x m
            av = tf.nn.softmax(av, axis=-1)

            # batch x n x e
            x_c_v = tf.matmul(av, vv)

            # batch x n x 2e
            x_c_e_f = tf.concat([x_c_e, x_c_v], axis=2)
            return x_c_e_f

        def build_vehicle_embedding(vehicles_set, active_vehicle, w_init, class_name, heads=4):
            wq = tf.get_variable('wq_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # e'
            hsize = int(self.emb_size / heads)

            # embedding of the active vehicle: batch x e
            x_v_vbar = tf.matmul(tf.gather_nd(vehicles_set, active_vehicle, name="active_vehicle_emb"), wq)

            # Query: batch x 1 x e
            q = tf.expand_dims(x_v_vbar, 1)

            # Key: batch x n x e
            k = tf.matmul(vehicles_set, wk)

            # Value: batch x n x e
            v = tf.matmul(vehicles_set, wv)

            # reshape to split
            batch_size, n_vehicles = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x 1 x e'
            q = tf.transpose(tf.reshape(q, [batch_size, 1, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_vehicles, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_vehicles, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x 1 x e' * batch x h x e' x n: batch x h x 1 x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # softmax: batch x h x 1 x n
            a = tf.nn.softmax(a, axis=-1)

            # vehicles embedding: batch x h x 1 x n * batch x h x n x e' -> batch x h x 1 x e' ->
            # batch x 1 x h x e' -> batch x 1 x e -> batch x e
            h_v_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, self.emb_size])

            # Feed-forward:
            h_v_e += x_v_vbar
            return h_v_e

        def build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name):

            w1 = tf.get_variable('w1_att_g', [h_z_size + 2*self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            w2 = tf.get_variable('w2_att_g', [self.emb_size, 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            batch_size, n_customers = tf.shape(h_c)[0], tf.shape(h_c)[1]
            # batch x e' -> batch x 1 x e' -> batch x n x e'
            context_extended = tf.broadcast_to(tf.expand_dims(h_z, axis=1),
                                               [batch_size, n_customers, h_z_size])

            # batch x n x (e + e')
            mu = tf.concat([context_extended, h_c], axis=-1)

            # batch x n
            weights = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(mu, w1)), w2), [batch_size, n_customers])

            # Masking
            weights += (1. - available_customers) * (-1e9)
            # batch x n: batch x 1 x n
            weights = tf.expand_dims(tf.nn.softmax(weights, axis=-1), axis=1)

            # batch x 1 x n * batch x n x e -> batch x 1 x e -> batch x e
            h_g = tf.reshape(tf.matmul(weights, h_c), [batch_size, 2 * self.emb_size])

            return h_g

        def build_fc_q_network(state, state_size, x_c, target_customers,
                               w_init, b_init, class_name):
            batch_size = tf.shape(state)[0]

            brange = tf.expand_dims(
                tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=1), [batch_size, self.nb]),
                axis=2)
            # batch x nb x 2
            indices = tf.concat([brange, tf.expand_dims(target_customers, axis=2)], axis=2)

            # gather the embedding of target customers for each batch: batch x nb x e
            x_t = tf.gather_nd(x_c, indices, name="gather_emb")

            # batch x nb x state_size
            bstate = tf.broadcast_to(tf.expand_dims(state, axis=1), [batch_size, self.nb, state_size])

            # batch x nb x (state_size + emb_size)
            state_actions = tf.concat([bstate, x_t], axis=2)
            state_size += 2 * self.emb_size
            h1 = int(state_size / 2.)

            w1 = tf.get_variable('w_fcd_1', [state_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name)

            # w1tl = tf.get_variable('w_fcd_1_tl', [state_size, h1], initializer=w_init,
            #                        collections=class_name, dtype=tf.float32)
            # b1tl = tf.get_variable('b_fcd_1_tl', [1, h1], initializer=b_init, collections=class_name)

            q_value = tf.nn.leaky_relu(tf.matmul(state_actions, w1) + b1)
            # q_value = tf.nn.relu(tf.matmul(state_actions, w1) + b1 + tf.matmul(state_actions, w1tl) + b1tl)

            w2 = tf.get_variable('w_fcd_2', [h1, 2], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b2 = tf.get_variable('b_fcd_2', [1, 2], initializer=b_init, collections=class_name)

            # w2tl = tf.get_variable('w_fcd_2_tl', [h1, 2], initializer=w_init,
            #                        collections=class_name, dtype=tf.float32)
            # b2tl = tf.get_variable('b_fcd_2_tl', [1, 2], initializer=b_init, collections=class_name)

            # batch x nb x 1
            q_value = tf.matmul(q_value, w2) + b2
            # q_value = tf.matmul(q_value, w2) + b2 + tf.matmul(q_value, w2tl) + b2tl

            # batch x nb*2
            q_value = tf.reshape(q_value, [batch_size, 2 * self.nb])

            # mask out dummy targets
            # q_value = q_value * available_targets
            return q_value

        def build_network(customers_set, vehicles_set, active_vehicle, target_customers,
                          available_customers, time, instance_chars,
                          w_init, b_init, class_name):

            # embed the raw data: batch x n x e, batch x m x e
            h_c = build_customer_embedding(customers_set, available_customers, vehicles_set,
                                           w_init, class_name, heads=4)
            h_v = build_vehicle_embedding(vehicles_set, active_vehicle, w_init, class_name, heads=4)

            h_z = tf.concat([h_v, time], axis=1)
            h_z_size = self.emb_size + 1

            h_g = build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name)
            obs = tf.concat([h_g, h_z, instance_chars], axis=1, name="concat_state")

            obs_size = h_z_size + 2 * self.emb_size + 6

            # we use x_c_e as the target customer
            # batch x nb x 2
            q_values = build_fc_q_network(obs, obs_size, h_c, target_customers, w_init, b_init, class_name)

            return q_values, obs

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval, self.state = build_network(self.customers, self.vehicles,
                                                    self.active_vehicle, self.target_customers,
                                                    self.available_customers, self.time, self.instance_chars,
                                                    w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            self.td_error = self.q_target - selected_q
            self.loss = tf.reduce_mean(tf.square(self.td_error))
            # huber = tf.keras.losses.Huber(delta=self.huber_p)
            # self.loss = tf.reduce_mean(huber(self.q_target, selected_q))
            # self.loss = tf.reduce_sum(tf.math.log(tf.math.cosh(self.td_error)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
            # self.gradient = self.optimizer.compute_gradients(self.loss, var_list=var_list)
            # self.gradinet = [(tf.clip_by_norm(grad, 5), var) for grad, var in self.gradinet]
            # self.opt = self.optimizer.apply_gradients(self.gradient)
            self.opt = self.optimizer.minimize(self.loss, var_list=var_list)

        # ------------------ build target_net ------------------

        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_network(self.customers, self.vehicles,
                                        self.active_vehicle, self.target_customers,
                                        self.available_customers, self.time, self.instance_chars,
                                        w_initializer, b_initializer, c_names)[0]

        # ------------------ replace network ------------------
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def optimize(self, obs, target, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.available_customers: obs["available_customers"],
                     self.time: obs["time"],
                     self.selected_target: obs["selected_target"],
                     self.instance_chars: obs["instance_chars"],
                     self.q_target: target
                     }
        # opt, loss, gradient = sess.run([self.opt, self.loss, self.gradient], feed_dict=feed_dict)
        opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)
        # avg_gradient = np.mean([abs(item) for g in gradient for item in g[0].reshape(-1)])
        return loss, 0

    def value(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_eval, feed_dict=feed_dict)
        # values, mm, ss = sess.run([self.q_eval, self.embeddings, self.state], feed_dict=feed_dict)

        # batch x n_target_customers
        return values

    def value_(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_next, feed_dict=feed_dict)

        return values

    def set_opt_param(self, sess, new_lr=None, new_hp=None):
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_hp is not None:
            sess.run(tf.assign(self.huber_p, new_hp))

    def get_learning_params(self, sess):
        # learning rate and huber param
        return sess.run([self.lr, self.huber_p])

    @staticmethod
    def expand_dim(a):
        return np.expand_dims(a, axis=0)


class DQNBono(object):
    """
    Diff with DQNP: vehicles embedding, no instance chars for VRPSCD. Q=1, all demands are normalized by Q
    Inputs: Customers, Vehicles, Previous_route, time, target_customers
    Output: Q value for a location
    The size of the target customers should be action_dim, if there are less customers, refer to customer n as a dummy.
    Operations:
    1- Embed customers and vehicles,
    2- Attention on customers to find their importance to each other
    3- Attention on vehicles to find their importance to customers
    4- Concat two attentions as h_i
    5- Encode the previous route by RNN (hidden state) as p_j
    6- Average h_i and p_j over n and m to find H and P
    7- make the context as C=[H, P, V', time] where V' is the embedding of the active vehicle
    8- Attention to find the the importance of customers for the context C.
    9- The output is Q values


    Notes:
        1- Each batch represents a vector of q values for all nodes in target_customers.
        2- customer set C has n+2 nodes, 0 to n-1 nodes are referring to customers, n is the depot and n+1 is a dummy
        node with values zero()
        3- The q value for dummy target customers is masked to zero.

        4- c -> x -> self-attention -> attention respect to vbar
    """

    def __init__(self, init_lr, emb_size, feat_size_c, feat_size_v, pos_hist=3, nb=15, use_path=True):
        self.emb_size = emb_size
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v
        self.nb = nb

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # in placeholders, the first axis is the batch size
        # batch x n_customers (height) x features_size (width)
        self.customers = tf.placeholder(tf.float32, shape=[None, None, feat_size_c], name="raw_c")

        # batch x n_vehicles x feature_size_vehicle
        self.vehicles = tf.placeholder(tf.float32, shape=[None, None, feat_size_v], name="raw_v")

        # batch x nb
        self.target_customers = tf.placeholder(tf.int32, shape=[None, nb], name="target_customers_ind")

        # batch x nb: real targets=1, dummy targets=0.
        self.available_targets = tf.placeholder(tf.float32, shape=[None, 2*nb], name="available_targets")

        # batch x n_customers: [1, 1, 0, 1, ...] => |C| + 1 + 1
        self.available_customers = tf.placeholder(tf.float32, shape=[None, None], name="available_customers")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        # batch x 2 (batch_id, vehicle_id)
        self.active_vehicle = tf.placeholder(tf.int32, [None, 2], name='active_vehicle_ind')

        # distance table: batch x n_customers x n_customers
        self.distance_table = tf.placeholder(tf.float32, [None, None, None], name='distance_table')

        # batch x n_vehicles x pos_hist (3) x position size (x, y)
        # self.vehicles_pos = tf.placeholder(tf.float32, [None, None, pos_hist, 2], name='vehicles_pos')
        self.vehicles_pos = tf.placeholder(tf.float32, [None, pos_hist, 2], name='vehicles_pos')

        self.time = tf.placeholder(tf.float32, [None, 1], name='time')

        # instance characteristics: DL, Q, SV
        self.instance_chars = tf.placeholder(tf.float32, [None, 5], name='instance_chars')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_customer_embedding(customers_set, available_customers, w_init, class_name,
                                     heads=4):
            # customers, node embedding
            wq = tf.get_variable('wq_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # Query: batch x n x e
            x_c = tf.matmul(customers_set, wq)

            # e'
            hsize = int(self.emb_size / heads)

            # Query: batch x n x e
            q = x_c

            # Key: batch x n x e
            k = tf.matmul(customers_set, wk)

            # Value: batch x n x e
            v = tf.matmul(customers_set, wv)

            # reshape to split
            batch_size, n_customers = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x n x e'
            q = tf.transpose(tf.reshape(q, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x n x e' * batch x h x e' x n: batch x h x n x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # involve the distance table
            # ww = tf.get_variable('wd1_att_cc', [1, heads], initializer=w_init, collections=class_name, dtype=tf.float32)
            # distance_table = tf.reshape(tf.matmul(tf.expand_dims(distance_table, axis=-1), ww),
            #                             [batch_size, heads, n_customers, n_customers, 1])
            # a = tf.concat([tf.expand_dims(a, axis=-1), distance_table], axis=-1)
            # wd = tf.get_variable('wd_att_cc', [2, 1], initializer=w_init, collections=class_name, dtype=tf.float32)
            # # batch x h x n x n
            # a = tf.reshape(tf.matmul(a, wd), [batch_size, heads, n_customers, n_customers])

            # eliminate the unavailable customers: batch x n x n ** batch x n

            a += tf.broadcast_to(tf.expand_dims(tf.expand_dims((1. - available_customers) * (-1e9), axis=1), axis=1),
                                 [batch_size, heads, n_customers, n_customers])

            # softmax: batch x h x n x n
            a = tf.nn.softmax(a, axis=-1)

            # graph embedding: batch x h x n x n * batch x h x n x e' -> batch x h x n x e' ->
            # batch x n x h x e' -> batch x n x e
            x_c_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, n_customers, self.emb_size])

            # Feed-forward:
            x_c_e += x_c

            return x_c_e

        def build_vehicle_embedding(vehicles_set, active_vehicle, customers_embedding, w_init, class_name, heads=4):
            wq = tf.get_variable('wq_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wk = tf.get_variable('wk_att_vv', [self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            wv = tf.get_variable('wv_att_vv', [self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            # b x m x e
            hv = tf.matmul(vehicles_set, wq, name="vm1")

            # b x n x e
            hc1 = tf.matmul(customers_embedding, wk, name="vm2")
            hc2 = tf.matmul(customers_embedding, wv, name="vm3")

            # v: b x m x e * b x e x n -> b x m x n
            a = tf.matmul(hv, tf.transpose(hc1, [0, 2, 1])) / tf.math.sqrt(self.emb_size * 1.)
            # softmax: batch x m x n
            a = tf.nn.softmax(a, axis=-1)

            # b x m x n * b x n x e -> b x m x e
            vembs = tf.matmul(a, hc2, name="vm4")

            # embedding of the active vehicle: batch x e
            x_v_vbar = tf.expand_dims(tf.gather_nd(vembs, active_vehicle, name="active_vehicle_emb"), dim=1)

            wk2 = tf.get_variable('wk_att_vv2', [self.emb_size, self.emb_size], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)
            wv2 = tf.get_variable('wv_att_vv2', [self.emb_size, self.emb_size], initializer=w_init,
                                  collections=class_name, dtype=tf.float32)

            vembs1 = tf.matmul(vembs, wk2, name="vm5")
            vembs2 = tf.matmul(vembs, wv2, name="vm6")

            # b x 1 x e * b x e x m -> b x 1 x m
            a2 = tf.matmul(x_v_vbar, tf.transpose(vembs1, [0, 2, 1]), name="vm7")
            # softmax: batch x 1 x m
            a2 = tf.nn.softmax(a2, axis=-1)

            # b x 1 x m * b x m x e -> b x 1 x e
            vbaremb = tf.reshape(tf.matmul(a2, vembs2, name="vm8"), [-1, self.emb_size])

            return vbaremb

        def build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name):

            w1 = tf.get_variable('w1_att_g', [h_z_size + self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            w2 = tf.get_variable('w2_att_g', [self.emb_size, 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)

            batch_size, n_customers = tf.shape(h_c)[0], tf.shape(h_c)[1]
            # batch x e' -> batch x 1 x e' -> batch x n x e'
            context_extended = tf.broadcast_to(tf.expand_dims(h_z, axis=1),
                                               [batch_size, n_customers, h_z_size])

            # batch x n x (e + e')
            mu = tf.concat([context_extended, h_c], axis=-1)

            # batch x n
            weights = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(mu, w1)), w2), [batch_size, n_customers])

            # Masking
            weights += (1. - available_customers) * (-1e9)
            # batch x n: batch x 1 x n
            weights = tf.expand_dims(tf.nn.softmax(weights, axis=-1), axis=1)

            # batch x 1 x n * batch x n x e -> batch x 1 x e -> batch x e
            h_g = tf.reshape(tf.matmul(weights, h_c), [batch_size, self.emb_size])

            return h_g

        def build_fc_q_network(state, state_size, x_c, target_customers,
                               w_init, b_init, class_name):
            batch_size = tf.shape(state)[0]

            brange = tf.expand_dims(
                tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=1), [batch_size, self.nb]),
                axis=2)
            # batch x nb x 2
            indices = tf.concat([brange, tf.expand_dims(target_customers, axis=2)], axis=2)

            # gather the embedding of target customers for each batch: batch x nb x e
            x_t = tf.gather_nd(x_c, indices, name="gather_emb")

            # batch x nb x state_size
            bstate = tf.broadcast_to(tf.expand_dims(state, axis=1), [batch_size, self.nb, state_size])

            # batch x nb x (state_size + emb_size)
            state_actions = tf.concat([bstate, x_t], axis=2)
            state_size += self.emb_size
            h1 = int(state_size / 2.)

            w1 = tf.get_variable('w_fcd_1', [state_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name)

            # q_value = tf.nn.leaky_relu(tf.matmul(state_actions, w1) + b1)
            q_value = tf.nn.relu(tf.matmul(state_actions, w1) + b1)

            w2 = tf.get_variable('w_fcd_2', [h1, 2], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b2 = tf.get_variable('b_fcd_2', [1, 2], initializer=b_init, collections=class_name)

            # batch x nb x 1
            # q_value = tf.nn.leaky_relu(tf.matmul(q_value, w2) + b2)
            q_value = tf.matmul(q_value, w2) + b2

            # batch x nb*2
            q_value = tf.reshape(q_value, [batch_size, 2*self.nb])

            # mask out dummy targets
            # q_value = q_value * available_targets
            return q_value

        def build_network(customers_set, vehicles_set, active_vehicle, target_customers,
                          available_customers, time,
                          w_init, b_init, class_name):

            # embed the raw data: batch x n x e, batch x m x e
            h_c = build_customer_embedding(customers_set, available_customers,
                                           w_init, class_name, heads=4)
            h_v = build_vehicle_embedding(vehicles_set, active_vehicle, h_c, w_init, class_name, heads=4)

            obs = tf.concat([h_v, time], axis=1)
            obs_size = self.emb_size + 1
            #
            # h_g = build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name)
            # obs = tf.concat([h_g, h_z], axis=1, name="concat_state")

            # obs_size = h_z_size + self.emb_size

            # we use x_c_e as the target customer
            # batch x nb x 2
            q_values = build_fc_q_network(obs, obs_size, h_c, target_customers, w_init, b_init, class_name)

            return q_values, obs

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval, self.state = build_network(self.customers, self.vehicles,
                                                    self.active_vehicle, self.target_customers,
                                                    self.available_customers, self.time,
                                                    w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            # self.td_error = self.q_target - selected_q
            # self.loss = tf.reduce_mean(tf.square(self.td_error))
            huber = tf.keras.losses.Huber(delta=self.huber_p)
            self.loss = tf.reduce_mean(huber(self.q_target, selected_q))
            # self.loss = tf.reduce_sum(tf.math.log(tf.math.cosh(self.td_error)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
            # self.gradient = self.optimizer.compute_gradients(self.loss, var_list=var_list)
            # self.gradinet = [(tf.clip_by_norm(grad, 5), var) for grad, var in self.gradinet]
            # self.opt = self.optimizer.apply_gradients(self.gradient)
            self.opt = self.optimizer.minimize(self.loss, var_list=var_list)

        # ------------------ build target_net ------------------

        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_network(self.customers, self.vehicles,
                                        self.active_vehicle, self.target_customers,
                                        self.available_customers, self.time,
                                        w_initializer, b_initializer, c_names)[0]

        # ------------------ replace network ------------------
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def optimize(self, obs, target, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.available_customers: obs["available_customers"],
                     self.time: obs["time"],
                     self.selected_target: obs["selected_target"],
                     self.instance_chars: obs["instance_chars"],
                     self.q_target: target
                     }
        # opt, loss, gradient = sess.run([self.opt, self.loss, self.gradient], feed_dict=feed_dict)
        opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)
        # avg_gradient = np.mean([abs(item) for g in gradient for item in g[0].reshape(-1)])
        return loss, 0

    def value(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_eval, feed_dict=feed_dict)
        # values, mm, ss = sess.run([self.q_eval, self.embeddings, self.state], feed_dict=feed_dict)

        # batch x n_target_customers
        return values

    def value_(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_next, feed_dict=feed_dict)

        return values

    def set_opt_param(self, sess, new_lr=None, new_hp=None):
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_hp is not None:
            sess.run(tf.assign(self.huber_p, new_hp))

    def get_learning_params(self, sess):
        # learning rate and huber param
        return sess.run([self.lr, self.huber_p])

    @staticmethod
    def expand_dim(a):
        return np.expand_dims(a, axis=0)


# transfer learning
class DQNPTransferred(object):
    """

    Inputs: Customers, Vehicles, Previous_route, time, target_customers
    Output: Q value for a location
    The size of the target customers should be action_dim, if there are less customers, refer to customer n as a dummy.
    Operations:
    1- Embed customers and vehicles,
    2- Attention on customers to find their importance to each other
    3- Attention on vehicles to find their importance to customers
    4- Concat two attentions as h_i
    5- Encode the previous route by RNN (hidden state) as p_j
    6- Average h_i and p_j over n and m to find H and P
    7- make the context as C=[H, P, V', time] where V' is the embedding of the active vehicle
    8- Attention to find the the importance of customers for the context C.
    9- The output is Q values


    Notes:
        1- Each batch represents a vector of q values for all nodes in target_customers.
        2- customer set C has n+2 nodes, 0 to n-1 nodes are referring to customers, n is the depot and n+1 is a dummy
        node with values zero()
        3- The q value for dummy target customers is masked to zero.

        4- c -> x -> self-attention -> attention respect to vbar
    """

    def __init__(self, init_lr, emb_size, feat_size_c, feat_size_v, pos_hist=3, nb=15, fine_tune=False):
        self.emb_size = emb_size
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v
        self.nb = nb

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # in placeholders, the first axis is the batch size
        # batch x n_customers (height) x features_size (width)
        self.customers = tf.placeholder(tf.float32, shape=[None, None, feat_size_c], name="raw_c")

        # batch x n_vehicles x feature_size_vehicle
        self.vehicles = tf.placeholder(tf.float32, shape=[None, None, feat_size_v], name="raw_v")

        # batch x nb
        self.target_customers = tf.placeholder(tf.int32, shape=[None, nb], name="target_customers_ind")

        # batch x nb: real targets=1, dummy targets=0.
        self.available_targets = tf.placeholder(tf.float32, shape=[None, 2*nb], name="available_targets")

        # batch x n_customers: [1, 1, 0, 1, ...] => |C| + 1 + 1
        self.available_customers = tf.placeholder(tf.float32, shape=[None, None], name="available_customers")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        # batch x 2 (batch_id, vehicle_id)
        self.active_vehicle = tf.placeholder(tf.int32, [None, 2], name='active_vehicle_ind')

        # distance table: batch x n_customers x n_customers
        self.distance_table = tf.placeholder(tf.float32, [None, None, None], name='distance_table')

        # batch x n_vehicles x pos_hist (3) x position size (x, y)
        # self.vehicles_pos = tf.placeholder(tf.float32, [None, None, pos_hist, 2], name='vehicles_pos')
        self.vehicles_pos = tf.placeholder(tf.float32, [None, pos_hist, 2], name='vehicles_pos')

        self.time = tf.placeholder(tf.float32, [None, 1], name='time')

        # instance characteristics: DL, Q, SV
        self.instance_chars = tf.placeholder(tf.float32, [None, 6], name='instance_chars')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_customer_embedding(customers_set, available_customers, w_init, class_name,
                                     heads=4):
            # customers, node embedding
            wq = tf.get_variable('wq_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=fine_tune)
            wk = tf.get_variable('wk_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=fine_tune)
            wv = tf.get_variable('wv_att_cc', [self.feat_size_c, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=fine_tune)

            # Query: batch x n x e
            x_c = tf.matmul(customers_set, wq)

            # e'
            hsize = int(self.emb_size / heads)

            # Query: batch x n x e
            q = x_c

            # Key: batch x n x e
            k = tf.matmul(customers_set, wk)

            # Value: batch x n x e
            v = tf.matmul(customers_set, wv)

            # reshape to split
            batch_size, n_customers = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x n x e'
            q = tf.transpose(tf.reshape(q, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_customers, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x n x e' * batch x h x e' x n: batch x h x n x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # involve the distance table
            # ww = tf.get_variable('wd1_att_cc', [1, heads], initializer=w_init, collections=class_name,
            # dtype=tf.float32)
            # distance_table = tf.reshape(tf.matmul(tf.expand_dims(distance_table, axis=-1), ww),
            #                             [batch_size, heads, n_customers, n_customers, 1])
            # a = tf.concat([tf.expand_dims(a, axis=-1), distance_table], axis=-1)
            # wd = tf.get_variable('wd_att_cc', [2, 1], initializer=w_init, collections=class_name, dtype=tf.float32)
            # # batch x h x n x n
            # a = tf.reshape(tf.matmul(a, wd), [batch_size, heads, n_customers, n_customers])

            # eliminate the unavailable customers: batch x n x n ** batch x n

            a += tf.broadcast_to(tf.expand_dims(tf.expand_dims((1. - available_customers) * (-1e9), axis=1), axis=1),
                                 [batch_size, heads, n_customers, n_customers])

            # softmax: batch x h x n x n
            a = tf.nn.softmax(a, axis=-1)

            # graph embedding: batch x h x n x n * batch x h x n x e' -> batch x h x n x e' ->
            # batch x n x h x e' -> batch x n x e
            x_c_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, n_customers, self.emb_size])

            # Feed-forward:
            x_c_e += x_c

            return x_c_e

        def build_vehicle_embedding(vehicles_set, active_vehicle, w_init, class_name, heads=4):
            wq = tf.get_variable('wq_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=fine_tune)
            wk = tf.get_variable('wk_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=fine_tune)
            wv = tf.get_variable('wv_att_vv', [self.feat_size_v, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=fine_tune)

            # e'
            hsize = int(self.emb_size / heads)

            # embedding of the active vehicle: batch x e
            x_v_vbar = tf.matmul(tf.gather_nd(vehicles_set, active_vehicle, name="active_vehicle_emb"), wq)

            # Query: batch x 1 x e
            q = tf.expand_dims(x_v_vbar, 1)

            # Key: batch x n x e
            k = tf.matmul(vehicles_set, wk)

            # Value: batch x n x e
            v = tf.matmul(vehicles_set, wv)

            # reshape to split
            batch_size, n_vehicles = tf.shape(k)[0], tf.shape(k)[1]
            # batch x h x 1 x e'
            q = tf.transpose(tf.reshape(q, [batch_size, 1, heads, hsize]), [0, 2, 1, 3])
            # batch x h x n x e'
            k = tf.transpose(tf.reshape(k, [batch_size, n_vehicles, heads, hsize]), [0, 2, 1, 3])
            v = tf.transpose(tf.reshape(v, [batch_size, n_vehicles, heads, hsize]), [0, 2, 1, 3])

            ll = hsize * 1.

            # compatibility weights: batch x h x 1 x e' * batch x h x e' x n: batch x h x 1 x n
            a = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.math.sqrt(ll)

            # softmax: batch x h x 1 x n
            a = tf.nn.softmax(a, axis=-1)

            # vehicles embedding: batch x h x 1 x n * batch x h x n x e' -> batch x h x 1 x e' ->
            # batch x 1 x h x e' -> batch x 1 x e -> batch x e
            h_v_e = tf.reshape(tf.transpose(tf.matmul(a, v), [0, 2, 1, 3]), [batch_size, self.emb_size])

            # Feed-forward:
            h_v_e += x_v_vbar
            return h_v_e

        def build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name):
            w1 = tf.get_variable('w1_att_g', [h_z_size + self.emb_size, self.emb_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=fine_tune)
            w2 = tf.get_variable('w2_att_g', [self.emb_size, 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=fine_tune)

            batch_size, n_customers = tf.shape(h_c)[0], tf.shape(h_c)[1]
            # batch x e' -> batch x 1 x e' -> batch x n x e'
            context_extended = tf.broadcast_to(tf.expand_dims(h_z, axis=1),
                                               [batch_size, n_customers, h_z_size])

            # batch x n x (e + e')
            mu = tf.concat([context_extended, h_c], axis=-1)

            # batch x n
            weights = tf.reshape(tf.matmul(tf.nn.tanh(tf.matmul(mu, w1)), w2), [batch_size, n_customers])

            # Masking
            weights += (1. - available_customers) * (-1e9)
            # batch x n: batch x 1 x n
            weights = tf.expand_dims(tf.nn.softmax(weights, axis=-1), axis=1)

            # batch x 1 x n * batch x n x e -> batch x 1 x e -> batch x e
            h_g = tf.reshape(tf.matmul(weights, h_c), [batch_size, self.emb_size])

            return h_g

        def build_fc_q_network(state, state_size, x_c, target_customers,
                               w_init, b_init, class_name):
            batch_size = tf.shape(state)[0]

            brange = tf.expand_dims(
                tf.broadcast_to(tf.expand_dims(tf.range(batch_size), axis=1), [batch_size, self.nb]),
                axis=2)
            # batch x nb x 2
            indices = tf.concat([brange, tf.expand_dims(target_customers, axis=2)], axis=2)

            # gather the embedding of target customers for each batch: batch x nb x e
            x_t = tf.gather_nd(x_c, indices, name="gather_emb")

            # batch x nb x state_size
            bstate = tf.broadcast_to(tf.expand_dims(state, axis=1), [batch_size, self.nb, state_size])

            # batch x nb x (state_size + emb_size)
            state_actions = tf.concat([bstate, x_t], axis=2)
            state_size += self.emb_size
            h1 = int(state_size / 2.)

            w1 = tf.get_variable('w_fcd_1', [state_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=False)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name, trainable=False)

            w1tl = tf.get_variable('w_fcd_1_tl', [state_size, h1], initializer=w_init,
                                   collections=class_name, dtype=tf.float32)
            b1tl = tf.get_variable('b_fcd_1_tl', [1, h1], initializer=b_init, collections=class_name)

            # q_value = tf.nn.leaky_relu(tf.matmul(state_actions, w1) + b1)
            q_value = tf.nn.relu(tf.matmul(state_actions, w1) + b1 + tf.matmul(state_actions, w1tl) + b1tl)

            w2 = tf.get_variable('w_fcd_2', [h1, 2], initializer=w_init,
                                 collections=class_name, dtype=tf.float32, trainable=False)
            b2 = tf.get_variable('b_fcd_2', [1, 2], initializer=b_init, collections=class_name, trainable=False)

            w2tl = tf.get_variable('w_fcd_2_tl', [h1, 2], initializer=w_init,
                                   collections=class_name, dtype=tf.float32)
            b2tl = tf.get_variable('b_fcd_2_tl', [1, 2], initializer=b_init, collections=class_name)

            # batch x nb x 1
            q_value = tf.matmul(q_value, w2) + b2 + tf.matmul(q_value, w2tl) + b2tl

            # batch x nb*2
            q_value = tf.reshape(q_value, [batch_size, 2 * self.nb])

            # mask out dummy targets
            # q_value = q_value * available_targets
            return q_value

        def build_network(customers_set, vehicles_set, active_vehicle, target_customers,
                          available_customers, vehicles_pos, time, instance_chars,
                          w_init, b_init, class_name):
            # embed the raw data: batch x n x e, batch x m x e
            h_c = build_customer_embedding(customers_set, available_customers,
                                           w_init, class_name, heads=4)
            h_v = build_vehicle_embedding(vehicles_set, active_vehicle, w_init, class_name, heads=4)

            h_z = tf.concat([h_v, time], axis=1)
            h_z_size = self.emb_size + 1

            h_g = build_graph_attention(h_z, h_z_size, h_c, available_customers, w_init, class_name)
            obs = tf.concat([h_g, h_z, instance_chars], axis=1, name="concat_state")

            obs_size = h_z_size + self.emb_size + 6

            # we use x_c_e as the target customer
            # batch x nb x 2
            q_values = build_fc_q_network(obs, obs_size, h_c, target_customers, w_init, b_init, class_name)

            return q_values, obs

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval, self.state = build_network(self.customers, self.vehicles,
                                                    self.active_vehicle, self.target_customers,
                                                    self.available_customers, self.vehicles_pos, self.time,
                                                    self.instance_chars,
                                                    w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            # self.td_error = self.q_target - selected_q
            # self.loss = tf.reduce_mean(tf.square(self.td_error))
            huber = tf.keras.losses.Huber(delta=self.huber_p)
            self.loss = tf.reduce_mean(huber(self.q_target, selected_q))
            # self.loss = tf.reduce_sum(tf.math.log(tf.math.cosh(self.td_error)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
            # self.gradient = self.optimizer.compute_gradients(self.loss, var_list=var_list)
            # self.gradinet = [(tf.clip_by_norm(grad, 5), var) for grad, var in self.gradinet]
            # self.opt = self.optimizer.apply_gradients(self.gradient)
            self.opt = self.optimizer.minimize(self.loss, var_list=var_list)

        # ------------------ build target_net ------------------

        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_network(self.customers, self.vehicles,
                                        self.active_vehicle, self.target_customers,
                                        self.available_customers, self.vehicles_pos, self.time,
                                        self.instance_chars,
                                        w_initializer, b_initializer, c_names)[0]

        # ------------------ replace network ------------------
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def optimize(self, obs, target, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.available_customers: obs["available_customers"],
                     self.time: obs["time"],
                     self.selected_target: obs["selected_target"],
                     self.instance_chars: obs["instance_chars"],
                     self.q_target: target
                     }
        opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)
        # avg_gradient = np.mean([item for g in gradient for item in g[0].reshape(-1)])
        return loss, 0

    def value(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_eval, feed_dict=feed_dict)
        # values, mm, ss = sess.run([self.q_eval, self.embeddings, self.state], feed_dict=feed_dict)

        # batch x n_target_customers
        return values

    def value_(self, obs, sess):
        feed_dict = {self.customers: obs["customers"],
                     self.vehicles: obs["vehicles"],
                     self.target_customers: obs["target_customers"],
                     self.available_customers: obs["available_customers"],
                     self.active_vehicle: obs["active_vehicle"],
                     self.instance_chars: obs["instance_chars"],
                     self.time: obs["time"]
                     }
        values = sess.run(self.q_next, feed_dict=feed_dict)

        return values

    def set_opt_param(self, sess, new_lr=None, new_hp=None):
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_hp is not None:
            sess.run(tf.assign(self.huber_p, new_hp))

    def get_learning_params(self, sess):
        # learning rate and huber param
        return sess.run([self.lr, self.huber_p])

    @staticmethod
    def expand_dim(a):
        return np.expand_dims(a, axis=0)


# DQN for the centralized formulation
class CentralizedDQN(object):
    """
    it receives the full state (with dummy values for missing customers)
    and returns a qvalue for each customer

    """

    def __init__(self, init_lr, n, m, feat_size_c, feat_size_v):
        self.n = n
        self.m = m
        self.feat_size_c = feat_size_c
        self.feat_size_v = feat_size_v
        self.state_size = n * feat_size_c + m * feat_size_v + 1

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # in placeholders, the first axis is the batch size
        # batch x state_size (width)
        self.state = tf.placeholder(tf.float32, shape=[None, self.state_size], name="raw_state")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_fc_q_network(state, w_init, b_init, class_name):
            h1 = np.floor(2 * (self.state_size - self.n - 1) / 3.) + self.n + 1
            w1 = tf.get_variable('w_fcd_1', [self.state_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name)

            q_value = tf.nn.leaky_relu(tf.matmul(state, w1) + b1)

            h2 = np.floor((self.state_size - self.n - 1) / 3.) + self.n + 1
            w2 = tf.get_variable('w_fcd_2', [h1, h2], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b2 = tf.get_variable('b_fcd_2', [1, h2], initializer=b_init, collections=class_name)
            q_value = tf.matmul(q_value, w2) + b2

            w3 = tf.get_variable('w_fcd_3', [h2, self.n + 1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b3 = tf.get_variable('b_fcd_3', [1, self.n + 1], initializer=b_init, collections=class_name)
            q_value = tf.matmul(q_value, w3) + b3

            return q_value

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval = build_fc_q_network(self.state, w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            self.td_error = self.q_target - selected_q
            self.loss = tf.reduce_mean(tf.square(self.td_error))
            # huber = tf.keras.losses.Huber(delta=self.huber_p)
            # self.loss = tf.reduce_mean(huber(self.q_target, selected_q))
            # self.loss = tf.reduce_sum(tf.math.log(tf.math.cosh(self.td_error)))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")
            # self.gradient = self.optimizer.compute_gradients(self.loss, var_list=var_list)
            # self.gradinet = [(tf.clip_by_norm(grad, 5), var) for grad, var in self.gradinet]
            # self.opt = self.optimizer.apply_gradients(self.gradient)
            self.opt = self.optimizer.minimize(self.loss, var_list=var_list)

        # ------------------ build target_net ------------------

        with tf.variable_scope("target_net"):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_fc_q_network(self.state, w_initializer, b_initializer, c_names)

        # ------------------ replace network ------------------
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def optimize(self, states, selected_targets, target_values, sess):
        feed_dict = {self.state: states,
                     self.selected_target: selected_targets,
                     self.q_target: target_values
                     }
        # opt, loss, gradient = sess.run([self.opt, self.loss, self.gradient], feed_dict=feed_dict)
        opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)
        # avg_gradient = np.mean([abs(item) for g in gradient for item in g[0].reshape(-1)])
        return loss, 0

    def value(self, state, sess):
        feed_dict = {self.state: state}
        values = sess.run(self.q_eval, feed_dict=feed_dict)
        # values, mm, ss = sess.run([self.q_eval, self.embeddings, self.state], feed_dict=feed_dict)

        # batch x n_target_customers
        return values

    def value_(self, state, sess):
        feed_dict = {self.state: state}
        values = sess.run(self.q_next, feed_dict=feed_dict)

        return values

    def set_opt_param(self, sess, new_lr=None, new_hp=None):
        if new_lr is not None:
            sess.run(tf.assign(self.lr, new_lr))
        if new_hp is not None:
            sess.run(tf.assign(self.huber_p, new_hp))

    def get_learning_params(self, sess):
        # learning rate and huber param
        return sess.run([self.lr, self.huber_p])

    @staticmethod
    def expand_dim(a):
        return np.expand_dims(a, axis=0)