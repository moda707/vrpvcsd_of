import numpy as np
import tensorflow as tf


# Double Q Network
class DQN(object):
    """
    Input: full state or observation
    Output: a set of q values for the action set (all customers or a subset of them)

    init_lr: initial learning rate

    """

    def __init__(self, init_lr, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.huber_p = tf.Variable(initial_value=20., dtype=tf.float32, trainable=False)

        # batch x state_size
        self.state = tf.placeholder(tf.float32, shape=[None, self.state_size], name="raw_state")

        # batch x 2
        self.selected_target = tf.placeholder(tf.int32, [None, 2], name='selected_target_ind')

        self.lr = tf.Variable(initial_value=init_lr, dtype=tf.float32, trainable=False)

        self.q_target = tf.placeholder(tf.float32, [None], name='Q_target')

        def build_fc_q_network(state, w_init, b_init, class_name):
            """
            A simple neural network with two hidden layers with sizes 2/3 and 1/3 of (input_size-output_size)
            Args:
                state: the input state
                w_init: weight initializer
                b_init: bias initializer
                class_name:

            Returns: a set of q values b x action_size

            """
            h1 = np.floor(2 * (self.state_size - self.action_size - 1) / 3.) + self.action_size + 1
            w1 = tf.get_variable('w_fcd_1', [self.state_size, h1], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b1 = tf.get_variable('b_fcd_1', [1, h1], initializer=b_init, collections=class_name)

            q_value = tf.nn.leaky_relu(tf.matmul(state, w1) + b1)

            h2 = np.floor((self.state_size - self.action_size - 1) / 3.) + self.action_size + 1
            w2 = tf.get_variable('w_fcd_2', [h1, h2], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b2 = tf.get_variable('b_fcd_2', [1, h2], initializer=b_init, collections=class_name)
            q_value = tf.matmul(q_value, w2) + b2

            w3 = tf.get_variable('w_fcd_3', [h2, self.action_size], initializer=w_init,
                                 collections=class_name, dtype=tf.float32)
            b3 = tf.get_variable('b_fcd_3', [1, self.action_size], initializer=b_init, collections=class_name)
            q_value = tf.matmul(q_value, w3) + b3

            return q_value

        with tf.variable_scope("eval_net"):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(mean=0., stddev=.1), tf.constant_initializer(0.)  # config of layers

            self.q_eval = build_fc_q_network(self.state, w_initializer, b_initializer, c_names)

            # batch x 1
            selected_q = tf.gather_nd(self.q_eval, self.selected_target, name="gather_qselected")

            # MSE loss:
            # self.td_error = self.q_target - selected_q
            # self.loss = tf.reduce_mean(tf.square(self.td_error))

            # Hubber loss:
            huber = tf.keras.losses.Huber(delta=self.huber_p)
            self.loss = tf.reduce_mean(huber(self.q_target, selected_q))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="eval_net")

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
        opt, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)
        return loss, 0

    def value(self, state, sess):
        feed_dict = {self.state: state}
        values = sess.run(self.q_eval, feed_dict=feed_dict)
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

