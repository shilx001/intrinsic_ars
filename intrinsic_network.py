import tensorflow as tf
import numpy as np
import utils


class IntrinsicNetwork(object):
    def __init__(self, state_dim, action_dim, action_bound=1, hidden_size=64, lr=1e-3, namescope='default', seed=1,
                 weight=0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.action_bound = action_bound
        self.weight = weight
        self.namescope = namescope
        tf.set_random_seed(seed)
        self.sess = tf.Session()
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])
        self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.predict_state = self.build_forward_network(self.state, self.action, self.namescope + 'forward')
        self.predict_action = self.build_curiosity_network(self.state, self.next_state, self.namescope + 'curiosity')
        self.loss_forward = tf.reduce_mean(tf.square(self.next_state - self.predict_state))
        self.loss_curiosity = tf.reduce_mean(tf.square(self.action - self.predict_action))
        self.train_forward_network = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_forward)
        self.train_curiosity_network = tf.train.AdadeltaOptimizer(learning_rate=self.lr).minimize(self.loss_curiosity)
        self.sess.run(tf.global_variables_initializer())

    def get_bonus(self, state, action, next_state):
        state = np.reshape(state, [-1, self.state_dim])
        action = np.reshape(action, [-1, self.action_dim])
        next_state = np.reshape(next_state, [-1, self.state_dim])
        predict_state = self.sess.run(self.predict_state, feed_dict={self.state: state,
                                                                     self.action: action})
        predict_action = self.sess.run(self.predict_action, feed_dict={self.state: state,
                                                                       self.next_state: next_state})
        return self.weight * np.mean(np.square(predict_state - next_state)) + (1 - self.weight) * np.mean(
            np.square(predict_action - action))

    def build_forward_network(self, state, action, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            inputs = tf.concat([state, action], axis=1)
            l1 = tf.layers.dense(inputs, self.hidden_size, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
            l2 = tf.layers.dense(l1, self.hidden_size, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
            l3 = tf.layers.dense(l2, self.state_dim, activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
            return l3

    def build_curiosity_network(self, state, next_state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            inputs = tf.concat([state, next_state], axis=1)
            l1 = tf.layers.dense(inputs, self.hidden_size, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
            l2 = tf.layers.dense(l1, self.hidden_size, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
            l3 = tf.layers.dense(l2, self.action_dim, activation=tf.nn.tanh,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
        return l3 * self.action_bound

    def learn(self, state, action, next_state):
        state = np.reshape(state, [-1, self.state_dim])
        action = np.reshape(action, [-1, self.action_dim])
        next_state = np.reshape(next_state, [-1, self.state_dim])
        self.sess.run([self.train_forward_network, self.train_curiosity_network], feed_dict={self.state: state,
                                                                                             self.action: action,
                                                                                             self.next_state: next_state})
