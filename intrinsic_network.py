import tensorflow as tf
import numpy as np
import utils


class IntrinsicNetwork(object):
    def __init__(self, state_dim, action_dim, hidden_size=64, lr=1e-3, namescope='default', seed=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.namescope = namescope
        tf.set_random_seed(seed)
        self.sess = tf.Session()
        self.state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])
        self.next_state = tf.placeholder(dtype=tf.float32, shape=[None, state_dim])
        self.predict_state = self.build_network(self.state, self.action, self.namescope)
        self.loss = tf.reduce_mean(tf.square(self.next_state - self.predict_state))
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        # self.replay = utils.ReplayBuffer()
        self.sess.run(tf.global_variables_initializer())

    def get_bonus(self, state, action, next_state):
        state = np.reshape(state, [-1, self.state_dim])
        action = np.reshape(action, [-1, self.action_dim])
        predict_state = self.sess.run(self.predict_state, feed_dict={self.state: state,
                                                                     self.action: action})
        return np.mean(np.square(predict_state - next_state))

    def build_network(self, state, action, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            inputs = tf.concat([state, action], axis=1)
            l1 = tf.layers.dense(inputs, self.hidden_size, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
            l2 = tf.layers.dense(l1, self.hidden_size, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
            l3 = tf.layers.dense(l2, self.state_dim, activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.03))
            return l3

    def learn(self, state, action, next_state):
        state = np.reshape(state, [-1, self.state_dim])
        action = np.reshape(action, [-1, self.action_dim])
        next_state = np.reshape(next_state, [-1, self.state_dim])
        self.sess.run(self.train, feed_dict={self.state: state,
                                             self.action: action,
                                             self.next_state: next_state})
