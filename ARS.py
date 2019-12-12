import numpy as np
import gym
import collections
import datetime
import utils


# neural networks as policy, with adam optimizer
# Periodically learning the agents with td3
class HP:
    # hyper parameters
    def __init__(self, env_name='Hopper-v2', total_episodes=1000, action_bound=1,
                 episode_length=1000, learning_rate=0.01, weight=0.01, learning_steps=100,
                 num_samples=8, noise=0.02, bc_index=[], std_dev=0.03, syn_step=10,
                 meta_population_size=5, seed=1, hidden_size=300):
        self.env = gym.make(env_name)
        np.random.seed(seed)
        self.env.seed(seed)
        self.action_bound = action_bound
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.shape[0]
        self.total_episodes = total_episodes
        self.episode_length = episode_length
        self.lr = learning_rate
        self.num_samples = num_samples
        self.noise = noise
        self.meta_population_size = meta_population_size
        self.seed = seed
        self.syn_step = syn_step
        self.learning_steps = learning_steps
        self.bc_index = bc_index
        self.weight = weight
        self.normalizer = utils.Normalizer(self.env.observation_space.shape[0])
        self.hidden_size = hidden_size
        self.stddev = std_dev


class Archive:
    # the archive, store the behavior
    def __init__(self, number_neighbour=10):
        self.data = []
        self.k = number_neighbour

    def add_policy(self, policy_bc):
        # 只存policy的bc
        self.data.append(policy_bc)

    def initialize(self, meta_population):
        for policy in meta_population.population:
            policy.evaluate()
            self.data.append(policy.bc)

    def novelty(self, policy_bc):
        # calculate the novelty of policy
        dist = np.sort(np.sum(np.square(policy_bc - np.array(self.data)), axis=1), axis=None)
        return np.mean(dist[:self.k])


class Normalizer:
    # Normalizes the input observations
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):  # observe a space, dynamic implementation of calculate mean of variance
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)  # 方差，清除了小于1e-2的

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


class Adam_optimizer:
    def __init__(self, lr):
        self.m_t = 0
        self.v_t = 0
        self.t = 0
        self.alpha = lr
        self.beta_1 = 0.9
        self.beta_2 = 0.99
        self.epsilon = 1e-8

    def update(self, g_t):
        self.t += 1
        self.m_t = self.beta_1 * self.m_t + (1 - self.beta_1) * g_t
        self.v_t = self.beta_2 * self.v_t + (1 - self.beta_2) * (g_t * g_t)
        m_cap = self.m_t / (1 - (self.beta_1 ** self.t))
        v_cap = self.v_t / (1 - (self.beta_2 ** self.t))
        return (self.alpha * m_cap) / (np.sqrt(v_cap) + self.epsilon)


class Policy:
    def __init__(self, hp):
        # behavior characteristic初始化为None
        self.hp = hp
        # 针对每个层
        self.w1 = np.random.randn(self.hp.input_size, self.hp.output_size) * self.hp.stddev
        self.b1 = np.zeros([self.hp.output_size, ])
        # used for adam update
        self.wa_1 = Adam_optimizer(self.hp.lr)
        self.ba_1 = Adam_optimizer(self.hp.lr)

    def get_params(self):
        return [self.w1,
                self.b1]

    def get_action(self, state, delta=None):
        state = np.reshape(state, [1, self.hp.input_size])
        if delta is None:
            output1 = np.dot(state, self.w1) + self.b1
            action = (output1)
        else:
            output1 = np.dot(state, self.w1 + delta[0]) + self.b1 + delta[1]
            action = (output1)
        return action

    def evaluate(self, delta=None):
        # 根据当前state执在环境中执行一次，返回获得的reward和novelty
        # env为环境，为了防止多次初始化这里传入环境
        total_reward = 0
        total_step = 0
        obs = self.hp.env.reset()
        for i in range(self.hp.episode_length):
            self.hp.normalizer.observe(obs)
            action = np.clip(self.get_action(self.hp.normalizer.normalize(obs), delta=delta), -1, 1)
            # action = np.clip(self.get_action(obs, delta=delta), -1, 1)
            next_obs, reward, done, _ = self.hp.env.step(action)
            obs = next_obs
            total_reward += reward
            total_step += 1
            if done:
                break
        return total_reward, total_step

    def adam_update(self, rollouts, sigma_rewards):
        step = 0
        for r, delta in rollouts:
            step += r * delta[0]
        grad = self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        self.w1 += self.wa_1.update(grad)
        step = 0
        for r, delta in rollouts:
            step += r * delta[1]
        grad = self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        self.b1 += self.ba_1.update(grad)

    def update(self, rollouts, sigma_rewards):
        step = 0
        # 针对每个参数进行更新
        for r, delta in rollouts:
            step += r * delta[0]
        self.w1 += self.hp.lr / (self.hp.num_samples * sigma_rewards) * step
        step = 0
        for r, delta in rollouts:
            step += r * delta[1]
        self.b1 += self.hp.lr / (self.hp.num_samples * sigma_rewards) * step

    def sample_deltas(self):
        return [np.random.randn(*self.w1.shape) * self.hp.noise,
                np.random.randn(*self.b1.shape) * self.hp.noise]

    def td3_soft_update(self, params):
        self.w1 = self.w1 * (1 - self.hp.weight) + self.hp.weight * params[0]
        self.b1 = self.b1 * (1 - self.hp.weight) + self.hp.weight * params[1]

    def td3_update(self, params):
        self.w1 = params[0]
        self.b1 = params[1]


class ARS_TD3:
    def __init__(self, hp):
        self.hp = hp

    def train(self):
        policy = Policy(self.hp)
        reward_memory = []
        total_step = []
        current_step = 0
        for t in range(self.hp.total_episodes):
            start_time = datetime.datetime.now()
            deltas = [policy.sample_deltas() for _ in
                      range(self.hp.num_samples)]
            forward_reward_list = []
            backward_reward_list = []
            for i in range(self.hp.num_samples):
                delta = deltas[i]
                reward_forward, step1 = policy.evaluate(delta)
                neg_delta = [-delta[_] for _ in range(len(delta))]
                reward_backward, step2 = policy.evaluate(neg_delta)
                forward_reward_list.append(reward_forward)
                backward_reward_list.append(reward_backward)
                current_step += step1 + step2
            rollouts = [((forward_reward_list[j] - backward_reward_list[j]),
                         deltas[j]) for j in range(self.hp.num_samples)]
            sigma_rewards = np.std(np.array(forward_reward_list + backward_reward_list))
            policy.update(rollouts, sigma_rewards)
            #policy.adam_update(rollouts, sigma_rewards)
            test_reward, _ = policy.evaluate()
            total_step.append(current_step)
            print('#######')
            print('Episode ', t)
            print('Total reward is: ', test_reward)
            print('Total step is: ', current_step)
            print('Running time:', (datetime.datetime.now() - start_time).seconds)
            reward_memory.append(test_reward)
        return reward_memory, total_step
