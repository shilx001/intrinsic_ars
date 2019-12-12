from iARS import *

env = 'HalfCheetah-v2'
seeds = [1, 2, 3, 4, 5]

for seed in seeds:
    hp = HP(env_name=env, seed=seed, num_samples=8, weight=1)
    agent = IntrinsicARS(hp)
    reward, step = agent.train()
    pickle.dump((reward, step), open(env + '_iars_seeds_' + str(seed), mode='wb'))
