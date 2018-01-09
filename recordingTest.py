import gym
from gym import wrappers

env = gym.make('Enduro-v0')
env = wrappers.Monitor(env, '/10703-assignment2-YHuang-YZhou', force=True)
observation = env.reset()
for i_episode in range(101):
    for t in range(1, 101):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
            break

env.close()
gym.upload('/10703-assignment2-YHuang-YZhou', api_key='sk_tkcjGIDjS1K6ch6y0grpQ')