import random
import numpy as np
import gym

env = gym.make('SpaceInvaders-v0')

for x in range(10):
    terminal = True
    if terminal:     
        terminal = False
        state = env.reset()
    while not terminal:
        env.render()
        # action_idx = np.random.choice(6,1)
        action_idx = env.action_space.sample()
        state, reward, terminal, _ = env.step(action_idx)
        print('action:', action_idx, 'reward:', reward, len(state[0]), len(state[1]))
        input('next step')