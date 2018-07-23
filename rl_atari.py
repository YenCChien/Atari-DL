import random
import numpy as np
import gym
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imresize

env = gym.make('SpaceInvaders-v0')

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))

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
        observation0 = preprocess(state)
        print("After processing: " + str(np.array(observation0).shape))
        plt.imshow(np.array(np.squeeze(observation0)))
        plt.show()