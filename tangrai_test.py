import matplotlib.pyplot as plt
import copy
import numpy as np

import gym

env = gym.make("TangrAI-v0")
observation = env.reset()
for _ in range(7):
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    board_ = copy.deepcopy(observation)
    board_ = np.reshape(board_, (20, 10))
    plt.imshow(board_)
    plt.show()
