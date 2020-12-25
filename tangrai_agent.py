# EXECUTE ENVIRONMENT

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
import matplotlib.pyplot as plt
from tangrai import __init__ as aaaaa
import copy
import time


class Agent():
    def __init__(self, env_id, path, episodes, max_env_steps, win_threshold, epsilon_decay,
                 state_size=None, action_size=None, epsilon=1.0, epsilon_min=0.01,
                 gamma=1.0, learning_rate=.001, alpha_decay=.01, batch_size=16, prints=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make(env_id)
        if state_size is None:
            self.state_size = self.env.observation_space.n
        else:
            self.state_size = state_size

        if action_size is None:
            self.action_size = self.env.action_space.n
        else:
            self.action_size = action_size

        self.episodes = episodes
        self.env._max_episode_steps = max_env_steps
        self.win_threshold = win_threshold
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.path = path  # location where the model is saved to
        self.prints = prints  # if true, the agent will print his scores
        self.learning_rate = learning_rate
        self.model = self.NN_model()
        self.mse_list = []

    def NN_model(self):
        model = Sequential()
        model.add(Dense(256 * 2, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128 * 2, activation='relu'))
        model.add(Dense(64 * 2, activation='relu'))
        model.add(Dense(32 * 2, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate, decay=self.alpha_decay), metrics=['mse'])
        return model

    def act(self, state):
        if (np.random.random() <= self.epsilon):
            print(self.model.predict(state))  # Q-Table!!!!
            return self.env.action_space.sample()

        return np.argmax(self.model.predict(state))  # Returns the position XY of the next piece

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))  # select random samples in batchsize
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        history = self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

        self.mse_list = np.append(self.mse_list, history.history['mse'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self):
        for episode in range(self.episodes):  # Number of desired episodes
            state = self.env.reset()  # Empty the board and score
            done = False
            counter_steps = 0
            score = 0
            if episode % 10 == True:
                print('EPISODE', episode)
                print('Mean MSE', np.mean(self.mse_list))

            for i in range(self.env._max_episode_steps):  # Number of movements = pieces = 7
                action_space = self.act(state)  # Returns the XY of the next piece
                next_state, reward, done, information = self.env.step(action_space)  # Movement-> reward
                self.remember(state, action_space, reward, next_state, done)  # Save the results
                self.replay(self.batch_size)  # Fit the model using old results

                score += reward  # Add up the score
                state = next_state

                # Uncomment this to see the plots
                board_ = copy.deepcopy(state)
                board_ = np.reshape(board_, (20, 10))
                
                """
                if (episode % 10000 == 0):
                 
                    plt.imshow(board_)
                    plt.show()
                    print("hola")

                    plt.close
                    # Board and Piece in different plot
                    print('#' * 10)
                    plt.figure()
                    plt.title('Board')
                    plt.imshow(board_[:10])
                    plt.show()
                    
                    plt.close

                    plt.figure()
                    plt.title('Piece')
                    plt.imshow(board_[10:21])
                    plt.show()
                """

                if counter_steps == 6:
                    done = True
                    break
                else:
                    done = False

                counter_steps += 1
        self.model.save_weights(self.path)


if __name__ == "__main__":
    agent = Agent(env_id='TangrAI-v0',
                  path='model/model_RL.h5',
                  episodes=5000,
                  max_env_steps=7,
                  win_threshold=None,
                  epsilon_decay=1,  # Effect a lot to the model
                  state_size=200,
                  action_size=None,
                  epsilon=0.8,
                  epsilon_min=0.01,
                  gamma=0.8,
                  learning_rate=.001,
                  alpha_decay=0.1,
                  batch_size=4,
                  prints=True)

    agent.train()
