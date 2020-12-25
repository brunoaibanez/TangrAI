import gym, numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from tangrai.envs import tangrai_engine as game

screenHeight = 480
screenWidth = 480


class tangraiENV(gym.Env):
    def __init__(self):
        self.game_state = game.GameState()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.observation_space = spaces.Box(low=0, high=255, shape=(screenHeight, screenWidth, 3))
        self._seed = None

    def _step(self, a):
        self._action_set = a
        reward = 0.0
        state, reward, done = self.game_state.frame_step(self._action_set)
        return state, reward, done, {}

    @property
    def _n_actions(self):
        return self._action_set

    def _reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(screenHeight, screenWidth, 3))
        state = self.game_state.frame_step(100)
        return state
