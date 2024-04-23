import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def observe(self, state, action, next_state, reward):
        pass

    def select_action(self, state):
        return np.random.choice(self.env.action_space.n)

    def update(self):
        pass