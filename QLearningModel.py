import numpy as np
from random import randrange
from scipy.ndimage.interpolation import rotate


class QLearningModel:
    def __init__(self):
        self.alpha = 0.4
        self.gamma = 0.3
        self.angle = 10
        # executing: self.actions[0](picture)
        self.actions = dict([(0, self.action_rotate_plus), (1, self.action_rotate_minus), (2, self.action_invariant)])
        self.states = [0, 1, 2]
        self.tableQ = np.array((len(self.actions), len(self.states)))

    def action_rotate_plus(self, picture):
        return rotate(picture, self.angle)

    def action_rotate_minus(self, picture):
        return rotate(picture, -self.angle)

    def action_invariant(self, picture):
        return picture

    def selectAction(self):
        return self.actions[randrange(len(self.actions))]

    def get_features_metric(self, features):
        return np.std(features)

    def get_reward(self, m1, m2):
        return np.sign(m2-m1)
