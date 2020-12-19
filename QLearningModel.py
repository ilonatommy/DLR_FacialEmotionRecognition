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
        self.states = [0, 1]
        self.tableQ = np.zeros((len(self.states), len(self.actions)))
        self.maxIter = len(self.actions) * 5

    def action_rotate_plus(self, picture):
        return rotate(picture, self.angle, reshape=False)

    def action_rotate_minus(self, picture):
        return rotate(picture, -self.angle, reshape=False)

    def action_invariant(self, picture):
        return picture

    def selectAction(self):
        return randrange(len(self.actions))

    def apply_action(self, action, img):
        return self.actions[action](img)

    def get_features_metric(self, features):
        return np.std(features)

    def get_reward(self, m1, m2):
        return np.sign(m2-m1)

    def define_state(self, reward):
        return 0 if reward > 0 else 1

    def update_tableQ(self, state, action, reward):
        self.tableQ[state][action] = self.tableQ[state][action] + self.alpha * (
            reward + self.gamma * max(self.tableQ[state]) - self.tableQ[state][action]
        )

    def perform_iterative_Q_learning(self, cnn, img, statsController):
        img_features = cnn.get_output_base_model(img)
        m1 = self.get_features_metric(img_features)
        for i in range(self.maxIter):
            action = self.selectAction()
            statsController.updateAllActionStats(action)
            modified_img = self.apply_action(action, img)
            modified_img_features = cnn.get_output_base_model(modified_img)
            m2 = self.get_features_metric(modified_img_features)
            reward = self.get_reward(m1, m2)
            state = self.define_state(reward)
            self.update_tableQ(state, action, reward)

    def choose_optimal_action(self):
        return np.where(self.tableQ == np.amax(self.tableQ))[1][0]
