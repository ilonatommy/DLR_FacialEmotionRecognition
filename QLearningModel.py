import numpy as np
from random import randrange

from PIL import Image, ImageOps
from scipy.ndimage.interpolation import rotate


class QLearningModel:
    def __init__(self):
        self.alpha = 0.4
        self.gamma = 0.3
        self.angle1 = 90
        self.angle2 = 180
        self.angle3 = 10
        self.angle4 = -10
        # executing: self.actions[0](picture)
        self.actions = dict([(0, self.action_rotate_1), (1, self.action_rotate_2), (2, self.diagonal_translation)])
        self.states = [0, 1]
        self.tableQ = np.zeros((len(self.states), len(self.actions)))
        self.maxIter = len(self.actions) * 20

    def action_rotate_1(self, picture):
        return rotate(picture, self.angle1, reshape=False)

    def action_rotate_2(self, picture):
        return rotate(picture, self.angle2, reshape=False)

    def action_rotate_3(self, picture):
        return rotate(picture, self.angle3, reshape=False)

    def action_rotate_4(self, picture):
        return rotate(picture, self.angle4, reshape=False)

    def action_invariant(self, picture):
        return picture

    def diagonal_translation(self, picture):
        img = Image.fromarray(picture.astype('uint8'), 'RGB')
        w = int(img.size[0] * 0.75)
        h = int(img.size[1] * 0.75)
        border = (15, 15, img.size[0] - w - 15, img.size[1] - h - 15)
        img = img.resize((w, h), Image.ANTIALIAS)
        translated = ImageOps.expand(img, border=border, fill='black')
        return np.array(translated)

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

    def action_space_search_choose_optimal(self, cnn, img, statsController):
        img_features = cnn.get_output_base_model(img)
        m1 = self.get_features_metric(img_features)
        optimal_action = 4
        for idx, action in enumerate(self.actions):
            statsController.updateAllActionStats(action)
            modified_img = self.apply_action(action, img)
            modified_img_features = cnn.get_output_base_model(modified_img)
            m2 = self.get_features_metric(modified_img_features)
            if m2 > m1:
                optimal_action = idx
        return optimal_action

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
