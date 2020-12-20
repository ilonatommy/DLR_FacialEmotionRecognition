import numpy as np


class StatisticsController:
    def __init__(self, classes, actions_cnt=0):
        self.classes = classes
        self.trainingHistory = None
        self.confMatrix = np.zeros((len(classes), len(classes)))
        self.optimalActionsStats = [0] * actions_cnt
        self.allActionsStats = [0] * actions_cnt
        self.predictedLabels = []
        self.recall = None
        self.precision = None
        self.f1Score = None
        self.report = None
        self.accuracy = None

    def updateOptimalActionsStats(self, action):
        self.optimalActionsStats[action] += 1

    def updateAllActionStats(self, action):
        self.allActionsStats[action] += 1
