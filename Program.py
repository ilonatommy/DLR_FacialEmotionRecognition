from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks
from DataLoader import DataLoader
from Plotter import *
from ImageHelper import NumpyImg2Tensor, ShowNumpyImg
from QLearningModel import QLearningModel
from sklearn.model_selection import train_test_split
import time
from StatisticsController import StatisticsController
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
import numpy as np

# set to true is algorithm is launched for the first time
LOAD_DATA = False
TRAIN_NETWORK = False
# limit of photos per 1 emotion per 1 person is 100
LIMIT = 10
ACTION_NAMES = ['rotate +90', 'rotate +180', 'diagonal translation']
networkName = "Inception"

# ----------Data Load-----------------
t1 = time.time()
IMG_SIZE = 75

classes = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
dl = DataLoader("/home/ilona/Data/Dokumenty/Master_Studies/semestr_2/inz_wiedzy_symboliczne_ml/FERG_DB_256",
                ".png",
                classes,
                IMG_SIZE,
                LIMIT)
if LOAD_DATA:
    images, labels = dl.load()
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=1)
    dl.save_train_test_split(X_train, X_test, y_train, y_test)
else:
    X_train, X_test, y_train, y_test = dl.load_train_test_split()
print("Data Load time: " + str(time.time() - t1))

# ---------CNN training---------------
t2 = time.time()
cnn = ConvolutionalNeuralNetworks(networkName, dl.datasetInfo)
cnn.create_model_architecture(X_train[0].shape)
statControllerNoRl = StatisticsController(classes)
if TRAIN_NETWORK:
    statControllerNoRl.trainingHistory = cnn.model.fit(X_train, dl.toOneHot(y_train), batch_size=20, epochs=400,
                                                       validation_split=0.2, callbacks=cnn.callbacks).history
    dl.save_training_history(statControllerNoRl.trainingHistory)
    dl.save_model(cnn.networkName, cnn.model)
else:
    dl.load_model_weights(networkName, cnn.model)
    statControllerNoRl.trainingHistory = dl.load_training_history()
print("CNN training time: " + str(time.time() - t2))

# ----------RL execution--------------
t4 = time.time()
q = QLearningModel()
statControllerRl = StatisticsController(classes, len(ACTION_NAMES))
verbose = True
for img, label in zip(X_test, y_test):
    no_lr_probabilities_vector = cnn.model.predict(NumpyImg2Tensor(img))
    predictedLabel = np.argmax(no_lr_probabilities_vector)
    statControllerNoRl.predictedLabels.append(predictedLabel)

    # article version:
    """
    if predictedLabel != label:
        q.perform_iterative_Q_learning(cnn, img, statControllerRl)
        optimal_action = q.choose_optimal_action()
        statControllerRl.updateOptimalActionsStats(optimal_action)
        corrected_img = q.apply_action(optimal_action, img)

        probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
        statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))
    else:
        statControllerRl.predictedLabels.append(predictedLabel)
    """

    # correct testing of article version:
    q.perform_iterative_Q_learning(cnn, img, statControllerRl)
    optimal_action = q.choose_optimal_action()
    statControllerRl.updateOptimalActionsStats(optimal_action)
    corrected_img = q.apply_action(optimal_action, img)

    probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
    statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))

    # best action, no RL version:
    """
    #optimal_action = q.action_space_search_choose_optimal(cnn, img, statControllerRl)

    #statControllerRl.updateOptimalActionsStats(optimal_action)
    #corrected_img = q.apply_action(optimal_action, img)

    #probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
    #statControllerRl.predictedLabels.append(np.argmax(probabilities_vector))
    """

print("RL execution time: " + str(time.time() - t4))

plot_actions_stats(dl, networkName, ACTION_NAMES, statControllerRl.allActionsStats, "allActionsRL")
plot_actions_stats(dl, networkName, ACTION_NAMES, statControllerRl.optimalActionsStats, "optimalActionsRL")

conf_matrix_no_RL = confusion_matrix(y_test, statControllerNoRl.predictedLabels)
conf_matrix_RL = confusion_matrix(y_test, statControllerRl.predictedLabels)
plot_conf_matrix(dl, networkName, conf_matrix_no_RL, classes, "NoRL")
plot_conf_matrix(dl, networkName, conf_matrix_RL, classes, "RL")
plot_history(dl, networkName, statControllerNoRl.trainingHistory)

statControllerNoRl.f1Score = f1_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.precision = precision_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.recall = recall_score(y_test, statControllerNoRl.predictedLabels, average="macro")
statControllerNoRl.report = classification_report(y_test, statControllerNoRl.predictedLabels)
statControllerNoRl.accuracy = accuracy_score(y_test, statControllerNoRl.predictedLabels)

statControllerRl.f1Score = f1_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.precision = precision_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.recall = recall_score(y_test, statControllerRl.predictedLabels, average="macro")
statControllerRl.report = classification_report(y_test, statControllerRl.predictedLabels)
statControllerRl.accuracy = accuracy_score(y_test, statControllerRl.predictedLabels)

print_classification_details(statControllerNoRl)
print_classification_details(statControllerRl)
dl.save_details(statControllerNoRl, networkName, "NoRL")
dl.save_details(statControllerRl, networkName, "RL")

