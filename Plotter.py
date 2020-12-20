import os

from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd


def plot_history(dataLoader, networkName, history):
    plt.figure(figsize=(10, 7))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(dataLoader.resultsDir, 'train_history_loss' + networkName + dataLoader.datasetInfo + '.png'))
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(dataLoader.resultsDir, 'train_history_acc' + networkName + dataLoader.datasetInfo + '.png'))
    plt.show()


def plot_actions_stats(dataLoader, networkName, actions, stats, filename):
    plt.bar(actions, height=stats)
    plt.title('actions statistics')
    plt.ylabel('number of times action was chosen')
    plt.xlabel('action name')
    plt.savefig(os.path.join(dataLoader.resultsDir, 'actions_stats' + filename + networkName + dataLoader.datasetInfo +
                             '.png'))
    plt.show()


def plot_conf_matrix(dataLoader, networkName, conf_matrix, classes, filename):
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(dataLoader.resultsDir, 'conf_matrix' + filename + networkName + dataLoader.datasetInfo +
                             '.png'))


def print_classification_details(statController):
    print("accuracy: ", statController.accuracy)
    print("precision: ", statController.precision)
    print("recall: ", statController.recall)
    print("F1 score: ", statController.f1Score)
    print("report: ", statController.report)
