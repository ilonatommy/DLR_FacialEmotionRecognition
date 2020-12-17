from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks
from DataLoader import DataLoader
from Plotter import *
from ImageHelper import NumpyImg2Tensor
from QLearningModel import QLearningModel
from sklearn.model_selection import train_test_split
import time

LOAD_DATA = False
TRAIN_NETWORK = False
LIMIT = 100

# ----------Data Load-----------------
t1 = time.time()
IMG_SIZE = 64

classes = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
# limit of photos per 1 emotion per 1 person is 100
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
cnn = ConvolutionalNeuralNetworks("ResNet", dl.datasetInfo, (IMG_SIZE, IMG_SIZE, 3))
if TRAIN_NETWORK:
    cnn.history = cnn.model.fit(X_train, dl.toOneHot(y_train), batch_size=20, epochs=400, validation_split=0.2,
                                callbacks=cnn.callbacks)
    plot_history(cnn.history.history)
    dl.save_training_history(cnn.history.history)
    dl.save_model(cnn.networkName, cnn.model)
else:
    dl.load_model(cnn.networkName)
    history = dl.load_training_history()
    plot_history(history)
print("CNN training time: " + str(time.time() - t2))

# ---------CNN test evaluation--------
t3 = time.time()
results = cnn.model.evaluate(X_test, dl.toOneHot(y_test))
print("test loss, test acc:", results)
print("CNN evaluate time: " + str(time.time() - t3))

# ----------RL execution--------------
t4 = time.time()
q = QLearningModel()
verbose = True
qAcc = 0.0
for img, label in zip(X_test, y_test):
    no_lr_probabilities_vector = cnn.model.predict(NumpyImg2Tensor(img))
    q.perform_iterative_Q_learning(cnn, img)
    optimal_action = q.choose_optimal_action()
    corrected_img = q.apply_action(optimal_action, img)
    probabilities_vector = cnn.model.predict(NumpyImg2Tensor(corrected_img))
    qAcc = q.evaluate(qAcc, probabilities_vector, label)
qAcc /= len(y_test)
print("acc before using RL: ", results[1]) #0.13411764705882354
print("acc after using RL: ", qAcc) #0.15764705882352942
print("RL execution time: " + str(time.time() - t4))
