from ConvolutionalNeuralNetworks import ConvolutionalNeuralNetworks
from DataLoader import DataLoader
from ImageHelper import showNumpyImg, NumpyImg2Tensor
from QLearningModel import QLearningModel

IMG_SIZE = 64

classes = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
dl = DataLoader("/home/ilona/Data/Dokumenty/Master_Studies/semestr_2/inz_wiedzy_symboliczne_ml/FERG_DB_256",
                ".png",
                classes,
                IMG_SIZE,
                7000)
images, labels = dl.demo_load()

# TODO: train the networks, save them on disk and load them instead of default CNN in ConolutionalNeuralNetwork class
cnn = ConvolutionalNeuralNetworks("ResNet", (IMG_SIZE, IMG_SIZE, 3))
q = QLearningModel()

verbose = True
for img in images:
    no_lr_probabilities_vector = cnn.classifier.predict(NumpyImg2Tensor(img))
    q.perform_iterative_Q_learning(cnn, img)
    optimal_action = q.choose_optimal_action()
    corrected_img = q.apply_action(optimal_action, img)
    probabilities_vector = cnn.classifier.predict(NumpyImg2Tensor(corrected_img))
    print("before: ", no_lr_probabilities_vector)
    print("after: ", probabilities_vector)
