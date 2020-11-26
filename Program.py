import numpy as np

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

# model inside cnn should be trained I think - check it in the article and alter accordingly
cnn = ConvolutionalNeuralNetworks("ResNet", (IMG_SIZE, IMG_SIZE, 3))
q = QLearningModel()

verbose = True
for img in images:
    img_features = cnn.model.predict(NumpyImg2Tensor(img))
    m1 = q.get_features_metric(img_features)
    modified_img = q.selectAction()(img)
    if verbose:
        showNumpyImg(modified_img)
        showNumpyImg(img)
    modified_img_features = cnn.model.predict(NumpyImg2Tensor(modified_img))
    m2 = q.get_features_metric(modified_img_features)
    print(m1)
    print(m2)
