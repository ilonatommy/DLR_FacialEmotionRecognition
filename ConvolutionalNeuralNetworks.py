from keras.models import Model
from keras.applications import ResNet50, InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout


class ConvolutionalNeuralNetworks:
    def __init__(self, networkName, shape=(64, 64, 3)):
        self.networkName = networkName
        if self.networkName == "ResNet":
            self.model = ResNet50(include_top=False, weights="imagenet", input_shape=shape)
            gavp = GlobalAveragePooling2D()(self.model.output)
            d1 = Dense(1024, 'relu')(gavp)
            d2 = Dense(1024, 'relu')(d1)
            d3 = Dense(1024, 'relu')(d2)
            d4 = Dense(512, 'relu')(d3)
            d5 = Dense(7, 'softmax')(d4)
            self.classifier = Model(inputs=self.model.input, outputs=d5)
        if self.networkName == "Inception":
            self.model = InceptionV3(include_top=False, weights="imagenet", input_shape=shape)
            f = Flatten(self.model.output)
            d1 = Dense(1024, 'relu')(f)
            do1 = Dropout(0.2)(d1)
            d2 = Dense(7, 'softmax')(do1)
            self.classifier = Model(inputs=self.model.input, outputs=d2)
