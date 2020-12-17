import os
from keras.utils.np_utils import to_categorical
import keras
import numpy as np
from PIL import Image
import glob


class DataLoader:
    def __init__(self, path, extension, classes, img_size, limit):
        self.path = path
        self.extension = str(extension)
        self.classes = classes
        self.img_size = img_size
        self.limit = limit
        self.datasetInfo = '_' + str(self.img_size) + '_limit_' + str(self.limit)

    def load(self):
        images = []
        labels = []
        for character_dir in glob.glob(os.path.join(self.path, "*", "")):
            for emotion_dir in glob.glob(os.path.join(character_dir, "*", "")):
                emotion_dir_name = os.path.basename(os.path.normpath(emotion_dir))
                emotion_name = emotion_dir_name.split("_")[1]
                emotion_idx = self.classes[emotion_name] # make one-hot from this
                i = 0
                for img_name in glob.glob(os.path.join(emotion_dir, "*" + self.extension)):
                    if self.limit and i > self.limit:
                        break
                    img = Image.open(img_name).resize((self.img_size, self.img_size))
                    # removing the 4th dim which is transparency and rescaling to 0-1 range
                    im = np.array(img)[..., :3] / 255
                    images.append(im)
                    labels.append(emotion_idx)
                    i += 1
        return np.array(images), np.array(labels)

    def demo_load(self):
        path1 = os.path.join(self.path, 'malcolm/malcolm_anger/malcolm_anger_1.png')
        path2 = os.path.join(self.path, 'malcolm/malcolm_anger/malcolm_anger_2.png')
        img1 = np.array(Image.open(path1).resize((self.img_size, self.img_size)))[..., :3]
        img2 = np.array(Image.open(path2).resize((self.img_size, self.img_size)))[..., :3]
        return np.array([img1, img2]), np.array([0, 0])

    def save_train_test_split(self, X_train, X_test, y_train, y_test):
        np.save('X_train_size' + self.datasetInfo + '.npy', X_train)
        np.save('X_test_size' + self.datasetInfo + '.npy', X_test)
        np.save('y_train_size' + self.datasetInfo + '.npy', y_train)
        np.save('y_test_size' + self.datasetInfo + '.npy', y_test)

    def load_train_test_split(self):
        X_train = np.load('X_train_size' + self.datasetInfo + '.npy')
        X_test = np.load('X_test_size' + self.datasetInfo + '.npy')
        y_train = np.load('y_train_size' + self.datasetInfo + '.npy')
        y_test = np.load('y_test_size' + self.datasetInfo + '.npy')
        return X_train, X_test, y_train, y_test

    def toOneHot(self, yData):
        return to_categorical(yData, num_classes=len(self.classes))

    def save_training_history(self, history):
        np.save('history' + self.datasetInfo + '.npy', history)

    def load_training_history(self):
        return np.load('history' + self.datasetInfo + '.npy', allow_pickle=True).item()

    def save_model(self, networkName, model):
        model.save('model' + networkName + self.datasetInfo)

    def load_model(self, networkName):
        model = keras.models.load_model('model' + networkName + self.datasetInfo)
        # model.load_weights('best' + networkName + self.datasetInfo + '.h5')
        return model
