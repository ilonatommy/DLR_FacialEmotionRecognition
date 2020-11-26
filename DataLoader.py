import os

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

    def __get_labels(self, images):
        labels = [None] * sum([len(emotionImages) for emotionImages in images])
        offset = 0
        for i in range(len(self.classes)):
            size = len(images[i])
            end = size + offset
            labels[offset: end] = [i] * size
            offset = size
        return labels

    def load(self):
        images = [None] * len(self.classes)
        for character_dir in glob.glob(os.path.join(self.path, "*", "")):
            for emotion_dir in glob.glob(os.path.join(character_dir, "*", "")):
                emotion_dir_name = os.path.basename(os.path.normpath(emotion_dir))
                emotion_name = emotion_dir_name.split("_")[1]
                imgs = []
                for img_name in glob.glob(os.path.join(emotion_dir, "*" + self.extension)):
                    img = Image.open(img_name).resize((self.img_size, self.img_size))
                    # removing the 4th dim which is transparency
                    imgs.append(np.array(img)[..., :3])
                if images[self.classes[emotion_name]]:
                    images[self.classes[emotion_name]].append(imgs)
                else:
                    images[self.classes[emotion_name]] = imgs
            if sum([len(emotionImages) for emotionImages in images]) > self.limit:
                break
        return images, self.__get_labels(images)

    def demo_load(self):
        path1 = os.path.join(self.path, 'malcolm/malcolm_anger/malcolm_anger_1.png')
        path2 = os.path.join(self.path, 'malcolm/malcolm_anger/malcolm_anger_2.png')
        img1 = np.array(Image.open(path1).resize((self.img_size, self.img_size)))[..., :3]
        img2 = np.array(Image.open(path2).resize((self.img_size, self.img_size)))[..., :3]
        return [img1, img2], [0, 0]
