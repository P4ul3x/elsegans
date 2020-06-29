from keras.datasets import mnist, fashion_mnist, cifar10
from keras.preprocessing import image

import numpy as np
import cv2
import glob

from .utils import shuffle_in_unison

def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    return X_train, Y_train, X_test, Y_test

def load_fashion_mnist():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    return X_train, Y_train, X_test, Y_test

def load_cifar10():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    return X_train, Y_train, X_test, Y_test

def local_dataset(paths, shape):
    if (not isinstance(paths, list) and not isinstance(paths, tuple)):
        paths = [paths]

    X = []
    Y = []

    # each path corresponds to a single class
    for label in range(len(paths)):
        images = glob.glob(paths[label])
        images.sort()

        for img_name in images:
            if shape[2] == 1:
                img = cv2.imread(img_name, 0)
            else: #IMG_SHAPE[2] == 3
                img = cv2.imread(img_name, 1)
            img = cv2.resize(img, shape[:2], interpolation=cv2.INTER_AREA)
            
            X.append(image.img_to_array(img))
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y, dtype=int)

    return X,Y