from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.preprocessing import image
import keras.backend as back
from keras.callbacks import CSVLogger, EarlyStopping
from keras.utils import to_categorical

import itertools

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, recall_score , precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score 
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.utils.multiclass import unique_labels

import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import sys
import os
import time
import glob
import csv

import numpy as np
import pandas as pd
import configparser as cfgp

import math
import cv2
import h5py

import functools

from _scripts.utils import *
#from metrics import *


class Classifier:
    
    def __init__(self,
                 model_name,
                 path,
                 nr_classes,
                 learn_rate = 0.0002,
                 img_rows=28, img_cols=28, channels=1,
                 model=None):

        #set model name
        self.model_name = model_name

        #set path
        self.path = path

        # Create directory for model
        if self.model_name:
            try:
                path = os.path.join(self.path, self.model_name,"model")
                if not os.path.isdir(path):
                    os.makedirs(path, mode=0o777)
            
            except OSError:
                print("<Error> Could not create directory '%s' for new model" % path)
                return None        

        # Input shape
        self.img_shape = (img_rows, img_cols, channels)
        
        # classes
        self.nr_classes = nr_classes

        self.learn_rate = learn_rate
        self.optimizer = Adam(learn_rate, 0.5)

        # Build and compile the discriminator

        self.model = KerasClassifier(build_fn=self.build)

    def build(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(self.nr_classes, activation='sigmoid'))

        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        model = Model(img, validity)

        model.compile(loss='binary_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy']) #,'precision_macro','recall_macro','f1_macro','roc_auc_macro', 'average_precision'

        return model
    
    def train(self, x_train, y_train, x_test, y_test, epochs, batch_size=32, nr_folds=5, shuffle=False, callbacks=[]):

        y_train_categorical = to_categorical(y_train, self.nr_classes)
        y_test_categorical = to_categorical(y_test, self.nr_classes)

        stats = pd.DataFrame(columns=['val_acc', 'test_acc',
                                      'val_precision', 'test_precision',
                                      'val_recall', 'test_recall',
                                      'val_f1', 'test_f1',
                                      'val_roc_auc', 'test_roc_auc',
                                      'val_avg_precision', 'test_avg_precision'])

        if (nr_folds <= 1):
          kfold = StratifiedKFold(n_splits=5, shuffle=shuffle)
          folder = [list(kfold.split(x_train, y_train))[0]]

        else:
          kfold = StratifiedKFold(n_splits=nr_folds, shuffle=shuffle)
          folder = list(kfold.split(x_train, y_train))

        y_train_cat = to_categorical(y_train, self.nr_classes)
        y_test_cat = to_categorical(y_test, self.nr_classes)

        history=[]
        confusion=[]
        (model_to_save, save_score) = (None, 0)

        for train_idx, val_idx in folder:

            h = self.model.fit(x=x_train[train_idx],
                               y=y_train_cat[train_idx], 
                               validation_data=(x_train[val_idx], y_train_cat[val_idx]),
                               epochs=epochs,
                               batch_size=batch_size,
                               callbacks=callbacks,
                               verbose=1)
            history.append(h)

            # integer metrics
            val_pred = self.model.predict(x_train[val_idx])
            test_pred = self.model.predict(x_test)

            confusion.append(confusion_matrix(y_test, test_pred))

            val_acc = accuracy_score(y_train[val_idx], val_pred)
            test_acc = accuracy_score(y_test, test_pred)

            val_precision = precision_score(y_train[val_idx], val_pred, average='macro')
            test_precision = precision_score(y_test, test_pred, average='macro')

            val_recall = recall_score(y_train[val_idx], val_pred, average='macro')
            test_recall = recall_score(y_test, test_pred, average='macro')

            val_f1 = f1_score(y_train[val_idx], val_pred, average='macro')
            test_f1 = f1_score(y_test, test_pred, average='macro')

            # categorical metrics
            val_pred_cat = to_categorical(val_pred, self.nr_classes)
            test_pred_cat = to_categorical(test_pred, self.nr_classes)

            val_roc_auc = roc_auc_score(y_train_cat[val_idx], val_pred_cat, average='macro')
            test_roc_auc = roc_auc_score(y_test_cat, test_pred_cat, average='macro')

            val_avg_precision = average_precision_score(y_train_cat[val_idx], val_pred_cat, average='macro')
            test_avg_precision = average_precision_score(y_test_cat, test_pred_cat, average='macro')

            if test_f1 > save_score: 
              (model_to_save, save_score) = (self.model.model, test_f1)

            stats.loc[len(stats)] = [val_acc, test_acc,
                                     val_precision, test_precision,
                                     val_recall, test_recall,
                                     val_f1, test_f1,
                                     val_roc_auc, test_roc_auc,
                                     val_avg_precision, test_avg_precision]

        # confusion matrix
        confusion = np.array(confusion).sum(axis=0)
        np.save(os.path.join(self.path, self.model_name,"confusion_matrix.npy"), confusion)

        # train evolution statistics
        train_stats = {}
        for key in history[0].history.keys():
            key_stats = [fold.history[key] for fold in history]
            key_stats = np.array(key_stats)
            train_stats[key+"_mean"] = np.mean(key_stats, axis=0)
            train_stats[key+"_std"] = np.std(key_stats, axis=0)

        # save each fold individually
        stats_path = os.path.join(self.path, self.model_name,"train_stats.csv")
        with open(stats_path,"w") as stats_file:
            pd.DataFrame(data=train_stats).to_csv(stats_file, index = False, float_format='%.6f')

        # save summary for each metric
        mean = stats.mean()
        std = stats.std()
        stats_path = os.path.join(self.path, self.model_name,"test_stats_summary.csv")
        with open(stats_path, "w") as stats_file:
            stats_summary = pd.DataFrame(columns=['acc_val_mean',           'acc_val_std',           'acc_test_mean',           'acc_test_std',
                                                  'precision_val_mean',     'precision_val_std',     'precision_test_mean',     'precision_test_std',
                                                  'recall_val_mean',        'recall_val_std',        'recall_test_mean',        'recall_test_std',
                                                  'f1_val_mean',            'f1_val_std',            'f1_test_mean',            'f1_test_std',
                                                  'roc_auc_val_mean',       'roc_auc_val_std',       'roc_auc_test_mean',       'roc_auc_test_std',
                                                  'avg_precision_val_mean', 'avg_precision_val_std', 'avg_precision_test_mean', 'avg_precision_test_std'])
            stats_summary.loc[len(stats_summary)] = [mean["val_acc"],           std["val_acc"],           mean["test_acc"],           std["test_acc"],
                                                     mean["val_precision"],     std["val_precision"],     mean["test_precision"],     std["test_precision"],
                                                     mean["val_recall"],        std["val_recall"],        mean["test_recall"],        std["test_recall"],
                                                     mean["val_f1"],            std["val_f1"],            mean["test_f1"],            std["test_f1"],
                                                     mean["val_roc_auc"],       std["val_roc_auc"],       mean["test_roc_auc"],       std["test_roc_auc"],
                                                     mean["val_avg_precision"], std["val_avg_precision"], mean["test_avg_precision"], std["test_avg_precision"]]
            stats_summary.to_csv(stats_file, index = False, float_format='%.6f')

        save_destination = str(os.path.join(self.path, self.model_name, "model", "classifier.h5"))
        model_to_save.save(save_destination)

    def train_part(self, x_train, y_train, x_test, y_test, x_sup, y_sup, warmup, epochs_per_part, parts, images_per_part, batch_size=32, nr_folds=5, shuffle=False, callbacks=[]):

        y_train_categorical = to_categorical(y_train, self.nr_classes)
        y_test_categorical = to_categorical(y_test, self.nr_classes)

        stats = pd.DataFrame(columns=['val_acc', 'test_acc',
                                      'val_precision', 'test_precision',
                                      'val_recall', 'test_recall',
                                      'val_f1', 'test_f1',
                                      'val_roc_auc', 'test_roc_auc',
                                      'val_avg_precision', 'test_avg_precision'])

        if (nr_folds <= 1):
          kfold = StratifiedKFold(n_splits=5, shuffle=shuffle)
          folder = [list(kfold.split(x_train, y_train))[0]]

        else:
          kfold = StratifiedKFold(n_splits=nr_folds, shuffle=shuffle)
          folder = list(kfold.split(x_train, y_train))

        part_split_fold = StratifiedShuffleSplit(n_splits=parts, train_size=images_per_part)
        part_folder = list(part_split_fold.split(x_sup, y_sup))

        y_train_cat = to_categorical(y_train, self.nr_classes)
        y_test_cat = to_categorical(y_test, self.nr_classes)
        y_sup_cat = to_categorical(y_sup, self.nr_classes)

        history=[]
        confusion=[]
        (model_to_save, save_score) = (None, 0)
        
        for train_idx, val_idx in folder:
            
            h = self.model.fit(x=x_train[train_idx],
                               y=y_train_cat[train_idx], 
                               validation_data=(x_train[val_idx], y_train_cat[val_idx]),
                               epochs=warmup,
                               batch_size=batch_size,
                               callbacks=callbacks,
                               verbose=1)

            p=0
            for part_idx, _ in part_folder:

                part_h = self.model.model.fit(x=np.concatenate((x_train[train_idx], x_sup[part_idx]), axis=0),
                                         y=np.concatenate((y_train_cat[train_idx], y_sup_cat[part_idx]), axis=0),
                                         validation_data=(x_train[val_idx], y_train_cat[val_idx]),
                                         initial_epoch=warmup+p*epochs_per_part,
                                         epochs=warmup+(p+1)*epochs_per_part,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         callbacks=callbacks,
                                         verbose=1) 

                if len(h.history.keys()) > 0:
                  for key in h.history.keys():
                      h.history[key] = h.history[key]+part_h.history[key]
                else:
                  h = part_h
                p+=1

            history.append(h)

            # integer metrics
            val_pred = self.model.predict(x_train[val_idx])
            test_pred = self.model.predict(x_test)

            confusion.append(confusion_matrix(y_test, test_pred))

            val_acc = accuracy_score(y_train[val_idx], val_pred)
            test_acc = accuracy_score(y_test, test_pred)

            val_precision = precision_score(y_train[val_idx], val_pred, average='macro')
            test_precision = precision_score(y_test, test_pred, average='macro')

            val_recall = recall_score(y_train[val_idx], val_pred, average='macro')
            test_recall = recall_score(y_test, test_pred, average='macro')

            val_f1 = f1_score(y_train[val_idx], val_pred, average='macro')
            test_f1 = f1_score(y_test, test_pred, average='macro')

            # categorical metrics
            val_pred_cat = to_categorical(val_pred, self.nr_classes)
            test_pred_cat = to_categorical(test_pred, self.nr_classes)

            val_roc_auc = roc_auc_score(y_train_cat[val_idx], val_pred_cat, average='macro')
            test_roc_auc = roc_auc_score(y_test_cat, test_pred_cat, average='macro')

            val_avg_precision = average_precision_score(y_train_cat[val_idx], val_pred_cat, average='macro')
            test_avg_precision = average_precision_score(y_test_cat, test_pred_cat, average='macro')

            if test_f1 > save_score: 
              (model_to_save, save_score) = (self.model.model, test_f1)

            stats.loc[len(stats)] = [val_acc, test_acc,
                                     val_precision, test_precision,
                                     val_recall, test_recall,
                                     val_f1, test_f1,
                                     val_roc_auc, test_roc_auc,
                                     val_avg_precision, test_avg_precision]

        # confusion matrix
        confusion = np.array(confusion).sum(axis=0)
        np.save(os.path.join(self.path, self.model_name,"confusion_matrix.npy"), confusion)

        # train evolution statistics
        train_stats = {}
        for key in history[0].history.keys():
            key_stats = [fold.history[key] for fold in history]
            key_stats = np.array(key_stats)
            train_stats[key+"_mean"] = np.mean(key_stats, axis=0)
            train_stats[key+"_std"] = np.std(key_stats, axis=0)

        # save each fold individually
        stats_path = os.path.join(self.path, self.model_name,"train_stats.csv")
        with open(stats_path,"w") as stats_file:
            pd.DataFrame(data=train_stats).to_csv(stats_file, index = False, float_format='%.6f')

        # save summary for each metric
        mean = stats.mean()
        std = stats.std()
        stats_path = os.path.join(self.path, self.model_name,"test_stats_summary.csv")
        with open(stats_path, "w") as stats_file:
            stats_summary = pd.DataFrame(columns=['acc_val_mean',           'acc_val_std',           'acc_test_mean',           'acc_test_std',
                                                  'precision_val_mean',     'precision_val_std',     'precision_test_mean',     'precision_test_std',
                                                  'recall_val_mean',        'recall_val_std',        'recall_test_mean',        'recall_test_std',
                                                  'f1_val_mean',            'f1_val_std',            'f1_test_mean',            'f1_test_std',
                                                  'roc_auc_val_mean',       'roc_auc_val_std',       'roc_auc_test_mean',       'roc_auc_test_std',
                                                  'avg_precision_val_mean', 'avg_precision_val_std', 'avg_precision_test_mean', 'avg_precision_test_std'])
            stats_summary.loc[len(stats_summary)] = [mean["val_acc"],           std["val_acc"],           mean["test_acc"],           std["test_acc"],
                                                     mean["val_precision"],     std["val_precision"],     mean["test_precision"],     std["test_precision"],
                                                     mean["val_recall"],        std["val_recall"],        mean["test_recall"],        std["test_recall"],
                                                     mean["val_f1"],            std["val_f1"],            mean["test_f1"],            std["test_f1"],
                                                     mean["val_roc_auc"],       std["val_roc_auc"],       mean["test_roc_auc"],       std["test_roc_auc"],
                                                     mean["val_avg_precision"], std["val_avg_precision"], mean["test_avg_precision"], std["test_avg_precision"]]
            stats_summary.to_csv(stats_file, index = False, float_format='%.6f')

        save_destination = str(os.path.join(self.path, self.model_name, "model", "classifier.h5"))
        model_to_save.save(save_destination)

    def save_model(self):
        dst = str(os.path.join(self.path, self.model_name, "model", "classifier.h5"))
        self.model.model.save(dst)

    @staticmethod
    def load_classifier(model_path):
        try:           
            model = load_model(os.path.join(model_path, "model", "classifier.h5"))
            config = Discriminator.configuration_parse(os.path.join(model_path, "config.cfg"))
        except OSError:
            print("<Error> Model could not be loaded")
            return None

        path = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        
        return Discriminator(path=path,
                             model_name=model_name,
                             nr_classes=config["nr_classes"],
                             learn_rate=config["learn_rate"],
                             img_rows=config["img_rows"],
                             img_cols=config["img_cols"],
                             channels=config["channels"],
                             model=model)

    def save_configuration(self, epochs, batch_size, shuffle, nr_folds):
        parser = cfgp.ConfigParser()
        parser['Configuration'] = { 
                                    'model_name' : self.model_name,
                                    'path'       : str(self.path),
                                    'learn_rate' : str(self.learn_rate),
                                    'nr_classes' : str(self.nr_classes),
                                    'img_rows' : str(self.img_shape[0]),
                                    'img_cols' : str(self.img_shape[1]),
                                    'channels' : str(self.img_shape[2]),
                                    'epochs'     : str(epochs),
                                    'batch_size' : str(batch_size),
                                    'shuffle': str(shuffle),
                                    'nr_folds': str(nr_folds)
                                  }
        
        with open(os.path.join(self.path, self.model_name, "config.cfg"), "w") as f:
            parser.write(f)

    @staticmethod
    def configuration_parse(filename):
        
        parser = cfgp.ConfigParser()
    
        with open(filename,"r") as f:

            try:
                parser.read_file(f)
            except cfg.ParsingError:
                print("<Error> File could not be parsed")
                return None

            assert parser.has_section('Configuration'), "<Error> Configuration section not found"

            model_name = configuration_parse_string(parser, 'model_name', None)
            path = configuration_parse_string(parser, 'path', 'Default')

            epochs = configuration_parse_value(parser, int, 'epochs', 100)
            if epochs is None: return
            batch_size = configuration_parse_value(parser, int, 'batch_size', 32)
            if batch_size is None: return
            learn_rate = configuration_parse_value(parser, float, 'learn_rate', 0.0002)
            if learn_rate is None: return
            img_rows = configuration_parse_value(parser, int, 'img_rows', 28)
            if img_rows is None: return
            img_cols = configuration_parse_value(parser, int, 'img_cols', 28)
            if img_cols is None: return
            channels = configuration_parse_value(parser, int, 'channels', 1)
            if channels is None: return
            shuffle = configuration_parse_value(parser, bool, 'shuffle', True)
            if shuffle is None: return
            nr_folds = configuration_parse_value(parser, int, 'nr_folds', 5)
            if shuffle is None: return

            return {"model_name" : model_name,
                    "path" : path,
                    "epochs" : epochs,
                    "batch_size" : batch_size,
                    "learn_rate" : learn_rate,
                    "img_rows": img_rows,
                    "img_cols": img_cols,
                    "channels": channels,
                    "shuffle": shuffle,
                    "nr_folds": nr_folds
                    }