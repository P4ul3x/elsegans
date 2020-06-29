# coding: utf-8

from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import initializers
from keras import backend as K
from functools import partial
from keras.layers.merge import _Merge

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import sys
import os
import time
import glob

import numpy as np
import pandas as pd
import configparser as cfgp

import math
import cv2
import h5py

from _scripts.utils import *

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    #BATCH_SIZE = 9

    def _merge_function(self, inputs):
        #https://stackoverflow.com/questions/59039886/get-randomly-weighted-averages-between-samples-in-a-batch-with-arbitrary-sample/59039887#59039887
        in_shape = K.shape(inputs[0])
        shape = K.concatenate([in_shape[0:1], K.ones_like(in_shape[1:], dtype='int32')], axis=0)
        alpha = K.random_uniform(shape)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
        #weights = K.random_uniform((RandomWeightedAverage.BATCH_SIZE, 1, 1, 1))
        #return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class WGAN():

    def __init__(self,
                 model_name,
                 path,
                 build_discriminator,
                 build_generator,
                 gradient_penalty_weight=10,
                 learn_rate = 0.0001,
                 img_rows=28,
                 img_cols=28,
                 channels=1,
                 latent_dim=100):

        #set model name
        self.model_name = model_name
        
        #set path
        self.path = path

        # Create directory for model
        try:
            path = os.path.join(self.path, self.model_name,"model")
            if not os.path.isdir(path):
                os.makedirs(path, mode=0o777)
            
            path = os.path.join(self.path, self.model_name,"images")
            if not os.path.isdir(path):
                os.makedirs(path, mode=0o777)
            
        
        except OSError:
            print("<Error> Could not create directory '%s' for new model" % path)
            return None

        # Input shape
        self.img_shape = (img_rows, img_cols, channels)
        self.latent_dim = latent_dim

        # Optimizer
        self.learn_rate = learn_rate
        optimizer = Adam(learn_rate, 0.5)

        # ---------------------
        # ---- Base Models ----
        # ---------------------

        self.generator = build_generator(self.latent_dim)
        self.discriminator = build_discriminator(self.img_shape)


        # ---------------------
        # Discriminator Trainer
        # ---------------------
        
        # The discriminator takes generated images as input and determines validity
        img = Input(shape=self.img_shape)
        valid = self.discriminator(img)

        g_img = Input(shape=self.img_shape)
        g_valid = self.discriminator(g_img)

        # The generator takes noise as input and generates imgs
        #z = Input(shape=(self.latent_dim,))
        #g_img = self.generator(z)
        #g_valid = self.discriminator(g_img)

        averaged_samples = RandomWeightedAverage()([img, g_img])
        averaged_samples_out = self.discriminator(averaged_samples)

        self.gradient_penalty_weight = gradient_penalty_weight
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=gradient_penalty_weight)
        # Functions need names or Keras will throw an error
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.t_discriminator= Model([img, g_img], [valid, g_valid, averaged_samples_out])
        self.t_discriminator.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                                     optimizer=optimizer)
        self.t_discriminator.summary()

        # ---------------------
        # Generator Trainer
        # ---------------------

        z = Input(shape=(self.latent_dim,))
        g_img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        valid = self.discriminator(g_img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=wasserstein_loss,
                              optimizer=optimizer)


    def train(self, x_train, epochs, batch_size=32, training_ratio=5, save_interval_batch=50, save_interval_epoch=100):

        # Adversarial ground truths
        positive_y = np.ones((batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

        # split dataset according to batch_size
        splits = x_train.shape[0]//batch_size
        split_idx = [(i+1)*batch_size for i in range(splits)]
        
        batches = np.split(x_train, split_idx)

        # get number of iterations
        nr_batches = len(batches)
        # if dataset_size//batch_size = 0
        if(batches[-1].shape[0] == 0): 
            nr_batches -= 1 
        
        # create dataframe to save statistics
        stats = pd.DataFrame(columns=['epoch','batch', 
                                      'batch_time', 'epoch_time', 'total_time',
                                      'batch_size',
                                      'd_ws_loss_real', 'd_ws_loss_fake',
                                      'd_ws_loss',
                                      'd_gp_loss',
                                      'g_loss'])
        stats_path = os.path.join(self.path, self.model_name,"stats.csv")
        stats_file = open(stats_path,"w")
        stats.to_csv(stats_file, index = False)

        # create noise for saved images
        r = 5
        c = 10
        save_noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        # Get starting time
        start_time = time.time()

        # Training
        for epoch in range(epochs):
               
            # Get epoch start time 
            start_epoch = time.time()
                
            for batch in range(nr_batches): 
                
                # Get batch start time 
                start_batch = time.time()
                
                # Last iteration may have a smaller size
                batch_size = batches[batch].shape[0]

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Sample noise and generate a batch of new images
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # Train the discriminator (real classified as ones)
                #d_loss_real = self.t_discriminator.train_on_batch(batches[batch], [positive_y[:batch_size], dummy_y[:batch_size]])
                
                # Train the discriminator (generated classified as zeros)
                #d_loss_fake = self.t_discriminator.train_on_batch(gen_imgs, [negative_y[:batch_size], dummy_y[:batch_size]])
                
                #Calculate combined discriminator losses
                #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                
                # Train the discriminator
                d_loss = self.t_discriminator.train_on_batch([batches[batch], gen_imgs], [positive_y[:batch_size], negative_y[:batch_size], dummy_y[:batch_size]])
                
                # ---------------------
                #  Train Generator
                # ---------------------

                # Train the generator (wants discriminator to mistake images as real)
                #g_loss = self.t_generator.train_on_batch(noise, positive_y[:batch_size])
                
                g_loss = self.combined.train_on_batch(noise, positive_y[:batch_size])
                

                # Time statistics
                end_batch = time.time()
                batch_time = end_batch - start_batch
                epoch_time = end_batch - start_epoch
                total_time = end_batch - start_time
                
                # Add stats to dataframe
                stats.loc[len(stats)] = [epoch, batch, 
                                         batch_time, epoch_time, total_time,
                                         batch_size,
                                         d_loss[0], d_loss[1],
                                         0.5 * np.add(d_loss[0], d_loss[1]),
                                         d_loss[2], 
                                         g_loss]
               
                # If at save interval or at end of epoch => save generated image samples
                if (batch % save_interval_batch == 0 or batch == nr_batches-1) and (epoch % save_interval_epoch == 0 or epoch == epochs-1):
                    self.generate_images_grid(save_noise,
                                              os.path.join(self.path,self.model_name,"images", "epoch_%d" % (epoch), "batch_%d.jpg" % (batch)))
            
            # Write statistics to file every 10 epochs
            if epoch % 100 == 0 or epoch == epochs-1:
                stats.to_csv(stats_file, index = False, header=False, float_format='%.4f')
                # Clean current statistics
                stats = stats.iloc[0:0]
            
            if epoch % save_interval_epoch == 0 or epoch == epochs-1:
                self.save_gan(epoch)
            
            # Progress
            t1 = time.time()
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f] in %04d secs" % (epoch, d_loss[0], 100*d_loss[1], g_loss, t1-start_time))
        
        # Close stats file
        stats_file.close()
        
        # Plot statistics
        
        
        #Total time
        finish_time = time.time()
        print("total training time: %04d min"%((finish_time-start_time)/60))

    def generate_images_grid(self, noise, save_path, scale_color=255):

        images = self.generator.predict(noise)
        images = (0.5 + images * 0.5)*scale_color
        images = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images])

        r,c = int_to_shape(images.shape[0])

        images = custom_reshape(images, (r,c,images.shape[1],images.shape[2],images.shape[3]))
        comb_image = cv2.vconcat([cv2.hconcat(image) for image in images])

        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), mode=0o777)
        cv2.imwrite(save_path, comb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    def save_gan(self, epoch=""):
        self.generator.save(str(os.path.join(self.path, self.model_name, "model", "generator"+("_"+str(epoch) if epoch != "" else "")+".h5")), include_optimizer=False)
        self.discriminator.save(str(os.path.join(self.path, self.model_name, "model", "discriminator"+("_"+str(epoch) if epoch != "" else "")+".h5")), include_optimizer=False)
        
    def save_configuration(self, epochs, batch_size, training_ratio, save_interval_batch, save_interval_epoch):
        parser = cfgp.ConfigParser()
        parser['Configuration'] = { 
                                    'model_name' : self.model_name,
                                    'path'       : str(self.path),
                                    'learn_rate' : str(self.learn_rate),
                                    'latent_dim' : str(self.latent_dim),
                                    'gradient_penalty_weight' : str(self.gradient_penalty_weight),
                                    'epochs'     : str(epochs),
                                    'batch_size' : str(batch_size),
                                    'training_ratio' : str(training_ratio),
                                    'save_interval_batch' : str(save_interval_batch),
                                    'save_interval_epoch' : str(save_interval_epoch),
                                    'img_rows' : str(self.img_shape[0]),
                                    'img_cols' : str(self.img_shape[1]),
                                    'channels' : str(self.img_shape[2])
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

            assert parser.has_section('Configuration'), "<Error> Configuration section not found on configuration file"

            model_name = configuration_parse_string(parser, 'model_name', None)
            assert model_name is not None, "<Error> Invalid 'model_name' in configuration file"
            path = configuration_parse_string(parser, 'path', None)
            assert path is not None, "<Error> Invalid 'path' in configuration file"

            epochs = configuration_parse_value(parser, int, 'epochs', 100)
            assert epochs is not None, "<Error> Invalid 'epochs' in configuration file"
            batch_size = configuration_parse_value(parser, int, 'batch_size', 32)
            assert batch_size is not None, "<Error> Invalid 'batch_size' in configuration file"
            gradient_penalty_weight = configuration_parse_value(parser, int, 'gradient_penalty_weight', 10)
            assert gradient_penalty_weight is not None, "<Error> Invalid 'gradient_penalty_weight' in configuration file"
            training_ratio = configuration_parse_value(parser, int, 'training_ratio', 5)
            assert training_ratio is not None, "<Error> Invalid 'training_ratio' in configuration file"
            latent_dim = configuration_parse_value(parser, int, 'latent_dim', 100)
            assert latent_dim is not None, "<Error> Invalid 'latent_dim' in configuration file"
            learn_rate = configuration_parse_value(parser, float, 'learn_rate', 0.0002)
            assert learn_rate is not None, "<Error> Invalid 'learn_rate' in configuration file"
            save_interval_batch = configuration_parse_value(parser, int, 'save_interval_batch', 50)
            assert save_interval_batch is not None, "<Error> Invalid 'save_interval_batch' in configuration file"
            save_interval_epoch = configuration_parse_value(parser, int, 'save_interval_epoch', 100)
            assert save_interval_epoch is not None, "<Error> Invalid 'save_interval_epoch' in configuration file"
            img_rows = configuration_parse_value(parser, int, 'img_rows', 28)
            assert img_rows is not None, "<Error> Invalid 'img_rows' in configuration file"
            img_cols = configuration_parse_value(parser, int, 'img_cols', 28)
            assert img_cols is not None, "<Error> Invalid 'img_cols' in configuration file"
            channels = configuration_parse_value(parser, int, 'channels', 1)
            assert channels is not None, "<Error> Invalid 'channels' in configuration file"

            return {
                    "model_name" : model_name,
                    "path" : path,
                    "gradient_penalty_weight" : gradient_penalty_weight,
                    "epochs" : epochs,
                    "batch_size" : batch_size,
                    "training_ratio" : training_ratio,
                    "latent_dim" : latent_dim,
                    "learn_rate" : learn_rate,
                    "save_interval_batch" : save_interval_batch,
                    "save_interval_epoch" : save_interval_epoch,
                    "img_rows": img_rows,
                    "img_cols": img_cols,
                    "channels": channels,
                    "use_best_fakes" : use_best_fakes
                   }    
    """
    @staticmethod
    def load_dcgan(model_path):

        try:           
            generator = load_model(os.path.join(model_path, "model", "generator.h5"))
            discriminator = load_model(os.path.join(model_path, "model", "discriminator.h5"))
            #combined = load_model(os.path.join(model_path, "model", "combined.h5"))
            combined = None
            
        except OSError:
            print("<Error> Model could not be loaded")
            return None

        path = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        
        config = DCGAN.configuration_parse(os.path.join(model_path, "config.cfg"))
        
        if config is None:
            print("Configuration not loaded")
            return None
        
        return DCGAN(learn_rate=config["learn_rate"], latent_dim=config["latent_dim"], 
                    model_name=model_name, class_n=config["class_n"], path=path,
                    img_rows=config["img_rows"], img_cols=config["img_cols"], channels=config["channels"],
                    generator=generator, discriminator=discriminator, combined=combined)
    """

def build_mnist_generator(latent_dim):

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    #model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def build_discriminator2(img_shape):

    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""
    model = Sequential()
    
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=img_shape))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal',
                            strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), kernel_initializer='he_normal', padding='same',
                            strides=[2, 2]))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))

    #model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def build_discriminator(img_shape):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    #model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

def build_hushem_generator(latent_dim):

    model = Sequential()

    model.add(Dense(256 * 33 * 33, activation="relu", input_dim=latent_dim))
    model.add(Reshape((33, 33, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    #model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def build_wikiart_generator(latent_dim):

    model = Sequential()

    model.add(Dense(256 * 32 * 32, activation="relu", input_dim=latent_dim))
    model.add(Reshape((32, 32, 256)))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    #model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    build_wikiart_generator(100)