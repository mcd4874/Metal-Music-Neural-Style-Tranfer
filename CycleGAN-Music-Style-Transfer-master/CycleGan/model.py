from __future__ import division
import os
import time
from shutil import copyfile
from glob import glob
import tensorflow as tf
import numpy as np
# import config
from collections import namedtuple
from module import *
from utils import *
from ops import *
# from metrics import *
import tensorflow_addons as tfa
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']


from tensorflow import keras as keras
class Generator(object):
    def __init__(self,input_dim,output_dim,filter_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.filter_dim = filter_dim
        self.model = self.build_model()

    def build_model(self,num_res_net_blocks = 10):
        def residue_block(input_data, filters, conv_size,strides=1):
            p = int((conv_size - 1) / 2)
            x = tf.pad(input_data, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            x = keras.layers.Conv2D(filters=filters, kernel_size=conv_size, strides=strides,activation='relu', padding='VALID')(x)
            x = tfa.layers.InstanceNormalization()(x)
            x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            x = keras.layers.Conv2D(filters, conv_size, activation='relu', padding='VALID')(x)
            x = tfa.layers.InstanceNormalization()(x)
            x = keras.layers.Add()([x, input_data])
            x = keras.layers.Activation('relu')(x)
            return x

        # 3 conv
        input = keras.Input(shape=(self.input_dim))
        c0 = tf.pad(input, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        # c0 is (# of images * 262 * 262 * 3)
        # 10 resnet
        c1 = keras.layers.Conv2D(filters=64, kernel_size=7, strides=1,activation='relu', padding='VALID')(c0)
        c1 = tfa.layers.InstanceNormalization()(c1)
        c2 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='VALID')(c1)
        c2 = tfa.layers.InstanceNormalization()(c2)
        c3 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='VALID')(c2)
        c3 = tfa.layers.InstanceNormalization()(c3)
        res = None
        for i in range(num_res_net_blocks):
            res = residue_block(c3, 256, 3)

        #3 deconv
        d1 = keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2,activation='relu')(res)
        d1 = tfa.layers.InstanceNormalization()(d1)
        d2 = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='relu')(d1)
        d2 = tfa.layers.InstanceNormalization()(d2)
        output = keras.layers.Conv2DTranspose(filters=1, kernel_size=7, strides=1, activation='sigmoid',padding='VALID')(d2)
        model = keras.Model(input,output)
        model.summary()

        return model



    # def train(self):
    #
    def predict(self,input_data):
        return self.model.predict(input_data)

class Discriminator(object):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,input_shape=(self.input_dim[0], self.input_dim[1], 1)))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=4, strides=2))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(tfa.layers.InstanceNormalization())
        model.add(keras.layers.Conv2D(filters=1, kernel_size=1, strides=1))
        model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        model.summary()
        return model
    # def train(self):
    #
    def predict(self,input_data):
        return self.model.predict(input_data)



class cyclegan(object):
    def __init__(self, sess, batch_size, crop_size, input_channels, output_channels, generatorAtoB,generatorBtoA, discriminator, discriminatorM):
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = crop_size  # cropped size
        self.input_c_dim = input_channels  # number of input image channels
        self.output_c_dim = output_channels  # number of output image channels
        self.generatorRealToFake = generatorAtoB
        self.generatorFakeToReal = generatorBtoA
        self.discriminator = discriminator
        self.discriminatorM = discriminatorM

        self.build_network()

    def build_network(self):
        # ensure the model we're updating is trainable
        self.generatorRealToFake.trainable()
        # mark discriminator as not trainable
        self.generatorFakeToReal.trainable(False)
        # mark other generator model as not trainable
        self.discriminator.trainable(False)
        # discriminator element
        input_gen = keras.Input(shape=self.image_size)
        gen1_out = self.generatorRealToFake(input_gen)
        output_d = self.discriminator(gen1_out)
        # identity element
        input_id = keras.Input(shape=self.image_size)
        output_id = self.generatorRealToFake(input_id)
        # forward cycle
        output_f = self.generatorFakeToReal(gen1_out)
        # backward cycle
        gen2_out = self.generatorFakeToReal(input_id)
        output_b = self.generatorRealToFake(gen2_out)
        # define model graph
        model = keras.Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
        # define optimization algorithm configuration
        opt = Adam(lr=0.0002, beta_1=0.5)
        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
        return model




    def train(self):


    def test(self):

    def generate_result(self,domain):