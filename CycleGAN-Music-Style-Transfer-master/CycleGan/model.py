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
from tensorflow import keras.backend as kb
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

    def trainable(self,trainable = True):
        self.model.trainable = trainable

    def save(self,path):
        self.model.save(path)
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
    def trainable(self,trainable = True):
        self.model.trainable = trainable

    def save(self,path):
        self.model.save(path)
    def predict(self,input_data):
        return self.model.predict(input_data)



class cyclegan(object):
    def __init__(self, sess, batch_size, crop_size, input_channels, output_channels ,generatorAtoB,generatorBtoA, discriminatorA,discriminatorB, discriminatorAandM,discriminatorBandM,
                 generator_weight_factor = 10,discriminator_weight_factor = 10, use_D_M = True):
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = crop_size  # cropped size
        self.input_c_dim = input_channels  # number of input image channels
        self.output_c_dim = output_channels  # number of output image channels
        self.generatorAToB = generatorAtoB
        self.generatorBToA = generatorBtoA
        self.discriminatorA = discriminatorA
        self.discriminatorB = discriminatorB
        self.use_D_M = use_D_M
        self.discriminatorAandM = discriminatorAandM
        self.discriminatorBandM = discriminatorBandM
        self.composite_model_A = self.build_composite_generator_network(self.generatorAToB,self.generatorBToA,self.discriminatorA)
        self.composite_model_B = self.build_composite_generator_network(self.generatorBToA,self.generatorAToB,self.discriminatorB)



    def build_composite_generator_network(self,generatorAtoB,generatorBtoA,discriminatorB):
        # ensure the model we're updating is trainable
        generatorAtoB.trainable()
        # mark discriminator as not trainable
        generatorBtoA.trainable(False)
        # mark other generator model as not trainable
        discriminatorB.trainable(False)

        input_real_A = keras.Input(shape=self.image_size)
        input_real_B = keras.Input(shape=self.image_size)

        # discriminator element
        fake_B = generatorAtoB(input_real_A)
        dis_B = discriminatorB(fake_B)

        #compare A vs Fake A through A->B->A
        fake_A_ = generatorBtoA(fake_B)

        # compare B vs Fake B through B->A->B
        fake_A = generatorBtoA(input_real_B)
        fake_B_ = generatorAtoB(fake_A)

        # define model graph
        model = keras.Model([input_real_A, input_real_B], [dis_B, fake_A_, fake_B_])
        # define optimization algorithm configuration
        opt = Adam(lr=0.0002, beta_1=0.5)
        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['mse', 'mae', 'mae'], loss_weights=[1, 10, 10], optimizer=opt)
        return model

    # def build_composite_discriminator_network(self, generatorAtoB, generatorBtoA, discriminatorB,discriminatorBandM):

    def generate_real_samples_batch(self,dataset,batch_size,output_shape):
        # choose random instances
        ix = np.random.randint(0, dataset.shape[0], batch_size)
        # retrieve selected images
        X = dataset[ix]
        # generate 'real' class labels (1)
        y = np.ones((batch_size, output_shape, output_shape, 1))
        return X, y

    # generate a batch of images, returns images and targets
    def generate_fake_samples_batch(self,generator_model, dataset, output_shape):
        # generate fake instance
        X = generator_model.predict(dataset)
        # create 'fake' class labels (0)
        y = np.zeros((len(X), output_shape, output_shape, 1))
        return X, y

    def generate_mix_sample_batch(self):

    # save the generator models to file
    def save_models(self,step, generator_model_AtoB, generator_model_BtoA,path):
        # save the first generator model
        filename1 = 'g_model_AtoB_%06d.h5' % (step + 1)
        filename1 = os.path.join(path,filename1)
        generator_model_AtoB.save(filename1)
        # save the second generator model
        filename2 = 'g_model_BtoA_%06d.h5' % (step + 1)
        filename2 = os.path.join(path, filename2)
        generator_model_BtoA.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))

    # generate samples and save as a plot and save the model
    def summarize_performance(self,step, g_model, trainX, name, n_samples=5):
        # select a sample of input images
        X_in, _ = self.generate_real_samples_batch(trainX, n_samples, 0)
        # generate translated images
        X_out, _ = self.generate_fake_samples_batch(g_model, X_in, 0)
        # scale all pixels from [-1,1] to [0,1]
        X_in = (X_in + 1) / 2.0
        X_out = (X_out + 1) / 2.0
        # plot real images
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + i)
            pyplot.axis('off')
            pyplot.imshow(X_in[i])
        # plot translated image
        for i in range(n_samples):
            pyplot.subplot(2, n_samples, 1 + n_samples + i)
            pyplot.axis('off')
            pyplot.imshow(X_out[i])
        # save plot to file
        filename1 = '%s_generated_plot_%06d.png' % (name, (step + 1))
        pyplot.savefig(filename1)
        pyplot.close()

    def update_image_pool(self,pool, images, max_size=50):
        selected = list()
        for image in images:
            if len(pool) < max_size:
                # stock the pool
                pool.append(image)
                selected.append(image)
            elif random() < 0.5:
                # use image, but don't add it to the pool
                selected.append(image)
            else:
                # replace an existing image and use replaced image
                ix = randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return asarray(selected)


    def train(self,epochs,batches,datasetA,datasetB):
        #discriminator output square shape
        d_output_shape = self.discriminatorA.output_shape[1]
        data_pool_A,data_pool_B = list(),list()
        batch_per_epoch = int(len(datasetA)/batches)
        # calculate the number of training iterations
        n_steps = batch_per_epoch * epochs
        # manually enumerate
        epoch_g_loss = 0
        epoch_d_loss = 0
        for i in range(n_steps):
            # select a batch of real samples
            X_realA, y_realA = self.generate_real_samples_batch(datasetA, batches, d_output_shape)
            X_realB, y_realB = self.generate_real_samples_batch(datasetB, batches, d_output_shape)
            # generate a batch of fake samples
            X_fakeA, y_fakeA = self.generate_fake_samples_batch(self.generatorBToA, X_realB, d_output_shape)
            X_fakeB, y_fakeB = self.generate_fake_samples_batch(self.generatorAToB, X_realA, d_output_shape)
            # update fakes from pool
            X_fakeA = self.update_image_pool(data_pool_A, X_fakeA)
            X_fakeB = self.update_image_pool(data_pool_B, X_fakeB)
            # update generator B->A via adversarial and cycle loss
            g_loss2, _, _, _, _ = self.composite_model_B.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
            # update discriminator for A -> [real/fake]
            dA_loss1 = self.discriminatorA.train_on_batch(X_realA, y_realA)
            dA_loss2 = self.discriminatorA.train_on_batch(X_fakeA, y_fakeA)
            dA_loss = (dA_loss1+dA_loss2)/2
            # update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _, _ = self.composite_model_A.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
            # update discriminator for B -> [real/fake]
            dB_loss1 = self.discriminatorB.train_on_batch(X_realB, y_realB)
            dB_loss2 = self.discriminatorB.train_on_batch(X_fakeB, y_fakeB)
            dB_loss = (dB_loss1+dB_loss2)/2


            if self.use_D_M:
                dAM_loss1 = self.discriminatorAandM.train_on_batch(X_real_mix_M, y_real_mix_M)
                dAM_loss2 = self.discriminatorAandM.train_on_batch(X_fakeA, y_fakeA)
                dAM_loss = (dAM_loss1+dAM_loss2)/2
                dBM_loss1 = self.discriminatorBandM.train_on_batch(X_real_mix_M, y_real_mix_M)
                dBM_loss2 = self.discriminatorBandM.train_on_batch(X_fakeB, y_fakeB)
                dBM_loss = (dBM_loss1 + dBM_loss2) / 2

            epoch_g_loss = epoch_g_loss
            epoch_d_loss = epoch_d_loss+dA_loss+dB_loss


        # summarize performance



    def test(self):

    def generate_result(self,domain):