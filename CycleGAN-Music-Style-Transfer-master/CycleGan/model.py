from __future__ import division
import os
import time
from shutil import copyfile
from glob import glob
import tensorflow as tf
import numpy as np
# import config
from collections import namedtuple
# from module import *
# from utils import *
# from ops import *
# from metrics import *
import tensorflow_addons as tfa
import tensorflow.keras.backend as kb
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow import keras as keras
class Generator(object):
    def __init__(self,input_dim):
        self.input_dim = input_dim
        self.output_channel = self.input_dim[2]
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

        c1 = keras.layers.Conv2D(filters=64, kernel_size=7, strides=1,activation='relu', padding='VALID')(c0)
        c1 = tfa.layers.InstanceNormalization()(c1)
        # c1 is (# of images * 256 * 256 * 64)

        c2 = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding ='SAME')(c1)
        c2 = tfa.layers.InstanceNormalization()(c2)
        # c2 is (# of images * 128 * 128 * 128)

        c3 = keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='relu',padding ='SAME')(c2)
        c3 = tfa.layers.InstanceNormalization()(c3)
        # c3 is (# of images * 64 * 64 * 256)

        res = None
        # 10 resnet
        for i in range(num_res_net_blocks):
            res = residue_block(c3, 256, 3)

        #3 deconv
        d1 = keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2,activation='relu',padding = 'SAME',name = 'g_d1')(res)
        d1 = tfa.layers.InstanceNormalization()(d1)
        d2 = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='relu',padding = 'SAME',name = 'g_d2')(d1)
        d2 = tfa.layers.InstanceNormalization()(d2)

        d3 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT",name = 'g_d3')
        output = keras.layers.Conv2D(filters=self.output_channel, kernel_size=7, strides=1, activation='sigmoid')(d3)
        model = keras.Model(input,output)
        model.summary()

        return model

    def trainable(self,trainable = True):
        self.model.trainable = trainable

    def save(self,path):
        self.model.save(path)

    def load(self,path):
        self.model = keras.models.load_model(path)

    def train_on_batch(self, X, y):
        print("generator trainable weight",self.model.trainable_weights)
        return self.model.train_on_batch(X, y)
    def translate_domain(self,input_data):
        return self.model.predict(input_data)

class Discriminator(object):
    def __init__(self,input_dim):
        self.input_dim = input_dim
        self.output_channel = self.input_dim[2]

        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,padding = 'SAME',input_shape=self.input_dim))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=4, strides=2,padding = 'SAME'))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(tfa.layers.InstanceNormalization())
        model.add(keras.layers.Conv2D(filters=self.output_channel, kernel_size=1, strides=1))
        model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        model.summary()
        return model
    # def train(self):
    #
    def trainable(self,trainable = True):
        self.model.trainable = trainable

    def save(self,path):
        self.model.save(path)
    def load(self,path):
        keras.models.load_model(path)
    def predict(self,input_data):
        return self.model.predict(input_data)

    def train_on_batch(self,X,y):
        # print("discriminator trainable weight",len(self.model.trainable_weights))

        return self.model.train_on_batch(X,y)



class CycleGan(object):
    def __init__(self, sess, batch_size, crop_size,generatorAtoB,generatorBtoA, discriminatorA,discriminatorB, discriminatorAandM = None,discriminatorBandM = None,
                 generator_weight_factor = 10,discriminator_weight_factor = 10, use_D_M = False, gamma = 0.9,lamda = 0.9):
        self.sess = sess
        self.batch_size = batch_size
        self.image_size = crop_size  # cropped size
        # self.input_c_dim = input_channels  # number of input image channels
        # self.output_c_dim = output_channels  # number of output image channels
        self.generatorAToB = generatorAtoB
        self.generatorBToA = generatorBtoA
        self.discriminatorA = discriminatorA
        self.discriminatorB = discriminatorB
        self.use_D_M = use_D_M
        self.discriminatorAandM = discriminatorAandM
        self.discriminatorBandM = discriminatorBandM
        self.gamma = gamma
        self.lamda = lamda


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
        fake_B = generatorAtoB.model(input_real_A)
        dis_B = discriminatorB.model(fake_B)

        #compare A vs Fake A through A->B->A
        fake_A_ = generatorBtoA.model(fake_B)

        # compare B vs Fake B through B->A->B
        fake_A = generatorBtoA.model(input_real_B)
        fake_B_ = generatorAtoB.model(fake_A)

        # define model graph
        model = keras.Model([input_real_A, input_real_B], [dis_B, fake_A_, fake_B_])
        # define optimization algorithm configuration
        opt = Adam(lr=0.0002, beta_1=0.5)
        # compile model with weighting of least squares loss and L1 loss
        model.compile(loss=['mse', 'mae', 'mae'], loss_weights=[1, 10, 10], optimizer=opt)
        return model

    # def build_composite_discriminator_network(self, generatorAtoB, generatorBtoA, discriminatorB,discriminatorBandM):

    def generate_real_samples_batch(self,dataset,batch_size,output_shape,channel_shape):
        # choose random instances
        ix = np.random.randint(0, dataset.shape[0], batch_size)
        # retrieve selected images
        X = dataset[ix]
        # generate 'real' class labels (1)
        y = np.ones((batch_size, output_shape, output_shape, channel_shape))
        return X, y

    # generate a batch of images, returns images and targets
    def generate_fake_samples_batch(self,generator_model, dataset, output_shape,channel_shape):
        # generate fake instance
        X = generator_model.translate_domain(dataset)
        # create 'fake' class labels (0)
        y = np.zeros((len(X), output_shape, output_shape, channel_shape))
        return X, y

    def generate_mix_sample_batch(self,datasetA,datasetB,batch_size,output_shape,channel_shape):
        ix_A = np.random.randint(0, datasetA.shape[0], int(batch_size/2))
        X_A = datasetA[ix_A]
        ix_B = np.random.randint(0, datasetB.shape[0], int(batch_size /2))
        X_B = datasetA[ix_B]
        y = np.ones((batch_size, output_shape, output_shape, channel_shape))
        print  ( np.concatenate((X_A,X_B),axis=0).shape)

        return [np.concatenate((X_A,X_B),axis=0),y]

    # save the generator models to file
    def save_models(self,epoch, generator_model_AtoB, generator_model_BtoA,path):
        # save the first generator model
        folder = os.path.join(path,str(epoch))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filename1 = 'g_model_AtoB_.h5'
        filename1 = os.path.join(folder,filename1)
        generator_model_AtoB.save(filename1)
        # save the second generator model
        filename2 = 'g_model_BtoA_.h5'
        filename2 = os.path.join(folder, filename2)
        generator_model_BtoA.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))

    def load_models(self,path,filename1,filename2):
        filename1 = os.path.join(path, filename1)
        self.generatorAToB.load(filename1)
        filename2 = os.path.join(path, filename2)
        self.generatorBToA.load(filename2)

    # generate samples and save as a plot and save the model
    def summarize_performance(self,epochs, g_model, trainX, name, n_samples=5):
        # select a sample of input images
        X_in, _ = self.generate_real_samples_batch(trainX, n_samples, 0)
        # generate translated images
        X_out, _ = self.generate_fake_samples_batch(g_model, X_in, 0)
        # scale all pixels from [-1,1] to [0,1]
        X_in = (X_in + 1) / 2.0
        X_out = (X_out + 1) / 2.0
        # plot real images
        for i in range(n_samples):
            plt.subplot(2, n_samples, 1 + i)
            plt.axis('off')
            plt.imshow(X_in[i])
        # plot translated image
        for i in range(n_samples):
            plt.subplot(2, n_samples, 1 + n_samples + i)
            plt.axis('off')
            plt.imshow(X_out[i])
        # save plot to file
        filename1 = '%s_generated_plot_%06d.png' % (name, (step + 1))
        plt.savefig(filename1)
        plt.close()

    def update_image_pool(self,pool, images, max_size=50):
        selected = list()
        for image in images:
            if len(pool) < max_size:
                # stock the pool
                pool.append(image)
                selected.append(image)
            elif np.random.rand() < 0.5:
                # use image, but don't add it to the pool
                selected.append(image)
            else:
                # replace an existing image and use replaced image
                ix = np.random.randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return np.asarray(selected)

    def summarize(self, tag_value_pairs, step):
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def train(self,epochs,batches,datasetA,datasetB,save_path = "",summary_folder = "logs/scalars/",history_path = "history_log",model_save_frequenly = 10):
        # summary_folder = "logs/scalars/"
        self.writer = tf.summary.create_file_writer(summary_folder)

        #discriminator output square shape
        d_output_shape = self.discriminatorA.model.output_shape[1]
        channel_num =  self.discriminatorA.model.output_shape[3]
        data_pool_A,data_pool_B = list(),list()
        # batch_per_epoch = int(len(datasetA)/batches)
        # calculate the number of training iterations
        n_steps = int(len(datasetA)/batches)
        # manually enumerate

        list_epoch_g_loss = []
        list_epoch_d_loss = []
        list_epoch_g_A_to_B_loss = []
        list_epoch_g_B_to_A_loss = []
        list_epoch_d_A__loss = []
        list_epoch_d_B__loss = []
        list_epoch_d_A_M_loss = []
        list_epoch_d_B_M_loss = []

        for e in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_g_A_to_B_loss = 0
            epoch_g_B_to_A_loss = 0
            epoch_d_A__loss = 0
            epoch_d_B__loss = 0
            epoch_d_A_M_loss = 0
            epoch_d_B_M_loss = 0

            for i in range(n_steps):
                # select a batch of real samples
                X_realA, y_realA = self.generate_real_samples_batch(datasetA, batches, d_output_shape,channel_num)
                X_realB, y_realB = self.generate_real_samples_batch(datasetB, batches, d_output_shape,channel_num)
                # generate a batch of fake samples
                X_fakeA, y_fakeA = self.generate_fake_samples_batch(self.generatorBToA, X_realB, d_output_shape,channel_num)
                X_fakeB, y_fakeB = self.generate_fake_samples_batch(self.generatorAToB, X_realA, d_output_shape,channel_num)
                # update fakes from pool
                # X_fakeA = self.update_image_pool(data_pool_A, X_fakeA)
                # X_fakeB = self.update_image_pool(data_pool_B, X_fakeB)
                # update generator B->A via adversarial and cycle loss
                # print("step : ",i)
                # print("A->B trainable weight : ",len(self.generatorAToB.model.trainable_weights))
                # print("B->A trainable weight : ",len(self.generatorBToA.model.trainable_weights))

                g_loss2, cycle_loss_2, _,_ = self.composite_model_B.train_on_batch([X_realB, X_realA], [y_realA, X_realB, X_realA])
                self.generatorBToA.trainable(False)
                self.generatorAToB.trainable(True)
                # print("A->B trainable weight : ", len(self.generatorAToB.model.trainable_weights))
                # print("B->A trainable weight : ", len(self.generatorBToA.model.trainable_weights))
                # update discriminator for A -> [real/fake]
                self.discriminatorA.trainable(True)
                dA_loss1 = self.discriminatorA.train_on_batch(X_realA, y_realA)
                dA_loss2 = self.discriminatorA.train_on_batch(X_fakeA, y_fakeA)
                self.discriminatorA.trainable(False)

                dA_loss = (dA_loss1+dA_loss2)/2
                # update generator A->B via adversarial and cycle loss
                # print("A->B trainable weight : ", len(self.generatorAToB.model.trainable_weights))
                # print("B->A trainable weight : ", len(self.generatorBToA.model.trainable_weights))
                g_loss1, cycle_loss_1, _, _ = self.composite_model_A.train_on_batch([X_realA, X_realB], [y_realB, X_realA, X_realB])
                self.generatorBToA.trainable(True)
                self.generatorAToB.trainable(False)
                # print("A->B trainable weight : ", len(self.generatorAToB.model.trainable_weights))
                # print("B->A trainable weight : ", len(self.generatorBToA.model.trainable_weights))
                # update discriminator for B -> [real/fake]
                self.discriminatorB.trainable(True)
                dB_loss1 = self.discriminatorB.train_on_batch(X_realB, y_realB)
                dB_loss2 = self.discriminatorB.train_on_batch(X_fakeB, y_fakeB)
                self.discriminatorB.trainable(False)

                dB_loss = (dB_loss1+dB_loss2)/2

                epoch_g_A_to_B_loss += g_loss1
                epoch_g_B_to_A_loss += g_loss2
                epoch_d_A__loss += dA_loss
                epoch_d_B__loss += dB_loss
                epoch_g_loss = epoch_g_loss+g_loss1+g_loss2+self.lamda*(cycle_loss_1+cycle_loss_2)
                epoch_d_loss = epoch_d_loss+dA_loss+dB_loss



                if self.use_D_M:
                    # print("fake shahep :",X_fakeA.shape)
                    X_real_mix_M,y_real_mix_M = self.generate_mix_sample_batch(datasetA,datasetB,batches,d_output_shape,channel_num)

                    dAM_loss1 = self.discriminatorAandM.train_on_batch(X_real_mix_M, y_real_mix_M)
                    dAM_loss2 = self.discriminatorAandM.train_on_batch(X_fakeA, y_fakeA)
                    dAM_loss = (dAM_loss1+dAM_loss2)/2
                    dBM_loss1 = self.discriminatorBandM.train_on_batch(X_real_mix_M, y_real_mix_M)
                    dBM_loss2 = self.discriminatorBandM.train_on_batch(X_fakeB, y_fakeB)
                    dBM_loss = (dBM_loss1 + dBM_loss2) / 2
                    epoch_d_A_M_loss += dAM_loss
                    epoch_d_B_M_loss += dBM_loss
                    epoch_d_loss = epoch_d_loss +self.gamma*(dAM_loss+dBM_loss)

                # print("at epoch : ", e)
                # print("at step :", i)
                # print("generator loss : ", epoch_g_loss/(i+1))
                # print("discriminator loss : ", epoch_g_loss/(i+1))
                print('For epoch {} batch {}/{} , G_Loss is {:7.2f}, D_Loss is {:7.2f}.'.format(e,(i+1),n_steps,epoch_g_loss/(i+1),epoch_d_loss/(i+1)))

            #save model
            epoch_g_A_to_B_loss /= n_steps
            epoch_g_B_to_A_loss /= n_steps
            epoch_d_A__loss /= n_steps
            epoch_d_B__loss /= n_steps
            epoch_d_A_M_loss /= n_steps
            epoch_d_B_M_loss /= n_steps
            epoch_g_loss /= n_steps
            epoch_d_loss /= n_steps

            #summary log
            list_tag_values = {
                ("generatorA_to_B loss", epoch_g_A_to_B_loss),
                ("generatorB_to_A loss", epoch_g_B_to_A_loss),
                ("discriminator B loss", epoch_d_B__loss),
                ("discriminator A loss", epoch_d_A__loss),
                ("epoch discriminators loss", epoch_d_loss),
                ("epoch generators loss", epoch_g_loss)
            }
            self.summarize(list_tag_values, e)

            #save history data

            list_epoch_g_loss.append(epoch_g_loss)
            list_epoch_d_loss.append(epoch_d_loss)
            list_epoch_g_A_to_B_loss.append(epoch_g_A_to_B_loss)
            list_epoch_g_B_to_A_loss.append(epoch_g_B_to_A_loss)
            list_epoch_d_A__loss.append(epoch_d_A__loss)
            list_epoch_d_B__loss.append(epoch_d_B__loss)
            list_epoch_d_A_M_loss.append(epoch_d_A_M_loss)
            list_epoch_d_B_M_loss.append(epoch_d_B_M_loss)
            #
            # print("at epoch : ",e)
            # print("generator loss : ",epoch_g_loss)
            # print("discriminator loss : ", epoch_g_loss)

            if (e+1)%model_save_frequenly == 0:
                self.save_models(e,self.generatorAToB,self.generatorBToA,save_path)
                history = {
                    "generatorA_to_B loss":list_epoch_g_A_to_B_loss,
                    "generatorB_to_A loss":list_epoch_g_B_to_A_loss,
                    "discriminator B loss": list_epoch_d_B__loss,
                    "discriminator A loss":list_epoch_d_A__loss,
                    "discriminator B_M loss":list_epoch_d_B_M_loss,
                    "discriminator A_M loss": list_epoch_d_A_M_loss,
                    "epoch discriminators loss": list_epoch_d_loss,
                    "epoch generators loss": list_epoch_g_loss
                }
                history_table = pd.DataFrame(history)

                history_table.to_csv(os.path.join(history_path,"history_record.csv"),index=False)




    def translate_A_to_B(self,inputs):
        return self.generatorAToB.translate_domain(inputs)

    def translate_B_to_A(self,inputs):
        return self.generatorBToA.translate_domain(inputs)


    # def load_models(self,path):
    #     self.generatorAToB.load(path)
    #     self.generatorBToA.load(path)
    #     self.discriminatorA.load(path)
    #     self.discriminatorB.load(path)
    #
    # def save_model(self,path):
    #     self.generatorAToB.save(path)
    #     self.generatorBToA.save(path)
    #     self.discriminatorA.save(path)
    #     self.discriminatorB.save(path)

    #
    #
    # def test(self):
    #
    # def generate_result(self,domain):

# def generate_result():

class StyleClassifier:
    def __init__(self,input_dim,lr = 0.0002,beta_1 = 0.5,loss_weight = 0.5):
        self.input_dim = input_dim
        self.output_channel = self.input_dim[2]
        self.lr = lr
        self.beta_1 = beta_1
        self.loss_weight = loss_weight

        self.model = self.build_model()

    def build_model(self):
        input = keras.Input(shape=(self.input_dim))

        c1 = keras.layers.Conv2D(filters=64, kernel_size=[1,12], strides=[1,12])(input)
        c1 = keras.layers.LeakyReLU(alpha=0.3)(c1)

        c2 = keras.layers.Conv2D(filters=128, kernel_size=[4, 1], strides=[4, 1])(c1)
        c2 = keras.layers.LeakyReLU(alpha=0.3)(c2)
        c2 = tfa.layers.InstanceNormalization()(c2)

        c3 = keras.layers.Conv2D(filters=256, kernel_size=[2, 1], strides=[2, 1])(c2)
        c3 = keras.layers.LeakyReLU(alpha=0.3)(c3)
        c3 = tfa.layers.InstanceNormalization()(c3)

        c4 = keras.layers.Conv2D(filters=512, kernel_size=[8, 1], strides=[8, 1])(c3)
        c4 = keras.layers.LeakyReLU(alpha=0.3)(c4)
        c4 = tfa.layers.InstanceNormalization()(c4)

        c5 = keras.layers.Conv2D(filters=2, kernel_size=[1, 7], strides=[1, 7])(c4)
        c5 = keras.layers.LeakyReLU(alpha=0.3)(c5)
        c5 = tfa.layers.InstanceNormalization()(c5)

        c6 = keras.layers.Flatten()(c5)
        c6 = keras.layers.Dense(2,activation = 'softmax')(c6)

        model = keras.Model(input,c6)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr, beta_1=self.beta_1), loss_weights=[self.loss_weight])
        model.summary()
        return model
        # def train(self):
        #

    def trainable(self, trainable=True):
        self.model.trainable = trainable

    def save(self, path):
        model_name = 'classifier_model.h5'
        path = os.path.join(path,model_name)
        self.model.save(path)

    def load(self, path):
        model_name = 'classifier_model.h5'
        path = os.path.join(path, model_name)
        self.model = keras.models.load_model(path)

    def predict(self, input_data):
        return self.model.predict(input_data)

    def train(self,train_X,train_Y,test_X,test_Y,batch,num_epochs):
        return self.model.fit(train_X, train_Y, batch_size=batch, epochs=num_epochs, verbose=1,validation_split=0.2)

    # def test(self,testA,testB):




def main():
    # directory_path_A = '../Input_Data_Small_Test/datasetA/npy/'
    # directory_path_B= '../Input_Data_Small_Test/datasetB/npy/'
    directory_path_A = 'dataset2/pia2vio_example/trainA/'
    directory_path_B = 'dataset2/pia2vio_example/trainB/'


    # directory_path_B = '../dataset/CP_P/test/'

    datasetA = []
    datasetB = []

    normalizeA = 14.0
    normalizeB = 15.0

    # print(np_vars.shape)
    max = 0
    min = 270
    for files in os.listdir(directory_path_A):
        path = os.path.join(directory_path_A,files)
        data = np.load(path)
        data = data[:,:,0:256]
        #only use first channel
        data = data[0:1, :, :]
        data = data.reshape((256, 256, 1))
        #shrink data to 256
        max = np.maximum(max,np.max(data))
        min = np.minimum(max,np.min(data))
        datasetA.append(data)

    print("current ata Max ", max)
    print("current data min : ", min)
    max = 0
    min = 270
    for files in os.listdir(directory_path_B):
        path = os.path.join(directory_path_B,files)
        data = np.load(path)
        data = data[:,:,0:256]
        data = data[0:1,:,:]
        data = data.reshape((256,256,1))
        # print(data)
        # print("fresh data sbape : ",data.shape)
        # print("max : ",np.amax(data))
        # print("min : ",np.min(data))
        # print("process data sbape : ", data.shape)
        max = np.maximum(max, np.max(data))
        min = np.minimum(max, np.min(data))
        datasetB.append(data)

    print("current ata Max ",max)
    print("current data min : ",min)

    datasetA = np.array(datasetA)/normalizeA
    datasetB = np.array(datasetB)/normalizeB

    datasetA = datasetA
    datasetB = datasetB

    print(datasetA.shape)
    print(np.max(datasetA))
    print(np.min(datasetA))
    print(datasetB.shape)
    print(np.max(datasetB))
    print(np.min(datasetB))

    image_dim = datasetA[0].shape

    Generator_A_to_B = Generator(input_dim=image_dim)
    Generator_B_to_A = Generator(input_dim=image_dim)
    Discriminator_B = Discriminator(input_dim=image_dim)
    Discriminator_A = Discriminator(input_dim=image_dim)
    Discriminator_B_M = Discriminator(input_dim=image_dim)
    Discriminator_A_M = Discriminator(input_dim=image_dim)
    use_D_M = True
    batch_size = 24



    # with tf.compat.v1.Session() as sess:
    gan = CycleGan(None,batch_size,image_dim,Generator_A_to_B,Generator_B_to_A,Discriminator_A,Discriminator_B,
                   use_D_M=use_D_M,discriminatorAandM=Discriminator_A_M,discriminatorBandM=Discriminator_B_M)

    Classifier = StyleClassifier(input_dim=image_dim)
    save_path = "classifier_model/model_1"
    Classifier.load(save_path)
    epoch = 30
    model_version = "model_4"
    save_model_path = "save_model"
    save_model_path = os.path.join(model_version,save_model_path)
    log_path = "logs/scalars/"
    log_path = os.path.join(model_version,log_path)
    history_path = "history_log"
    history_path = os.path.join(model_version,history_path)
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(history_path):
        os.makedirs(history_path)

    save_model_frequenly = 5
    # gan.train(epoch,batch_size,datasetA,datasetB,save_model_path,log_path,history_path,save_model_frequenly)

    #test the model
    path = "model_3/save_model/19/"
    generator_A_to_B_path = "g_model_AtoB_.h5"
    generator_B_to_A_path = "g_model_BtoA_.h5"
    gan.load_models(path,generator_A_to_B_path,generator_B_to_A_path)
    directory_path_A_test = 'dataset2/pia2vio_example/testA/'
    directory_path_B_test = 'dataset2/pia2vio_example/testB/'
    test_A = []
    test_B = []
    test_A_to_B_result_file_name = []
    test_B_to_A_result_file_name = []
    result_path = 'result/model_3'
    path_A_to_B = os.path.join(result_path,'A_to_B')
    path_B_to_A = os.path.join(result_path,'B_to_A')
    if not os.path.isdir(path_A_to_B):
        os.makedirs(path_A_to_B)
    if not os.path.isdir(path_B_to_A):
        os.makedirs(path_B_to_A)
    max = 0
    min = 270
    for files in os.listdir(directory_path_A_test):
        path = os.path.join(directory_path_A_test,files)
        data = np.load(path)
        data = data[:,:,0:256]
        data = data[0:1, :, :]
        data = data.reshape((256, 256, 1))
        max = np.maximum(max, np.max(data))
        min = np.minimum(max, np.min(data))
        test_A.append(data)
        newfile = "convert_"+files
        output_path = os.path.join(path_A_to_B, newfile)
        test_A_to_B_result_file_name.append(output_path)

    print("current ata Max ", max)
    print("current data min : ", min)
    max = 0
    min = 270
    for files in os.listdir(directory_path_B_test):
        path = os.path.join(directory_path_B_test,files)
        data = np.load(path)
        data = data[:,:,0:256]
        data = data[0:1, :, :]
        data = data.reshape((256, 256, 1))
        max = np.maximum(max, np.max(data))
        min = np.minimum(max, np.min(data))
        test_B.append(data)
        newfile = "convert_" + files
        output_path = os.path.join(path_B_to_A,newfile)
        test_B_to_A_result_file_name.append(output_path)
    print("current ata Max ", max)
    print("current data min : ", min)
    # #
    testA = np.array(test_A)/normalizeA
    testB = np.array(test_B)/normalizeA
    # #
    result_A_to_B = gan.translate_A_to_B(testA)
    result_B_to_A = gan.translate_B_to_A(testB)
    #
    result_A_to_B = result_A_to_B*normalizeA
    result_B_to_A = result_B_to_A*normalizeA
    #
    # testA = testA*normalize_metric
    # testB = testB*normalize_metric
    #
    #
    for i_A in range(len(test_A_to_B_result_file_name)):
        current = result_A_to_B[i_A]
        current = current.reshape((256,256))
        print("test shape : ",current.shape)
        np.save(test_A_to_B_result_file_name[i_A],current)
    for i_B in range(len(test_B_to_A_result_file_name)):
        current = result_B_to_A[i_B]
        current = current.reshape((256,256))
        print(current.shape)
        np.save(test_B_to_A_result_file_name[i_B],current)

    # sample_A = testA[0].reshape(256,256,4)
    # A_to_B = result_A_to_B[0].reshape(256,256,4)
    #
    # sample_A = sample_A*255.0
    # A_to_B = A_to_B*255.0
    #
    # print(np.max(sample_A[:,:,0]))
    # print(np.max(A_to_B[:, :, 0]))
    #
    # new_im = Image.fromarray((sample_A[:,:,0]).astype(np.float32)).convert('RGB')
    # new_im.save("numpy_altered_sampleA.png")
    # new_im = Image.fromarray((A_to_B[:,:,0]).astype(np.float32)).convert('RGB')
    # new_im.save("numpy_altered_A_to_B.png")




    # sample_A = datasetA[0]
    # generate_B = gan.translate_A_to_B(sample_A.reshape(1,sample_A.shape[0],sample_A.shape[1],sample_A.shape[2])).reshape(sample_A.shape)
    #
    # sample_B = datasetB[0]
    # generate_A = gan.translate_B_to_A(sample_B.reshape(1,sample_B.shape[0],sample_B.shape[1],sample_B.shape[2])).reshape(sample_B.shape)

    # max =  np.max(sample_B)
    # min = np.min(sample_B)
    # for s in datasetB:
    #
    #     max = np.maximum(np.max(s),max)
    #     min = np.minimum(np.min(s),min)
    # for s in datasetA:
    #
    #     max = np.maximum(np.max(s),max)
    #     min = np.minimum(np.min(s),min)
    #
    #
    # print("max : ",max)
    # print("min : ",min)
    # print(np.max(sample))
    # print(np.min(sample))
    # print(np.max(generate_B))
    # print(np.min(generate_B))
    # plt.imshow(sample)
    # plt.show()
    #
    # print(generate_B.shape)
    # plt.imshow(generate_B)
    # plt.show()
    # print(generate_A.shape)
    # plt.imshow(generate_A)
    # plt.show()
    # print(sample_A)
    # print(generate_B)
    # # sample_A = (sample_A/28)
    #
    # new_im = Image.fromarray((sample_A*255).astype(np.uint8)).convert('RGB')
    # new_im.save("numpy_altered_sample1.png")
    # new_im = Image.fromarray((generate_B * 255).astype(np.uint8)).convert('RGB')
    # new_im.save("numpy_altered_sample2.png")
    # new_im = Image.fromarray((sample_B * 255).astype(np.uint8)).convert('RGB')
    # new_im.save("numpy_altered_sample3.png")




# main()

def train_classifier():
    directory_path_A = 'dataset2/pia2vio_example/trainA/'
    directory_path_B = 'dataset2/pia2vio_example/trainB/'
    save_path = "classifier_model/model_1"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    datasetA = []
    datasetB = []
    normalizeA = 14.0
    normalizeB = 15.0
    for files in os.listdir(directory_path_A):
        path = os.path.join(directory_path_A, files)
        data = np.load(path)
        data = data[:, :, 0:256]
        # only use first channel
        data = data[0:1, :, :]
        data = data.reshape((256, 256, 1))
        # shrink data to 256
        datasetA.append(data)

    for files in os.listdir(directory_path_B):
        path = os.path.join(directory_path_B, files)
        data = np.load(path)
        data = data[:, :, 0:256]
        data = data[0:1, :, :]
        data = data.reshape((256, 256, 1))
        datasetB.append(data)

    datasetA = np.array(datasetA) / normalizeA
    datasetB = np.array(datasetB) / normalizeB

    datasetA = datasetA
    image_dim = datasetA[0].shape
    Y_A = np.zeros((len(datasetA),2))
    Y_A[:,1] = np.ones(len(datasetA))
    print(Y_A.shape)
    print(Y_A[3])
    datasetB = datasetB
    Y_B = np.zeros((len(datasetA),2))
    Y_B[:,0] = np.ones(len(datasetA))
    print(datasetA.shape)
    print(datasetB.shape)

    X = np.concatenate((datasetA,datasetB),axis=0)
    Y = np.concatenate((Y_A,Y_B),axis=0)
    print(X.shape)
    print(Y.shape)
    from sklearn.model_selection import train_test_split
    train_X,test_X,train_Y,test_Y = train_test_split(X, Y, test_size = 0.05, random_state = 42)
    Classifier = StyleClassifier(input_dim=image_dim)
    batch = 32
    epoch = 50
    # history = Classifier.train(train_X,train_Y,test_X,test_Y,num_epochs=epoch,batch=batch)
    # print(history['val'])
    # Classifier.save(save_path)
    Classifier.load(save_path)
    results = Classifier.predict(train_X)
    results[results > 0.5] = 1
    results[results <= 0.5] = 0
    result_vector = np.argmax(results,axis=1)
    target_vector = np.argmax(train_Y,axis=1)
    # print(np.argmax(results,axis=1)[:10])
    # print(results[:20])
    # print(train_Y[:20])
    print(np.mean(result_vector==target_vector))

# train_classifier()

def convert_hot_encode_to_vector(results):
    results[results > 0.5] = 1
    results[results <= 0.5] = 0
    result_vector = np.argmax(results, axis=1)
    return result_vector


def test_classifier():
    normalizeA = 14.0
    normalizeB = 15.0
    image_dim = (256,256,1)

    Generator_A_to_B = Generator(input_dim=image_dim)
    Generator_B_to_A = Generator(input_dim=image_dim)
    Discriminator_B = Discriminator(input_dim=image_dim)
    Discriminator_A = Discriminator(input_dim=image_dim)
    Discriminator_B_M = Discriminator(input_dim=image_dim)
    Discriminator_A_M = Discriminator(input_dim=image_dim)
    use_D_M = True
    batch_size = 24
    gan = CycleGan(None, batch_size, image_dim, Generator_A_to_B, Generator_B_to_A, Discriminator_A, Discriminator_B,
                   use_D_M=use_D_M, discriminatorAandM=Discriminator_A_M, discriminatorBandM=Discriminator_B_M)

    path = "model_2/save_model/19/"
    generator_A_to_B_path = "g_model_AtoB_.h5"
    generator_B_to_A_path = "g_model_BtoA_.h5"
    gan.load_models(path, generator_A_to_B_path, generator_B_to_A_path)
    # directory_path_A_test = 'dataset2/pia2vio_example/testA/'
    # directory_path_B_test = 'dataset2/pia2vio_example/testB/'
    directory_path_A_test = 'dataset2/pia2vio_example/trainA/'
    directory_path_B_test = 'dataset2/pia2vio_example/trainB/'
    test_A = []
    test_B = []
    test_A_to_B_result_file_name = []
    test_B_to_A_result_file_name = []
    result_path = 'result/model_3'
    path_A_to_B = os.path.join(result_path, 'A_to_B')
    path_B_to_A = os.path.join(result_path, 'B_to_A')
    if not os.path.isdir(path_A_to_B):
        os.makedirs(path_A_to_B)
    if not os.path.isdir(path_B_to_A):
        os.makedirs(path_B_to_A)
    for files in os.listdir(directory_path_A_test):
        path = os.path.join(directory_path_A_test, files)
        data = np.load(path)
        data = data[:, :, 0:256]
        data = data[0:1, :, :]
        data = data.reshape((256, 256, 1))
        test_A.append(data)
        newfile = "convert_" + files
        output_path = os.path.join(path_A_to_B, newfile)
        test_A_to_B_result_file_name.append(output_path)
    for files in os.listdir(directory_path_B_test):
        path = os.path.join(directory_path_B_test, files)
        data = np.load(path)
        data = data[:, :, 0:256]
        data = data[0:1, :, :]
        data = data.reshape((256, 256, 1))
        test_B.append(data)
        newfile = "convert_" + files
        output_path = os.path.join(path_B_to_A, newfile)
        test_B_to_A_result_file_name.append(output_path)

    testA = np.array(test_A) / normalizeA
    testB = np.array(test_B) / normalizeA
    result_A_to_B = gan.translate_A_to_B(testA)
    result_B_to_A = gan.translate_A_to_B(testB)
    result_A_to_B = result_A_to_B * normalizeA
    result_B_to_A = result_B_to_A * normalizeA
    testA = testA*normalizeA
    testB = testB*normalizeA


    #classifier
    save_path = "classifier_model/model_1"
    Classifier = StyleClassifier(input_dim=image_dim)
    Classifier.load(save_path)
    Y_testA = np.zeros((len(testA), 2))
    Y_testA[:, 1] = np.ones(len(testA))
    # print(Y_testA.shape)
    # print(Y_testA[3])
    Y_testB = np.zeros((len(testB), 2))
    Y_testB[:, 0] = np.ones(len(testB))

    Y_result_A_to_B = np.zeros((len(result_A_to_B), 2))
    Y_result_A_to_B[:, 0] = np.ones(len(result_A_to_B))

    Y_result_B_to_A = np.zeros((len(result_B_to_A), 2))
    Y_result_B_to_A[:, 1] = np.ones(len(result_B_to_A))


    predict_testA =Classifier.predict(testA)
    predict_testB = Classifier.predict(testB)
    predict_result_A_to_B = Classifier.predict(result_A_to_B)
    predict_result_B_to_A = Classifier.predict(result_B_to_A)

    predict_testA =convert_hot_encode_to_vector(predict_testA)
    predict_testB = convert_hot_encode_to_vector(predict_testB)
    predict_result_A_to_B = convert_hot_encode_to_vector(predict_result_A_to_B)
    predict_result_B_to_A = convert_hot_encode_to_vector(predict_result_B_to_A)

    Y_testA = convert_hot_encode_to_vector(Y_testA)
    Y_testB = convert_hot_encode_to_vector(Y_testB)
    Y_result_A_to_B = convert_hot_encode_to_vector(Y_result_A_to_B)
    Y_result_B_to_A = convert_hot_encode_to_vector(Y_result_B_to_A)

    print(np.mean(Y_testA == predict_testA))
    print(np.mean(Y_testB == predict_testB))

    print(np.mean(Y_result_A_to_B == predict_result_A_to_B))
    print(np.mean(Y_result_B_to_A == predict_result_B_to_A))

# test_classifier()
