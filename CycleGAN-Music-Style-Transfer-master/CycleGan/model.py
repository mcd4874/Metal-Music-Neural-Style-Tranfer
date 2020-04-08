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
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']


from tensorflow import keras as keras
class Generator(object):
    def __init__(self,input_dim):
        self.input_dim = input_dim
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
        output = keras.layers.Conv2D(filters=1, kernel_size=7, strides=1, activation='sigmoid')(d3)
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
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(filters=64, kernel_size=4, strides=2,padding = 'SAME',input_shape=self.input_dim))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(keras.layers.Conv2D(filters=256, kernel_size=4, strides=2,padding = 'SAME'))
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
        X = generator_model.translate_domain(dataset)
        # create 'fake' class labels (0)
        y = np.zeros((len(X), output_shape, output_shape, 1))
        return X, y

    def generate_mix_sample_batch(self,datasetA,datasetB,batch_size,output_shape):
        ix_A = np.random.randint(0, datasetA.shape[0], batch_size/2)
        X_A = datasetA[ix_A]

        ix_B = np.random.randint(0, datasetB.shape[0], batch_size /2)
        X_B = datasetA[ix_B]

        # X =

    # save the generator models to file
    def save_models(self,epoch, generator_model_AtoB, generator_model_BtoA,path):
        # save the first generator model
        filename1 = 'g_model_AtoB_%06d.h5' % (epoch)
        filename1 = os.path.join(path,filename1)
        generator_model_AtoB.save(filename1)
        # save the second generator model
        filename2 = 'g_model_BtoA_%06d.h5' % (epoch)
        filename2 = os.path.join(path, filename2)
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

    def train(self,epochs,batches,datasetA,datasetB,save_path = "",model_save_frequenly = 10):
        summary_folder = "logs/scalars/"
        self.writer = tf.summary.create_file_writer(summary_folder)

        #discriminator output square shape
        d_output_shape = self.discriminatorA.model.output_shape[1]
        data_pool_A,data_pool_B = list(),list()
        # batch_per_epoch = int(len(datasetA)/batches)
        # calculate the number of training iterations
        n_steps = int(len(datasetA)/batches)
        # manually enumerate

        for e in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_g_A_to_B_loss = 0
            epoch_g_B_to_A_loss = 0
            epoch_d_A__loss = 0
            epoch_d_B__loss = 0
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
                    X_real_mix_M,y_real_mix_M = self.generate_mix_sample_batch(datasetA,datasetB,batches,d_output_shape)

                    dAM_loss1 = self.discriminatorAandM.train_on_batch(X_real_mix_M, y_real_mix_M)
                    dAM_loss2 = self.discriminatorAandM.train_on_batch(X_fakeA, y_fakeA)
                    dAM_loss = (dAM_loss1+dAM_loss2)/2
                    dBM_loss1 = self.discriminatorBandM.train_on_batch(X_real_mix_M, y_real_mix_M)
                    dBM_loss2 = self.discriminatorBandM.train_on_batch(X_fakeB, y_fakeB)
                    dBM_loss = (dBM_loss1 + dBM_loss2) / 2
                    epoch_d_loss = epoch_d_loss +self.gamma*(dAM_loss+dBM_loss)



            #save model
            epoch_g_A_to_B_loss /= n_steps
            epoch_g_B_to_A_loss /= n_steps
            epoch_d_A__loss /= n_steps
            epoch_d_B__loss /= n_steps
            epoch_g_loss /= n_steps
            epoch_d_loss /= n_steps

            print("at epoch : ",e)
            print("generator loss : ",epoch_g_loss)
            print("discriminator loss : ", epoch_g_loss)
            list_tag_values = {
                ("generatorA_to_B loss",epoch_g_A_to_B_loss),
                ("generatorB_to_A loss",epoch_g_B_to_A_loss),
                ("discriminator B loss", epoch_d_B__loss),
                ("discriminator A loss", epoch_d_A__loss),
                ("epoch discriminators loss", epoch_d_loss),
                ("epoch generators loss", epoch_g_loss)
            }
            self.summarize(list_tag_values,e)
            if (e)%model_save_frequenly == 0:
                self.save_models(e,self.generatorAToB,self.generatorBToA,save_path)
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

def main():
    directory_path_A = '../Input_Data_Small_Test/datasetA/npy/'
    directory_path_B= '../Input_Data_Small_Test/datasetB/npy/'
    datasetA = []
    datasetB = []
    file_types = ['npy']
    file = '2000_Sabaton_Fist_For_Fight_01_Introduction_0001.npy'
    edge = int((302-256)/2)
    # print(np_vars.shape)
    for files in os.listdir(directory_path_A):
        path = os.path.join(directory_path_A,files)
        data = np.load(path)
        data = data.reshape(data.shape[::-1])
        #shrink data to 256
        data = data[edge:302-edge,:,:]
        datasetA.append(data)
    for files in os.listdir(directory_path_B):
        path = os.path.join(directory_path_B,files)
        data = np.load(path)
        data = data.reshape(data.shape[::-1])
        data = data[edge:302 - edge, :, :]
        datasetB.append(data)
    datasetA = np.array(datasetA)
    datasetB = np.array(datasetB)
    print(datasetA.shape)
    print(datasetB.shape)

    image_dim = datasetA[0].shape

    Generator_A_to_B = Generator(input_dim=image_dim)
    Generator_B_to_A = Generator(input_dim=image_dim)
    Discriminator_B = Discriminator(input_dim=image_dim)
    Discriminator_A = Discriminator(input_dim=image_dim)
    # Discriminator_B_M = Discriminator(input_dim=image_dim)
    # Discriminator_A_M = Discriminator(input_dim=image_dim)

    batch_size = 32



    # with tf.compat.v1.Session() as sess:
    gan = CycleGan(None,batch_size,image_dim,Generator_A_to_B,Generator_B_to_A,Discriminator_A,Discriminator_B)
    epoch = 20
    path = "/save_model"
    save_model_frequenly = 5
    gan.train(epoch,batch_size,datasetA,datasetB,path,save_model_frequenly)

    #test the model


main()

