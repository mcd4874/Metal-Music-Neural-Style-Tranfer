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

from model import Generator,Discriminator,CycleGan,StyleClassifier


def translate_results(gan,datasetA,datasetB,normalize_A = None, normalize_B = None):
    if normalize_A is None:
        normalize_A = int(np.max(datasetA))+1
    if normalize_B is None:
        normalize_B = int(np.max(datasetB))+1
    datasetA = datasetA/normalize_A
    datasetB = datasetB/normalize_B
    result_A_to_B = gan.translate_A_to_B(datasetA)
    result_B_to_A = gan.translate_B_to_A(datasetB)
    result_A_to_B = result_A_to_B * normalize_B
    result_B_to_A = result_B_to_A * normalize_A
    return [result_A_to_B,result_B_to_A]

def generate_classifer_result(classifier,datasetA,datasetB,normalize_A = None, normalize_B = None):
    if normalize_A is None:
        normalize_A = int(np.max(datasetA))+1
    if normalize_B is None:
        normalize_B = int(np.max(datasetB))+1
    datasetA = datasetA/normalize_A
    datasetB = datasetB/normalize_B

    result_A = classifier.predict(datasetA)
    result_B = classifier.predict(datasetB)

    return [result_A,result_B]

def convert_classifier_to_encode(result_A,result_B,threshold = 0.5):
    result_A[result_A>threshold] = 1
    result_A[result_A <= threshold] = 0

    result_B[result_B > threshold] = 1
    result_B[result_B <= threshold] = 0

    result_vector_A = np.argmax(result_A, axis=1)
    result_vector_B = np.argmax(result_B, axis=1)

    return [result_vector_A,result_vector_B]

def calculate_source_strength(gan,classifier,datasetA,datasetB,normalize_A = None, normalize_B = None):
    if normalize_A is None:
        normalize_A = int(np.max(datasetA))+1
    if normalize_B is None:
        normalize_B = int(np.max(datasetB))+1

    C_A,C_B = generate_classifer_result(classifier,datasetA,datasetB,normalize_A,normalize_B)

    A_B,B_A = translate_results(gan,datasetA,datasetB,normalize_A,normalize_B)

    C_B_A,C_A_B = generate_classifer_result(classifier,B_A,A_B,normalize_A,normalize_B)

    B_A_B,A_B_A = translate_results(gan,B_A,A_B,normalize_A,normalize_B)

    C_A_B_A,C_B_A_B = generate_classifer_result(classifier,A_B_A,B_A_B,normalize_A,normalize_B)


    S_A = C_A[:,1]

    S_B = C_B[:,0]

    S_A_B = C_A[:,1]  - C_A_B[:,1]

    S_B_A = C_B[:,0] -  C_B_A[:,0]

    S_A_B_A = (C_A[:,1] - C_A_B[:,1] + C_A_B_A[:,1] - C_A_B[:,1])/2

    S_B_A_B = (C_B[:,0] -  C_B_A[:,0] + C_B_A_B[:,0] -  C_B_A[:,0])/2


    return [np.mean(S_A),np.mean(S_B), np.mean(S_A_B) , np.mean(S_B_A) , np.mean(S_A_B_A), np.mean(S_B_A_B)]
