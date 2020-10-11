import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Conv1D, dot, Dropout
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.initializers import glorot_uniform
import tensorflow as tf
from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
import pickle

seed = 2556
np.random.seed(seed)

train = 'Transfer_learning'  # '' for scratch and 'Transfer_learning'
file_path = "/home/UTHSCSA/mostavi/PycharmProjects/FSL_revise_third/02transfer/saved_weights_trans/"
if train == 'Transfer_learning':
    with open(file_path + 'MET_wght_pretrained16.pkl', "rb") as f:
        wghts_pretrained = pickle.load(f)


def initialize_weights(shape, name=None, dtype=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, name=None, dtype=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def last_layer(encoded_l, encoded_r, lyr_name='cos'):
    if lyr_name == 'L1':
        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors: tf.keras.backend.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        add_dens = Dense(512, activation='relu', bias_initializer=initialize_bias)(L1_distance)
        # drp_lyr = Dropout(0.25)(add_dens)
        # xx =  Dense(128, activation='relu', bias_initializer=initialize_bias)(add_dens)
        # prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(L1_distance)
        prediction = Dense(1, activation='sigmoid')(xx)

    elif lyr_name == 'L2':

        # Write L2 here
        L2_layer = Lambda(lambda tensors: (tensors[0] - tensors[1]) ** 2 / (tensors[0] + tensors[1]))
        L2_distance = L2_layer([encoded_l, encoded_r])
        add_dens = Dense(512, activation='relu', bias_initializer=initialize_bias)(L2_distance)
        drp_lyr = Dropout(0.25)(add_dens)
        # xx =  Dense(128, activation='relu', bias_initializer=initialize_bias)(drp_lyr)
        # drp_lyr2 = Dropout(0.25)(xx)
        # x =  Dense(64, activation='relu', bias_initializer=initialize_bias)(xx)
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(drp_lyr)

    else:

        # Add cosine similarity function
        cos_layer = Lambda(lambda tensors: K.sum(tensors[0] * tensors[1], axis=-1, keepdims=True) /
                                           tf.keras.backend.l2_normalize(tensors[0]) * tf.keras.backend.l2_normalize(
            tensors[1]))
        cos_distance = cos_layer([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(cos_distance)

    return prediction


def get_siamese_model(input_shape):
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network

    model = Sequential()

    model.add(
        Conv1D(filters=256, kernel_size=50, strides=50, activation='relu', weights=wghts_pretrained[0], padding='same',
               input_shape=input_shape))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(
        Conv1D(filters=128, kernel_size=10, strides=1, activation='relu', weights=wghts_pretrained[1], padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=5, strides=1, activation='sigmoid', weights=wghts_pretrained[2],
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())

    # model.add(Dense((512),activation='sigmoid',kernel_regularizer=l2(1e-3),
    #                 kernel_initializer=initialize_weights, bias_initializer=initialize_bias))
    # model.add(Dropout(0.25))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input],
                        outputs=last_layer(encoded_l, encoded_r, lyr_name='L2'))  ## prediction and cosine_similarity
    return siamese_net



