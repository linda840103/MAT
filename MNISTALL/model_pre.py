#!/usr/bin/env python

"""
The main file for pre-training models to predict

"""

import sys

import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pylab import *

# keras
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense
import keras.optimizers


# visualisation
#
from keras.datasets import mnist
from keras.utils import np_utils

import mnist_nn as NN

# training the model from data
# or read the model from saved data file
# then start analyse the model

whichMode = "train"
 
def model_pre():
    
    # construct model according to the flage
    # whichMode == "read" read from saved file 
    # whichMode == "train" training from the beginning 
    if whichMode == "train": 
    
        (X_train, Y_train, X_test, Y_test, batch_size, nb_epoch) = NN.read_dataset()
        
        #print X_train.shape, Y_train.shape, Y_train[0]

        print ("Building network model ......")
        model_p = NN.build_model()

        start_time = time.time()
        model_p.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(X_test, Y_test))
        #model.fit(X_train_transferability, Y_train_transferability, batch_size=batch_size, nb_epoch=nb_epoch_transferability,
        #          verbose=1, validation_data=(X_test, Y_test))
        score = model_p.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        print("Fitting time: --- %s seconds ---" % (time.time() - start_time))   
        print("Training finished!")
      

    return (model_p)


        
"""
   validate the model by the test data from the package
""" 
def test(model):

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_test = X_test.astype('float32')
    X_test = X_test.astype('float32')
    X_test /= 255

    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print("Start testing model ... ")
    # prediction after training
    start_time = time.time()
    y_predicted = model.predict(X_test)
    print (y_predicted)

    print("Testing time: --- %s seconds ---" % (time.time() - start_time))

