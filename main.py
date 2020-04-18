#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:17:17 2020

@author: ronjaronnback

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import spectrogram

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv1D, Conv2D
from keras.layers import Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import MaxPooling2D, concatenate, Input, Concatenate
from keras.layers import Activation, Reshape, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical


path = "/Users/ronjaronnback/Desktop/University/Year_3/Semester 2/Introduction to Deep Learning/Assignment/Data_Raw_signals.pkl"
raw_train_data, test_data = pd.read_pickle(path)

X = raw_train_data
y = test_data

(unique, counts) = np.unique(y, return_counts=True)

frequencies = np.asarray((unique, counts)).T
print(frequencies)

#------------------------------------------------------------------------------
# MAKE SPECTROGRAMS
#------------------------------------------------------------------------------

def create_spectrograms(data):
    spectograms = []
    for d in data:
        freq, time, Sxx = spectrogram(d, 100, nperseg=200, noverlap=105)
        Sxx = Sxx[:, 1:,:]
        spectograms.append(Sxx)
    return np.array(spectograms), freq, time

train_data, frequencies, times = create_spectrograms(raw_train_data)

# 15000 samples, 2 spots on head, 100 Hz for 30 seconds
print(train_data.shape)
print(test_data.shape)
# should it be remodelled to 15375, 100, 30, 2??

train_data = train_data.reshape(15375, 100, 30, 2)

del X, y, unique, counts, path, raw_train_data

#------------------------------------------------------------------------------
# TRAIN TEST SPLIT
#------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(train_data, 
                                                    test_data, 
                                                    test_size=0.33, 
                                                    random_state=42)

X_train_left = X_train[:,:,:,0]
X_train_right = X_train[:,:,:,1]

X_test_left = X_test[:,:,:,0]
X_test_right = X_test[:,:,:,1]

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

del frequencies, times, test_data, train_data

#------------------------------------------------------------------------------
# CNN MODEL - PERFORMS BETTER
#------------------------------------------------------------------------------
"""
Source : https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

Best Performance: 0.59

"""

def evaluate_cnn_model(trainX, trainy, testX, testy):
    verbose = 1
    epochs = 20
    batch_size = 20
    
    height = trainX.shape[1]
    width = trainX.shape[2]
    depth = trainX.shape[3]
    n_outputs = trainy.shape[1]
    
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(height, width, depth)))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

evaluate_cnn_model(X_train, Y_train, X_test, Y_test)

#------------------------------------------------------------------------------
# CNN LSTM MODEL
#------------------------------------------------------------------------------
"""
Source : https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

Best Performance: 

"""


def evaluate_lstm_model(trainX, trainy, testX, testy):
    verbose = 1
    epochs = 20
    batch_size = 20
    
    height = trainX.shape[1]
    width = trainX.shape[2]
    depth = trainX.shape[3]
    n_outputs = trainy.shape[1]
    
    print(height, width, depth, n_outputs)
    
    lstm_model = Sequential()
    lstm_model.add(TimeDistributed(Flatten(input_shape=(30,2))))
    lstm_model.add(LSTM(100))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(6, activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(lstm_model.summary())
    
    # fit network
    lstm_model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	
    # evaluate model
    _, accuracy = lstm_model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

evaluate_lstm_model(X_train, Y_train, X_test, Y_test)

#------------------------------------------------------------------------------
# CHANNEL SPLIT MODEL
#------------------------------------------------------------------------------

# define two sets of inputs
inputLeft = Input(shape=(100,30))
inputRight = Input(shape=(100,30))

# the first branch operates on the first input
left = Conv1D(filters=64, kernel_size=3, activation='relu')(inputLeft)
left = Conv1D(filters=64, kernel_size=3, activation='relu')(left)
left = Dropout(0.5)(left)
left = MaxPooling1D(pool_size=2)(left)
left = Flatten()(left)
left = Dense(100, activation='relu')(left)
left = Model(inputs=inputLeft, outputs=left)

# the second branch opreates on the second input
right = Conv1D(filters=64, kernel_size=3, activation='relu')(inputRight)
right = Conv1D(filters=64, kernel_size=3, activation='relu')(right)
right = Dropout(0.5)(right)
right = MaxPooling1D(pool_size=2)(right)
right = Flatten()(right)
right = Dense(100, activation='relu')(right)
right = Model(inputs=inputRight, outputs=right)

# combine the output of the two branches
combined = concatenate([left.output, right.output])

# apply a FC layer and then a regression prediction on the
# combined outputs
final = Dense(10, activation="relu")(combined)
final = Dense(6, activation="linear")(final)
# our model will accept the inputs of the two branches and
# then output a single value
model = Model(inputs=[left.input, right.input], outputs = final)

print(model.summary)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit([X_train_left, X_train_right], Y_train, epochs=50, batch_size=20, verbose=2)

# evaluate model
model.evaluate([X_test_left, X_train_right], Y_test, batch_size=20, verbose=2)




"""
# THIS ONE WORKS BUT ONLY TAKES ONE CHANNEL INPUT
def evaluate_model(trainX, trainy, testX, testy):
	verbose, epochs, batch_size = 1, 10, 32
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(Dropout(0.5))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy


evaluate_model(X_train_left, Y_train, X_test_left, Y_test)
"""
   
   