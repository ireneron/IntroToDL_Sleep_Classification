#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:17:17 2020

@author: ronjaronnback

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pydot

from scipy.signal import spectrogram

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Flatten, TimeDistributed, Conv1D, Conv2D
from keras.layers import Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import MaxPooling2D, concatenate, Input, Concatenate
from keras.layers import Activation, Reshape, BatchNormalization, GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical

from keras.utils.vis_utils import plot_model

from sklearn.metrics import confusion_matrix

# =============================================================================
# TRAINING DATA
# =============================================================================
path = "/Users/ronjaronnback/Desktop/University/Year_3/Semester 2/Introduction to Deep Learning/Assignment/Data_Raw_signals.pkl"
raw_train_data, test_data = pd.read_pickle(path)

X = raw_train_data
y = test_data

(unique, counts) = np.unique(y, return_counts=True)

frequencies = np.asarray((unique, counts)).T
print(frequencies)

# =============================================================================
# FINAL TESTING
# =============================================================================
path = "/Users/ronjaronnback/Desktop/University/Year_3/Semester 2/Introduction to Deep Learning/Assignment/Test_Raw_signals_no_labels.pkl"
test_raw = pd.read_pickle(path)

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
final_test, frequencies, times = create_spectrograms(test_raw)

# 15000 samples, 2 spots on head, 100 Hz for 30 seconds
print(train_data.shape)
print(test_data.shape)
print(final_test.shape)
# should it be remodelled to 15375, 100, 30, 2??

final_test = final_test.reshape(1754, 1, 101, 30)
final_test = final_test.reshape(1754, 101, 30, 1)

train_data = train_data.reshape(15375, 100, 30, 2)

del X, y, unique, counts, path, raw_train_data, test_raw, times, frequencies

# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================
from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(train_data, 
                                                    test_data, 
                                                    test_size=0.2, 
                                                    random_state=42)


#https://stackoverflow.com/questions/46443566/keras-lstm-multiclass-classification
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

del test_data, train_data

#------------------------------------------------------------------------------
# LSTM MODEL
#------------------------------------------------------------------------------
"""
Source : https://stackoverflow.com/questions/52138290/how-can-we-define-one-to-one-one-to-many-many-to-one-and-many-to-many-lstm-ne

Best Performance: 0.81
- neurons = 100
- loss = categorical_crossentropy
- optimizer = adam
- dropout = 0.2

Other Try: 0.81 as well
- neurons = 100
- loss = 'kullback_leibler_divergence'
- optimizer = nadam
- dropout = 0.2

Inspiration for more?
https://github.com/CVxTz/EEG_classification/blob/f440a7f077d3dcb7c5ed5a5688ae62e8e1e100dc/code/models.py#L68

"""

verbose = 1
epochs = 25
batch_size = 100

height = X_train.shape[1]
width = X_train.shape[2]
depth = X_train.shape[3]
n_outputs = Y_train.shape[1]

lstm_model = Sequential()
lstm_model.add(TimeDistributed(Flatten(input_shape=(width,depth)))) # https://stackoverflow.com/questions/52936132/4d-input-in-lstm-layer-in-keras
lstm_model.add(GRU(200))
lstm_model.add(Dropout(0.5))
#lstm_model.add(Dense(50))
#lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(n_outputs, activation='softmax'))
lstm_model.compile(loss='kullback_leibler_divergence', optimizer='nadam', metrics=['accuracy'])

# loss 'categorical_crossentropy', 'kullback_leibler_divergence'
# opt 'Nadam', 'Adamax', 'Adadelta'

# fit network
history = lstm_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data = (X_test, Y_test))
print(lstm_model.summary())

#plot training and validation accuracy
plt.figure(0)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# =============================================================================
# 
# =============================================================================

# evaluate model
lstm_model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)

#plot confusion matrix
y_pred = lstm_model.predict_on_batch(final_test)
cm = confusion_matrix(np.argmax(Y_test, axis = 1), np.argmax(y_pred, axis=1))
df_cm = pd.DataFrame(cm, range(6), range(6))

plt.figure(1)
sn.set(font_scale=1.2) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 10},  fmt='g') # font size
plt.show()



#export answers


final_predictions = np.argmax(y_pred, axis=1)

a_file = open("/Users/ronjaronnback/Desktop/answer.txt", "w")
np.savetxt(a_file, final_predictions, fmt = '%d')
a_file.close()

# =============================================================================
# Save and Load Model
# =============================================================================
# save model and architecture to single file
lstm_model.save("/Users/ronjaronnback/Desktop/model.h5")
print("Saved model to disk")

from keras.models import load_model
# load model
model = load_model('/Users/ronjaronnback/Desktop/model.h5')
# summarize model.
model.summary()
