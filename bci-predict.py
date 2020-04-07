#!/usr/bin/env python

# //towardsdatascience.com/merging-with-ai-how-to-make-a-brain-computer-interface-to-communicate-with-google-using-keras-and-f9414c540a92

import sys
import os
import datetime
from time import time, strftime, gmtime
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
np_utils=tf.keras.utils
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from tensorflow import keras
from optparse import OptionParser

currentpath = os.path.dirname(os.path.realpath(sys.argv[0]))
default_epochs = 300

parser = OptionParser()
parser.add_option("-f", "--filename", dest="filename", type='str', help="The recorded and combined sample csv file.")
parser.add_option("-m", "--model", dest="model", type='str', help="The name for the model to restore.")
(options, args) = parser.parse_args()
if not options.filename:
    print("ERROR: please use -f to specify the recorded and combined sample csv file!")
    sys.exit(1)
dfile = str(options.filename)
print("Using combined datafile: " + dfile)

if not options.model:
    print("ERROR: please use -m to restore a specific trained model!")
    sys.exit(1)

save_model_path = currentpath + "/models/" + options.model + ".h5"
if not os.path.isfile(save_model_path):
    print("ERROR: The specificed trained model does not exists!")
    sys.exit(1)

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

my_data = pd.read_csv(dfile, sep=',', header=None, skiprows=1)
my_data = np.array(my_data)
dt_str = np.dtype(str)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Remove Noise
nsamples = my_data[:, 1].shape[0]

T = nsamples/400
t = np.linspace(0, T, nsamples, endpoint=False)
fs = 400.0
lowcut = 4.0
highcut = 50.0
my_data[:, 2] = butter_bandpass_filter(my_data[:, 2], lowcut, highcut, fs, order=6)
my_data[:, 3] = butter_bandpass_filter(my_data[:, 3], lowcut, highcut, fs, order=6)
my_data[:, 4] = butter_bandpass_filter(my_data[:, 4], lowcut, highcut, fs, order=6)
my_data[:, 5] = butter_bandpass_filter(my_data[:, 5], lowcut, highcut, fs, order=6)

# Separate words
lineIndex = 0
#currentWord = 2
currentWord = 1
imageLength = 110
currentImage = np.zeros(4)
imageDimensions = (imageLength, 4)
imageDirectory = np.zeros(imageDimensions)
answerDirectory = np.zeros(1)


while lineIndex < my_data.shape[0]:
    currentLine = np.array(my_data[lineIndex])
    if int(currentLine[0]) == currentWord:
        currentImage = np.vstack((currentImage, currentLine[2:]))
    else:
        currentImageTrimmed = np.delete(currentImage, 0, 0)
        currentImageTrimmed = np.vsplit(currentImageTrimmed, ([imageLength]))[0]
        if currentImageTrimmed.shape[0] < imageLength:
            print("ERROR: Invalid Image at currentWord = " + str(currentWord))
            exit(1)
        imageDirectory = np.dstack((imageDirectory, currentImageTrimmed))
        answerDirectory = np.vstack((answerDirectory, currentLine[1]))
        currentImage = np.zeros(4)
        currentWord = currentLine[0]
    lineIndex += 1

#print(imageDirectory)

imageDirectory = np.transpose(imageDirectory, (2, 0, 1))
imageDirectory = np.delete(imageDirectory, 0, 0)
answerDirectory = np.delete(answerDirectory, 0, 0)
answerDirectory = np_utils.to_categorical(answerDirectory)



# Split to Training and Testing Set
X_train, X_test, y_train, y_test = train_test_split(imageDirectory, answerDirectory, test_size=0.1)

#print("X_train:" + str(X_train))
#print("X_test:" + str(X_test))

model = tf.keras.models.load_model(save_model_path)
model.summary()
#loss2, acc2 = model.evaluate(X_test,  y_test, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100*acc2))




# Build Model
#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Conv1D(40, 10, strides=2, padding='same', activation='relu', input_shape=(imageLength, 4)))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.MaxPooling1D(3))
#model.add(tf.keras.layers.GlobalAveragePooling1D())
#model.add(tf.keras.layers.Dense(50, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))
#model.add(tf.keras.layers.Dense(4, activation='softmax'))

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.summary()

# add tensorboard logging
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train Model
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=default_epochs, callbacks=[tensorboard_callback])
#history_dict = history.history
#history_dict.keys()
#print("history: " + str(history_dict['accuracy']))

#import matplotlib.pyplot as plt

#acc = history_dict['accuracy']
#val_acc = history_dict['val_accuracy']
#loss = history_dict['loss']
#val_loss = history_dict['val_loss']

#epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
#plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()

#plt.show()


#plt.clf()   # clear figure

#plt.plot(epochs, acc, 'bo', label='Training acc')
#plt.plot(epochs, val_acc, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend(loc='lower right')

#plt.show()

# prediction
# https://www.tensorflow.org/tutorials/keras/classification
print("Prediction ----")
#y_predicted = model.predict(X_test)
y_predicted = model.predict(X_train)


count1 = 1
count2 = 2
count3 = 3
countloop = 0

print("Predictions :")
for p in y_predicted:
    #print(p)
    pv = np.argmax(p)
    print(pv)
    if pv == 1:
        count1 = count1 + 1
    if pv == 2:
        count2 = count2 + 1
    if pv == 3:
        count3 = count3 + 1
    countloop = countloop + 1

count1percent = (count1*100)/countloop
count2percent = (count2*100)/countloop
count3percent = (count3*100)/countloop

print("Predict 1: " + str(count1) + " =  {:5.2f}%".format(count1percent))
print("Predict 2: " + str(count2) + " =  {:5.2f}%".format(count2percent))
print("Predict 3: " + str(count3) + " =  {:5.2f}%".format(count3percent))




#print("X_test :")
#print(X_test)
#for x in X_test:
#    print(x)

# validate
#model = tf.keras.models.load_model(save_model_path)
#model.summary()
#loss2, acc2 = model.evaluate(X_test,  y_test, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100*acc2))







