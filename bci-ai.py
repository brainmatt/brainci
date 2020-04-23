#!/usr/bin/env python

# //towardsdatascience.com/merging-with-ai-how-to-make-a-brain-computer-interface-to-communicate-with-google-using-keras-and-f9414c540a92

import sys
import os
import datetime
from time import time, strftime, gmtime
import tensorflow as tf
import numpy as np
from numpy import genfromtxt, newaxis, zeros
np_utils=tf.keras.utils
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from tensorflow import keras
from optparse import OptionParser
import matplotlib.pyplot as plt

currentpath = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = OptionParser()
parser.add_option("-f", "--filename", dest="filename", type='str', help="The recorded and combined csv file.")
parser.add_option("-e", "--epochs", dest="epochs", type='int', help="The number of epochs to devide the data in.")
parser.add_option("-m", "--model", dest="model", type='str', help="The name for saving the model.")
parser.add_option("-l", "--loadmodel", dest="load", type='str', help="The name for the model to load.")
(options, args) = parser.parse_args()
if not options.filename:
    print("ERROR: please use -f to specify the recorded and combined csv file!")
    sys.exit(1)
dfile = str(options.filename)
print("Using combined datafile: " + dfile)

default_epochs = 300
if options.epochs:
    default_epochs = options.epochs
print("Using " + str(default_epochs) + " epochs")

save_model = False
if options.model:
    save_model = True

load_model_path = ""
load_model = False
if options.load:
    load_model = True
    load_model_path = currentpath + "/models/" + options.load + ".h5"
    if not os.path.isfile(load_model_path):
        print("ERROR: The specificed trained model to load does not exists!")
        sys.exit(1)


log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#my_data = pd.read_csv(dfile, sep=',', header=None, index_col='WORD', names=['WORD', 'TERM', 'A', 'B', 'C', 'D'])
my_data = pd.read_csv(dfile, sep=',', header=None, skiprows=1)
my_data = np.array(my_data)
#print(my_data.size)
#print(my_data.shape)
#print(repr(my_data))
#mydata_array = np.array(my_data)
#print(mydata_array)
#print(mydata_array[:, 0].shape[0])

dt_str = np.dtype(str)
#types = ['i4', 'U50', 'f8', 'f8', 'f8', 'f8']
#my_data = np.genfromtxt(dfile, delimiter=',', dtype=types, names=True)
#print(my_data['WORD'])
#print(my_data.size)
#print(my_data.shape)
#my_data = my_data.reshape(854, 6)
#my_data = my_data.reshape(my_data.shape[0]/5, 6)
#print(repr(my_data))

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


#print(repr(my_data))



# Separate words
lineIndex = 0
#currentWord = 2
currentWord = 1
imageLength = 110
currentImage = np.zeros(4)
imageDimensions = (imageLength, 4)
imageDirectory = np.zeros(imageDimensions)
answerDirectory = np.zeros(1)
term_arr = []

while lineIndex < my_data.shape[0]:
    currentLine = np.array(my_data[lineIndex])
    #print("lineIndex: " + str(lineIndex) + " < shape: " + str(my_data.shape[0]))
    #print(" - currentLine: " + str(currentLine))
    #print("currentLine[0]: " + str(currentLine[0]) + " == currentWord: " + str(currentWord))
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
        #print("imageDirectory: " + str(imageDirectory.shape) + "\n")
        currentImage = np.zeros(4)
        currentWord = currentLine[0]


    lineIndex += 1

imageDirectory = np.transpose(imageDirectory, (2, 0, 1))
imageDirectory = np.delete(imageDirectory, 0, 0)
answerDirectory = np.delete(answerDirectory, 0, 0)
answerDirectory = np_utils.to_categorical(answerDirectory)



# Split to Training and Testing Set
X_train, X_test, y_train, y_test = train_test_split(imageDirectory, answerDirectory, test_size=0.3)

#print(str(X_train))


# load existing trained model
model = None
if load_model:
    print("Loading trained model from: " + load_model_path)
    model = tf.keras.models.load_model(load_model_path)
    model.summary()

else:


    # Build Model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(40, 10, strides=2, padding='same', activation='relu', input_shape=(imageLength, 4)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # add tensorboard logging
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train Model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=100, epochs=default_epochs, callbacks=[tensorboard_callback])
    history_dict = history.history
    #history_dict.keys()
    print("history: " + str(history_dict['accuracy']))


    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #plt.show()


    #plt.clf()   # clear figure
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    #plt.show()



# prediction
# https://www.tensorflow.org/tutorials/keras/classification
print("Prediction ----")
y_predicted = model.predict(X_test)
#y_predicted = model.predict(X_train)


for x in y_test:
#    print(x)
    if x[1] > 0:
        term_arr.append(1)
    if x[2] > 0:
        term_arr.append(2)
    if x[3] > 0:
        term_arr.append(3)

#print(term_arr)

term_loop = 0
print("Predictions :")
for p in y_predicted:
    print("Term: " + str(term_arr[term_loop]) + " == " +  str(np.argmax(p)))
    term_loop = term_loop + 1


model.summary()
loss1, acc1 = model.evaluate(X_test,  y_test, verbose=2)
print('Trained model, accuracy: {:5.2f}%'.format(100*acc1))

while True:
    try:
        #sample = 3
        sample = int(input("Please enter a sample number: "))

        #print("single sample data: ")
        #print(X_test[sample])
        #print("single sample shape: ")
        #print(str(X_test[sample].shape))

        sample_reshaped = np.expand_dims(X_test[sample], axis=0)
        #print("single sample after reshape: ")
        #print(str(sample_reshaped.shape))

        #print("single sample data after reshape: ")
        #print(sample_reshaped)

        #print("full test data shape: ")
        #print(X_test.shape)

        print("Single Prediction sample: " + str(sample))
        print("Single Prediction answer: " + str(term_arr[sample]))
        single_predicted = model.predict(sample_reshaped)
        print("Single Prediction predicted: " + str(np.argmax(single_predicted)))
        print("---------------------------------")

    except KeyboardInterrupt:
        print("")
        break

if save_model:
    save_it = input("Would you like to save this trained model? (y/n)")
    if save_it=="y":
        save_model_path = currentpath + "/models/" + str(options.model) + ".h5"
        print("Saving trained model at: " + save_model_path)
        model.save(save_model_path)
        # validate
        new_model = tf.keras.models.load_model(save_model_path)
        new_model.summary()
        loss2, acc2 = model.evaluate(X_test,  y_test, verbose=2)
        print('Restored model, accuracy: {:5.2f}%'.format(100*acc2))
    else:
        print("Skipping saving trained model")





