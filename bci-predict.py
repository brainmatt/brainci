#!/usr/bin/env python

# //towardsdatascience.com/merging-with-ai-how-to-make-a-brain-computer-interface-to-communicate-with-google-using-keras-and-f9414c540a92

import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
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
#X_train, X_test, y_train, y_test = train_test_split(imageDirectory, answerDirectory, test_size=0.1)

# use all given data, no splitting
X_train = imageDirectory
y_train = answerDirectory

#print("X_train:" + str(X_train))

model = tf.keras.models.load_model(save_model_path)
model.summary()
#loss2, acc2 = model.evaluate(X_train,  y_train, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100*acc2))

# prediction
# https://www.tensorflow.org/tutorials/keras/classification
y_predicted = model.predict(X_train)

count1 = 0
count2 = 0
count3 = 0
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

#model.summary()
#loss1, acc1 = model.evaluate(X_train,  y_train, verbose=2)
#print('Trained model, accuracy: {:5.2f}%'.format(100*acc1))








