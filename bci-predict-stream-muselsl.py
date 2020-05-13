#!/usr/bin/env python

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import sys
import csv
import logging
import pandas as pd
import random
import argparse
from time import time, strftime, gmtime, sleep
from pylsl import StreamInlet, resolve_byprop
from sklearn.linear_model import LinearRegression
import numpy as np
from numpy import genfromtxt, newaxis, zeros
from scipy.signal import butter, lfilter
import tensorflow as tf
from tensorflow import keras
import subprocess

currentpath = os.path.dirname(os.path.realpath(sys.argv[0]))

# dejitter timestamps
dejitter = False

# addtional marker stream
inlet_marker = False

# how long to wait for the Muse device to connect
muse_connect_timout = 10

# default trained model
model_name = "mymodel_supi"

# global model
model = 0

# are we actively recording ?
recording = False

# initialyze recording arrays
sample_array = np.empty([0,110,4])
sample_single_sample_array = np.empty([0,4])

# sample count to fit into (110,4) np array
sample_count = 0
sample_array_count = 0

# how many samples to count per conv1d array
sample_count_elements_max = 110

# conv1d sample arrays max
conv1d_array_max = 10

# found trigger state
found_trigger = False

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



def predict_sample():
    global sample_array
    global model

    print("Now predicting recorded samples...")
    #print(sample_array)

    fs = 400.0
    lowcut = 4.0
    highcut = 50.0
    sample_array[:, 0] = butter_bandpass_filter(sample_array[:, 0], lowcut, highcut, fs, order=6)
    sample_array[:, 1] = butter_bandpass_filter(sample_array[:, 1], lowcut, highcut, fs, order=6)
    sample_array[:, 2] = butter_bandpass_filter(sample_array[:, 2], lowcut, highcut, fs, order=6)
    sample_array[:, 3] = butter_bandpass_filter(sample_array[:, 3], lowcut, highcut, fs, order=6)

    #print("sample_array after bandpass filter")
    #print(sample_array)

    print("Predictions: ")
    predicted_arr = model.predict(sample_array)
    #print(predicted_arr)

    count1 = 0
    count2 = 0
    count3 = 0
    countloop = 0

    for p in predicted_arr:
        #print(p)
        pv = np.argmax(p)
        #print(pv)
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
    print("--------------------------------------------")
    
    final_prediction = 0
    if count1 > count2 and count1 > count3:
        print("final Prediction = 1")
        final_prediction = 1
    if count2 > count1 and count2 > count3:
        print("final Prediction = 2")
        final_prediction = 2
    if count3 > count2 and count3 > count1:
        print("final Prediction = 3")
        final_prediction = 3

    if final_prediction == 1:
        # switch on tv
        print("ACTION: Switch my TV on")
        switchon = subprocess.Popen(args=[ currentpath + "/firetv/switch-on"], shell=True)
        sleep(5)
        if switchon:
            switchon.kill()

    if final_prediction == 2 or final_prediction == 3:
        # switch off tv
        print("ACTION: Switch my TV off")
        switchoff = subprocess.Popen(args=[ currentpath + "/firetv/switch-off"], shell=True)
        sleep(5)
        if switchoff:
            switchoff.kill()


    # reset main sample array
    sample_array = np.empty([0,110,4])


parser = argparse.ArgumentParser()
parser.add_argument("-l", "--load", help="name of the trained model to load")
args = vars(parser.parse_args())

if not args['load']:
    logging.info('BRAINMATT: no model name given, using: ' + str(model_name))

load_model_path = currentpath + "/models/" + model_name + ".h5"
if not os.path.isfile(load_model_path):
    print("ERROR: The specificed trained model to load does not exists!")
    sys.exit(1)
else:
    # load model 
    print("Loading trained model from: " + load_model_path)
    model = tf.keras.models.load_model(load_model_path)
    model.summary()

logging.info('BRAINMATT: loaded trained AI model from ' + load_model_path)

eeg_stream = False
eeg_stream = False
print("looking for an EEG + GRYO stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)
streams_gyro = resolve_byprop('type', 'GYRO', timeout=2)

if len(streams) == 0:
    print("No EEG stream running yet. Trying to start the Muse EEG stream ...")
    eeg_stream = subprocess.Popen([ currentpath + "/bci-stream-plus-gyro"])
    sleep(muse_connect_timout)
    streams = resolve_byprop('type', 'GYRO', timeout=2)

if len(streams) == 0:
    raise(RuntimeError, "Cant find EEG stream")
else:
    print("Success: found Muse EEG stream")

gyro_stream_retries = 5
gyro_stream_retry_loop = 0
while len(streams_gyro) == 0:
    print("Cant find GYRO stream! ... retrying")
    gyro_stream_retry_loop = gyro_stream_retry_loop + 1
    if gyro_stream_retry_loop >= gyro_stream_retries:
        print("Cant find GYRO stream! ... giving up")
    sleep(2)

print("Success: found Muse GYRO stream")

print("Start aquiring data")
# eeg
inlet = StreamInlet(streams[0], max_chunklen=12)
eeg_time_correction = inlet.time_correction()
info = inlet.info()
description = info.desc()
freq = info.nominal_srate()
Nchan = info.channel_count()

ch = description.child('channels').first_child()
ch_names = [ch.child_value('label')]
for i in range(1, Nchan):
    ch = ch.next_sibling()
    ch_names.append(ch.child_value('label'))

inlet.close_stream()

# gyro
#inlet_gyro = StreamInlet(streams_gyro[0], max_chunklen=12)

currentWord = 1
currentTerm = 1
# main loop
while True:
    try:

        print("Waiting for gyro trigger ......")
        gyro_data = []
        timestamp = []
        inlet_gyro = StreamInlet(streams_gyro[0], max_chunklen=12)
        while True:
            # read gyro data until we have found the trigger (moving the head to the right)
            gyro_data, timestamp = inlet_gyro.pull_chunk(timeout=1.0, max_samples=12)
            found_trigger = False
            for d in gyro_data:
                #print("X: " + str(d[0]) + " Y: " + str(d[1]) + " Z: " + str(d[2]))
                if d[0] > 20:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!! found gyro trigger !!!!!!!!!!!!!!!!!!")
                    found_trigger = True
                    break
            if found_trigger:
                d = []
                # suspend current gyro_stream
                inlet_gyro.close_stream()
                break


        t_init = time()
        print('Start recording at time t=%.3f' % t_init)
        recording = True
        data = []
        timestamp = []
        inlet = StreamInlet(streams[0], max_chunklen=12)
        while True:
            # read eeg data / conv1d_array_max * sample_count_elements_max
            data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=12)
            for e in data:
                #print(str(e[0]) + " - " + str(e[1]) + " - " + str(e[2]) + " - " + str(e[3]))
                # add EEG channels to single sample array
                sample_single_sample_array = np.append(sample_single_sample_array, [[e[0], e[1], e[2], e[3]]], axis=0)
                sample_count = sample_count + 1
                # print(sample_count)
                if sample_count == sample_count_elements_max:
                    sh = sample_single_sample_array.shape
                    # add single sample array into main sample array
                    # print(sample_single_sample_array)
                    sample_array = np.append(sample_array, [sample_single_sample_array], axis=0)
                    sample_count = 0
                    sample_array_count = sample_array_count + 1
                    # empty single sample array
                    sample_single_sample_array = np.empty([0, 4])
                    # check for how many main samples we want
                    #print(sample_array_count)
                    if sample_array_count == conv1d_array_max:
                        # stop recording
                        recording = False
                        sample_array_count = 0
                        # suspend current eeg stream
                        inlet.close_stream()
                        break

            if not recording:
                # predict sample array
                predict_sample()
                break



    except KeyboardInterrupt:
        break

if eeg_stream:
    print("Found running EEG stream. Stopping it")
    eeg_stream.kill()

print("Success")

