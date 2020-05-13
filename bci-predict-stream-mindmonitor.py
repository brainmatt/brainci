#!/usr/bin/env python
# runs only in python3 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import sys
import time
import logging
import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
import numpy as np
from numpy import genfromtxt, newaxis, zeros
from scipy.signal import butter, lfilter
import tensorflow as tf
from tensorflow import keras


# from https://mind-monitor.com/forums/viewtopic.php?t=858

currentpath = os.path.dirname(os.path.realpath(sys.argv[0]))
logging.basicConfig(filename=currentpath + '/brainmatt.log',level=logging.INFO)

# blinks
blinks = 0
# last blink timestamp
lastblink = 0
blinkinterval = 1

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

# model path
load_model_path = ""

# global model
model = 0

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


def eeg_handler(unused_addr,ch1,ch2,ch3,ch4,ch5,ch6):
    global recording
    global currentpath
    global sample_array
    global sample_count
    global sample_single_sample_array
    global conv1d_array_max
    global sample_count_elements_max
    global sample_array_count

    if recording:
        #print("EEG per channel: ",ch2,ch3,ch4,ch5)
        #print("recording ...................")
        if not ch2 or not ch3 or not ch4 or not ch5:
            print("!!!! invalid sample")
            return

        # add EEG channels to single sample array
        sample_single_sample_array = np.append(sample_single_sample_array, [[ch2,ch3,ch4,ch5]], axis=0)
        sample_count = sample_count + 1
        #print(sample_count)
        if sample_count == sample_count_elements_max:
            sh = sample_single_sample_array.shape
            if sh != (110, 4):
                print("single sample array invalid, skipping")
                print(sh)
                sample_single_sample_array = np.empty([0,4])
                sample_count = 0
            else:
                # add single sample array into main sample array
                #print(sample_single_sample_array)
                sample_array = np.append(sample_array, [sample_single_sample_array], axis=0)
                sample_count = 0
                sample_array_count = sample_array_count + 1
                # empty single sample array
                sample_single_sample_array = np.empty([0,4])
                # check for how many main samples we want
                print(sample_array_count)
                if sample_array_count == conv1d_array_max:
                    # stop recording
                    recording = False
                    sample_array_count = 0
                    # predict sample array
                    predict_sample()

        elif sample_count > sample_count_elements_max:
            print("Skipping outward count sample, resetting sample count")
            sample_count = 0
            sample_single_sample_array = np.empty([0,4])


def predict_sample():
    global recording
    global currentpath
    global sample_array
    global model

    print("Now predicting recorded samples...")
    print(sample_array)

    fs = 400.0
    lowcut = 4.0
    highcut = 50.0
    sample_array[:, 0] = butter_bandpass_filter(sample_array[:, 0], lowcut, highcut, fs, order=6)
    sample_array[:, 1] = butter_bandpass_filter(sample_array[:, 1], lowcut, highcut, fs, order=6)
    sample_array[:, 2] = butter_bandpass_filter(sample_array[:, 2], lowcut, highcut, fs, order=6)
    sample_array[:, 3] = butter_bandpass_filter(sample_array[:, 3], lowcut, highcut, fs, order=6)

    print("sample_array after bandpass filter")
    print(sample_array)

    print("Prediction: ")
    predicted_arr = model.predict(sample_array)
    print(predicted_arr)

    count1 = 0
    count2 = 0
    count3 = 0
    countloop = 0

    print("Predictions :")
    for p in predicted_arr:
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

    # reset main sample array
    sample_array = np.empty([0,110,4])


def blink_handler(unused_addr,ch1,ch2):
    #print("Blink: ",ch1,ch2)
    global blinks
    global recording
    global lastblink
    global blinkinterval
    global currentpath
    global sample_array
    global sample_single_sample_array

    # check if we blink more than once in the given blinkinterval
    ts = time.time()
    #print("lastblink = " + str(lastblink) + " ts = " + str(ts) + " lastblink = " + str(lastblink))
    if (ts - lastblink) < blinkinterval:
        if blinks > 0:
            print("!! blinked 2. time within " + str(blinkinterval) + "s")

            if recording:
                print('BRAINMATT: already recording, skipping restarting recording')
            else:
                print('BRAINMATT: start recording sample....')
                # initialyze global sample_array
                sample_array = np.empty([0,110,4])
                sample_single_sample_array = np.empty([0,4])
                # enable recording in eeg_handler
                recording = True
        else:
            blinks = 1
            print("setting blinks = " + str(blinks))
    else:
        #print("resetting blinks")
        blinks = 0

    lastblink = time.time()



if __name__ == '__main__':
    port = 5000
    ip = "192.168.88.109"
    model_name = "mymodel_supi"
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ip", help="local ip address to run the brainmatt-server on")
    parser.add_argument("-p", "--port", help="local port to run the brainmatt-server on")
    parser.add_argument("-l", "--load", help="name of the trained model to load")
    args = vars(parser.parse_args())

    if not args['ip']:
        logging.info('BRAINMATT: no ip given, using default ip: ' + str(ip))
    else:
        ip = args['ip']
        logging.info('BRAINMATT: ip given, using default ip: ' + str(ip))

    if not args['port']:
        logging.info('BRAINMATT: no port given using default port: ' + str(port))
    else:
        port = args['port']
        logging.info('BRAINMATT: port given, using port: ' + str(port))

    if not args['load']:
        logging.info('BRAINMATT: no model name given, using: ' + str(model_name))
    else:
        model_name = args['load']
        logging.info('BRAINMATT: model name given, using model: ' + str(port))

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

    logging.info('BRAINMATT: starting brainmatt-server')

    logging.info('BRAINMATT: initialyze dispatcher')
    # http://forum.choosemuse.com/t/muse-direct-osc-stream-to-python-osc-on-win10/3506/2
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_handler, "EEG")
    dispatcher.map("/muse/elements/blink", blink_handler, "EEG")

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    logging.info("BRAINMATT: serving on {}".format(server.server_address))
    print("BRAINMATT: serving on {}".format(server.server_address))
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        logging.info('BRAINMATT: stopping brainmatt-server')
        print('BRAINMATT: stopping brainmatt-server')



