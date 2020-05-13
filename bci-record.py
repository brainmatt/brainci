#!/usr/bin/env python

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import sys
import csv
import numpy as np
import pandas as pd
import random
from time import time, strftime, gmtime, sleep
from optparse import OptionParser
from pylsl import StreamInlet, resolve_byprop
from sklearn.linear_model import LinearRegression
import subprocess

currentpath = os.path.dirname(os.path.realpath(sys.argv[0]))

# dejitter timestamps
dejitter = False

# how long to wait for the Muse device to connect
muse_connect_timout = 5

parser = OptionParser()
parser.add_option("-d", "--duration",
                  dest="duration", type='int', default=10,
                  help="duration of the recording in seconds.")
parser.add_option("-p", "--path",
                  dest="path", type='str',
                  help="Directory for the recording file.")
parser.add_option("-s", "--sample",
                  dest="sample", type='str',
                  help="Record sample for specific term (1/2/3)")

(options, args) = parser.parse_args()


record_sample = False
record_sample_term = None
if options.sample:
    record_sample = True
    print("NOTICE: Creating sample dataset for term: " + str(options.sample))
    record_sample_term = options.sample
else:
    print("NOTICE: Creating training dataset with random terms")

if not options.path:
    print("ERROR: please use -p to specify the datapath to the recorded csv files!")
    sys.exit(1)

fname = (options.path + "/data_%s.csv" % strftime("%Y-%m-%d-%H.%M.%S", gmtime()))

# search for the last word id in eventually already existing datafiles
last_data_file = None
for file in sorted(os.listdir(options.path)):
    if file.endswith(".csv"):
        if not "eeg_data" in file:
            print(os.path.join(options.path, file))
            last_data_file = os.path.join(options.path, file)

if last_data_file:
    print("Found existing datafiles! Getting currentWord from last datafiles: " + last_data_file)
    
    line = subprocess.check_output(['tail', '-1', last_data_file])
    line = str(line)
    linesplit = line.split(",")
    #print(linesplit[1])
    print("Starting currentWord from " + str(linesplit[1]))
    currentWord = int(linesplit[1])
    currentWord = currentWord + 1

else:
    print("Did not found any existing datafiles! Starting currentWord from 1")
    currentWord = 1

print("-- currentWord: " + str(currentWord))


eeg_stream = False
print("looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', timeout=2)

if len(streams) == 0:
    print("No EEG stream running yet. Trying to start the Muse EEG stream ...")
    eeg_stream = subprocess.Popen([ currentpath + "/bci-stream"])
    sleep(muse_connect_timout)
    streams = resolve_byprop('type', 'EEG', timeout=2)

if len(streams) == 0:
    raise(RuntimeError, "Cant find EEG stream")
else:
    print("Success: found Muse EEG stream")
    
print("Start aquiring data")
inlet = StreamInlet(streams[0], max_chunklen=12)
eeg_time_correction = inlet.time_correction()

inlet_marker = False
#print("looking for a Markers stream...")
#marker_streams = resolve_byprop('type', 'Markers', timeout=2)
#if marker_streams:
#    inlet_marker = StreamInlet(marker_streams[0])
#    marker_time_correction = inlet_marker.time_correction()
#else:
#    inlet_marker = False
#    print("Cant find Markers stream")

info = inlet.info()
description = info.desc()

freq = info.nominal_srate()
Nchan = info.channel_count()

ch = description.child('channels').first_child()
ch_names = [ch.child_value('label')]
for i in range(1, Nchan):
    ch = ch.next_sibling()
    ch_names.append(ch.child_value('label'))


# Word Capturing    
#currentWord = 1
currentTerm = "1"
t_word = time() + 1 * 2
words = []
terms = []
termBank = ["1", "2", "3"]
subdisplay = False

res = []
timestamps = []
markers = []
t_init = time()
print('Start recording at time t=%.3f' % t_init)
print(currentTerm)
while (time() - t_init) < options.duration:
    if time() >= t_word:
        if subdisplay:
            subdisplay.kill()
	# Check for new word
    if time() >= t_word:
        # sample or training data recording ?
        if record_sample:
            currentTerm = record_sample_term
        else:
            currentTerm = random.choice(termBank)

        print(str(currentWord) +": " +currentTerm)
        subdisplay = subprocess.Popen([ "/usr/bin/display", currentpath + "/images/" + currentTerm + ".png"])

        currentWord += 1
        t_word = time() + 1 * 2
    try:
        data, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=12)
        if timestamp:
            res.append(data)
            timestamps.extend(timestamp)
            words.extend([currentWord] * len(timestamp))
            terms.extend([currentTerm] * len(timestamp))
        if inlet_marker:
            marker, timestamp = inlet_marker.pull_sample(timeout=0.0)
            if timestamp:
                markers.append([marker, timestamp])
    except KeyboardInterrupt:
        break

if subdisplay:
    subdisplay.kill()

res = np.concatenate(res, axis=0)
timestamps = np.array(timestamps)

if dejitter:
    y = timestamps
    X = np.atleast_2d(np.arange(0, len(y))).T
    lr = LinearRegression()
    lr.fit(X, y)
    timestamps = lr.predict(X)

res = np.c_[timestamps, words, terms, res]
data = pd.DataFrame(data=res, columns=['timestamps'] + ['words'] + ['terms'] + ch_names)

data['Marker'] = 0
# process markers:
for marker in markers:
    # find index of margers
    ix = np.argmin(np.abs(marker[1] - timestamps))
    val = timestamps[ix]
    data.loc[ix, 'Marker'] = marker[0][0]


data.to_csv(fname, float_format='%.3f', index=False)
print('Wrote datafile: ' + fname)

if eeg_stream:
    print("Found running EEG stream. Stopping it")
    eeg_stream.kill()

print("Success")

