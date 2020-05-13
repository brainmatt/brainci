#!/usr/bin/env python

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import sys
import csv
import numpy as np
import pandas as pd
import random
from time import time, strftime, gmtime
from optparse import OptionParser
from pylsl import StreamInlet, resolve_byprop
from sklearn.linear_model import LinearRegression
import subprocess

currentpath = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = OptionParser()
parser.add_option("-p", "--path", dest="path", type='str', help="Directory of the recorded csv files.")
(options, args) = parser.parse_args()
if not options.path:
    print("ERROR: please use -p to specify the path to the recorded csv files!")
    sys.exit(1)

filename_formatted = options.path + "/combined_eeg_data.csv"

def reformat_datafile(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        next(reader, None)  # skip the headers
        for row in reader:
            #print(row)
            word = int(row[1])
            term = int(row[2])
            aval = float(row[3])
            bval = float(row[4])
            cval = float(row[5])
            dval = float(row[6])
            d = str(word) + "," + str(term) + "," + str(aval) + "," + str(bval) + "," + str(cval) + "," + str(dval)
            #print(d)
            data.append(d)

    if os.path.isfile(filename_formatted):
        print("Combined datafile " + filename_formatted + " already exists. Adding data")
        outf = open(filename_formatted, "a")
    else:
        print("Creating combined datafile " + filename_formatted)
        outf = open(filename_formatted, "w")
        outf.write("WORD,TERM,A,B,C,D\n")

    for n in data:
        outf.write(n + "\n")

print("Using path: " + str(options.path))

for file in sorted(os.listdir(options.path)):
    if file.endswith(".csv"):
        if not "eeg_data" in file:
            print(os.path.join(options.path, file))
            reformat_datafile(os.path.join(options.path, file))

print("Combined Outputfile: " + filename_formatted)

