# Brainci reads your mind - A brain/computer interface for home automation
A collection of tools to record EEG brainwaves from a Muse device, train an artificial intelligence with the collected brain data to predict terms from thoughts. It can be easily used for home automation to e.g. turn your TV on or off.

[![Click for the introduction Youtube video: Brainci, a brain/computer interface for home automation](https://img.youtube.com/vi/2QpyqPsgJfI/0.jpg)](https://www.youtube.com/watch?v=2QpyqPsgJfI "Click for the introduction Youtube video: Brainci, a brain/computer interface for home automation")


bci-record.py:
- Starts "bci-stream" to initate the Muse EEG stream
- Records EEG brainwaves from a Muse 2 device
- Visual stimulation for a defined set of terms (1,2,3 by default)
- Marker stream to mark the recorded samples with the displayed terms
- Allows to record "training" and "sample" data for the AI

bci-stream:
- Starts Muse LSL stream
- Please adapt the Mac address of your Muse 2 device in this file

bci-combine.py:
- Combines and formats multiple recorded EEG datafiles in a specified directory

bci-ai.py:
- Reads the combined and formatted EEG datafile
- Trains a "tensorflow" artificial intelligence with the provided data
- Saves the trained model

bci-predict.py:
- Reads a combined and formatted EEG sample
- Restores a previously saved trained model
- Runs a prediction accross all the collected samples
- Output the prediction of each term in percent

Enjoy, hope you have fun with it!


