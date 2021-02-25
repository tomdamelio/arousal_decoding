# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
#         Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)
# Link https://mne.tools/dev/auto_examples/decoding/plot_decoding_spoc_CMC.html#sphx-glr-auto-examples-decoding-plot-decoding-spoc-cmc-py

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import Epochs
from mne.decoding import SPoC
from mne.datasets.fieldtrip_cmc import data_path

from autoreject import get_rejection_threshold


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from my_functions import extract_signal
from channel_names import channels_geneva, channels_twente 

# Define parameters
number_subject = '01'

#Extract signal
raw = extract_signal(directory = 'data', number_subject=number_subject,
                     extension = '.bdf')

# Rename channel EDA and set GSR as channel type
mne.rename_channels(info= raw.info , mapping={'GSR1':'EDA'})
# raw.ch_names # Reutrn list of channels

raw.set_channel_types({'EXG1': 'eog',
                       'EXG2': 'eog',
                       'EXG3': 'eog',
                       'EXG4': 'eog',
                       'EXG5': 'emg',
                       'EXG6': 'emg',
                       'EXG7': 'emg',
                       'EXG8': 'emg',
                       'EDA' : 'misc',
                       'GSR2': 'misc',
                       'Erg1': 'misc',
                       'Erg2': 'misc',
                       'Resp': 'misc',
                       'Plet': 'misc',
                       'Temp': 'misc'})

# Pick EDA and EEG
picks_eda = mne.pick_channels(ch_names = raw.ch_names ,include=['EDA'])
picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False)

# Clean data 

# 1)  Transform EDA (depending on recording procedure) --> 
#     http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html

if int(number_subject) < 23:
    raw.apply_function(fun=lambda x: x/1000, picks=picks_eda)
else:
    raw.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)

# 2) Clean signals --> 
#    -  Negative values            ==> subjects 01 02 03 08 14 15
def transform_negative_to_zero(x):
    x[x<0] = 0
    return x

raw.apply_function(fun=lambda x: transform_negative_to_zero(x), picks=picks_eda)

#    -  Out-of-range values        ==> 26
def out_of_range(x):
    x[x>40] = 0 # set 40 uS ad hoc
    return x

#raw.apply_function(fun=lambda x: out_of_range(x), picks=picks_eda)

# Filter EDA 
raw.filter(0.05, 5., fir_design='firwin', picks=picks_eda)

# FIlter EEG
raw.filter(0.1, 120., fir_design='firwin', picks=picks_eeg)

# Downsample to 250 Hz --> to reduce computation time. Is it necessary?
raw.resample(250.) 

# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=10.0, overlap= 2.0)
# 3 values:
#  1 - sample number
#  2 - what the event code was on the immediately preceding sample.
#      In practice, that value is almost always 0, but it can be used to detect the
#      endpoint of an event whose duration is longer than one sample.
#  3 - integer event code --> always 1 because there are not different stim values

# Epoch length is 1.5 second
epochs = Epochs(raw=raw, events=events, tmin=0., tmax=0., baseline=None)
#eda_epochs = Epochs(raw=raw_eda, events=events, tmin=0., tmax=0., baseline=None)

# Autoreject 
reject = get_rejection_threshold(epochs, decim=2)
# reject epochs
epochs.drop_bad(reject=reject)

# Prepare classification
X = raw_epochs.get_data(picks=picks_eeg)
#y = eda_epochs.get_data().var(axis=2)[:, 0]  # target is EDA power
y = raw_epochs.get_data(picks=picks_eda)

# Classification pipeline with SPoC spatial filtering and Ridge Regression
spoc = SPoC(n_components=2, log=True, reg='oas', rank='full')
clf = make_pipeline(spoc, Ridge())
# Define a two fold cross-validation
cv = KFold(n_splits=2, shuffle=False)

# Run cross validaton
y_preds = cross_val_predict(clf, X, y, cv=cv)

# Plot the True EMG power and the EMG power predicted from MEG data
fig, ax = plt.subplots(1, 1, figsize=[10, 4])
times = raw.times[eeg_epochs.events[:, 0] - raw.first_samp]
ax.plot(times, y_preds, color='b', label='Predicted EMG')
ax.plot(times, y, color='r', label='True EMG')
ax.set_xlabel('Time (s)')
ax.set_ylabel('EDA Power')
ax.set_title('SPoC EEG Predictions')
plt.legend()
mne.viz.tight_layout()
plt.show()
