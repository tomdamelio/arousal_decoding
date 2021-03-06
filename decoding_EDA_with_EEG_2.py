# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
#         Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)
# Link https://mne.tools/dev/auto_examples/decoding/plot_decoding_spoc_CMC.html#sphx-glr-auto-examples-decoding-plot-decoding-spoc-cmc-py
#%%
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import Epochs
from mne.decoding import SPoC
from mne.datasets.fieldtrip_cmc import data_path
from mne import pick_types

from autoreject import get_rejection_threshold
from autoreject.autoreject import _GlobalAutoReject
from autoreject.bayesopt import expected_improvement, bayes_opt

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, check_cv

from preprocessing import extract_signal, transform_negative_to_zero, out_of_range, get_rejection_threshold
from channel_names import channels_geneva, channels_twente 

# Define subject
number_subject = '01'

#Extract signal
raw = extract_signal(directory = 'data', number_subject=number_subject,
                     extension = '.bdf')
if int(number_subject)>24:
    raw.drop_channels('')
# Rename channel EDA and set GSR as channel type
raw.rename_channels(mapping={'GSR1':'EDA'})
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
raw.apply_function(fun=transform_negative_to_zero, picks=picks_eda)

#    -  Out-of-range values        ==> 26
#raw.apply_function(fun=out_of_range, picks=picks_eda)

# Filter EDA 
raw.filter(0.05, 5., fir_design='firwin', picks=picks_eda)

# FIlter EEG
raw.filter(8., 13., fir_design='firwin', picks=picks_eeg)
# Downsample to 250 Hz 
#raw.resample(250.) 

#%%
# Build epochs as sliding windows over the continuous raw file
events_reject = mne.make_fixed_length_events(raw, id=1, duration=5., overlap=0.)

epochs_reject = Epochs(raw, events_reject, tmin=0., tmax=5., baseline=None)
#eda_epochs = Epochs(raw=raw_eda, events=events, tmin=0., tmax=0., baseline=None)

# Autoreject 
reject = get_rejection_threshold(epochs_reject, decim=4)

#reject.update({'misc': '3.'}) # 3 times typical phasic incrase in conductance (Boucsein, 2012)

# events = events_reject
events = mne.make_fixed_length_events(raw, id=1, duration=10.0, overlap=2.0)
# epochs = epochs_reject
epochs = Epochs(raw=raw, events=events, tmin=0., tmax=10., baseline=None)

# Reject bad epochs
epochs.drop_bad(reject={k: v for k, v in reject.items() if k != "misc"})

#%%
# Prepare classification
X = epochs.get_data(picks=picks_eeg)
#y = eda_epochs.get_data().var(axis=2)[:, 0]  # target is EDA power
y = epochs.get_data(picks=picks_eda).mean(axis=2)[:, 0]

# Classification pipeline with SPoC spatial filtering and Ridge Regression
spoc = SPoC(n_components=15, log=True, reg='oas')
clf = make_pipeline(spoc, Ridge())
# Define a two fold cross-validation
cv = KFold(n_splits=2, shuffle=False)

#%%
# Run cross validaton
y_preds = cross_val_predict(clf, X, y, cv=cv)

# Plot the True EDA power and the EDA predicted from EEG data
fig, ax = plt.subplots(1, 1, figsize=[10, 4])
times = raw.times[epochs.events[:, 0] - raw.first_samp]
ax.plot(times, y_preds, color='b', label='Predicted EDA')
ax.plot(times, y, color='r', label='True EDA')
ax.set_xlabel('Time (s)')
ax.set_ylabel('EDA average')
ax.set_title('SPoC EEG Predictions')
plt.legend()
mne.viz.tight_layout()
plt.show()


# %%
