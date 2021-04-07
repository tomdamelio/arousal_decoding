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
import os.path as op


#%%
# Define subject
number_subject = '32'
directory = 'outputs/data'
extension = '.bdf'
#Extract signal
fname = op.join(directory, 's'+ number_subject + extension)
raw = mne.io.read_raw_bdf(fname, preload=True)

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
                       'Resp': 'emg',
                       'Plet': 'misc',
                       'Temp': 'misc'})

# Pick EDA and EEG
picks_eda = mne.pick_channels(ch_names = raw.ch_names ,include=['EDA'])

if int(number_subject) < 23:
    raw.apply_function(fun=lambda x: x/1000, picks=picks_eda)
else:
    raw.apply_function(fun=lambda x: (10**9/(x*1000)), picks=picks_eda)

%matplotlib

#%%
# Preprocessing EDA
# Filter EDA
raw.filter(None, 5., fir_design='firwin', picks=picks_eda)

# Preprocessing Respiration
picks_resp = mne.pick_channels(ch_names = raw.ch_names ,include=['Resp'])
# Filter resp
raw.filter(None, 5., fir_design='firwin', picks=picks_resp)

#%%
eda_idx = raw.ch_names.index('EDA')
x = raw.get_data()[eda_idx]
plt.plot(x)

#%%
raw.pick_channels(['EDA', 'Resp'])
raw.plot(color = dict(emg='m', misc='k'), order=[1,0],
         scalings=dict(misc='1e-1', emg='auto'))

#%%
extension = '.fif'
fname = op.join('s'+ number_subject + extension)
raw.save(fname = fname, overwrite=True)

#%%
# Extract signal

raw_2 = mne.io.read_raw_fif(fname, preload=True) 
raw_2.plot(order=[1,0], scalings=dict(misc='1e-1', emg='1e-1'))
# %%
