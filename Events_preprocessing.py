# Mark and then remove all the events that don't correspond to stimulus
# presentation (e.g. inter-block interval, self-reporting, etc).

'''
Steps:
1) Plot one subject with stims
2) Understand code of stims
3) mark bad epochs as:
    a)  bad_inter_block_inteval
    b)  bad_self_report
    c)  bad_eda_artifact
4) Save in other folder this step
5) Insert in function run_all_subjects clean subjects

'''
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
# Read .fif files (with respiration annotations)
directory = 'outputs/data/clean_EDA/'
number_subject = '01'
events_label = '_without_ITI'
extension = '.fif'
fname = op.join(directory + 's'+ number_subject + extension)

raw_fif = mne.io.read_raw_fif(fname, preload=True) 

# Plot EDA and resp with annotations
# raw_fif.plot(order=[1,0], scalings=dict(misc='1e-1', emg='1e-1'))
# %matplotlib


# Read bdf files (without annotations)
extension = '.bdf'
directory = 'data/'
#Read bdf
fname_bdf = op.join(directory + 's'+ number_subject + extension)
raw_bdf = mne.io.read_raw_bdf(fname_bdf, preload=True) 

mne.rename_channels(info= raw_bdf.info , mapping={'GSR1':'EDA'})

raw_bdf.set_channel_types({ 'EXG1': 'eog',
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

# Pick STIM, EEG and EOG from .bdf
raw_bdf2 = raw_bdf.pick_types(stim = True, eeg = True, eog = True,
                              misc = False, emg = False)

# Add channels from raw_bdf (stim, EEG and EOG) to raw_fif (resp and EDA)
raw_fif.add_channels([raw_bdf2])

#%%
# Save .fif file
extension = '.fif'
directory = 'outputs/data/EDA+EEG'
fname_2 = op.join(directory,'s'+ number_subject + extension)
raw_fif.save(fname = fname_2, overwrite=True)

#%%
# Create events based on stim channel
events = mne.find_events(raw_fif, stim_channel='Status')

# Select events with stim value == 4 --> start music stimulus
rows=np.where(events[:,2]==4)
events_4 = events[rows]

#%%
#
mapping = {4: 'music_stim'}
annot_from_events = mne.annotations_from_events(
    events=events_4, event_desc=mapping, sfreq=raw_fif.info['sfreq'],
    orig_time=raw_fif.info['meas_date'])
raw_fif.set_annotations(annot_from_events)

# Epoche considering this epochs, from -5 secs (fix) to 63 sec (stim + postfix)
#epochs = Epochs(raw=raw_fif, events=events_4, tmin=-5., tmax=63., baseline=None)








######################
#%%
events_from_annot, event_dict = mne.events_from_annotations(raw_bdf)
print(event_dict)
print(events_from_annot[:5])


#%%
# Find events with status code == 4
events = mne.find_events(raw_bdf, stim_channel='Status')
epochs = mne.Epochs(raw_bdf, events=events, event_id=None, tmin=-5., tmax=60.)
epochs['4']


#%%
# Clean raw based on events




#%%



#%%
# Save output fif file
fname2 = op.join(directory + 's'+ number_subject + events_label + extension)
raw.save(fname = fname2, overwrite=True)
