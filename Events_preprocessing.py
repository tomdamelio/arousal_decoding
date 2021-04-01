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

directory = 'outputs/data/'
number_subject = '01'
events_label = '_without_ITI'
extension = '.fif'
fname = op.join(directory + 's'+ number_subject + extension)

#%%
# Extract signal
raw_fif = mne.io.read_raw_fif(fname, preload=True) 
#raw_fif.plot(order=[1,0], scalings=dict(misc='1e-1', emg='1e-1'))

#%%
extension = '.bdf'
directory = 'data/'
#Read bdf
fname_bdf = op.join(directory + 's'+ number_subject + extension)
raw_bdf = mne.io.read_raw_bdf(fname_bdf, preload=True) 

#raw_bdf.pick_channels(['EDA', 'Status'])
# Add annotation to complete bdf file
# No agregaga annotations a raw_bdf. Intentar en cambio sumar todos los canales al fif.
#raw_bdf.annotations.__add__(raw_fif.annotations)

raw_fif = raw_fif.add_channels()

%matplotlib

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
