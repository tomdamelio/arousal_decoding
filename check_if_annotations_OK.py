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
from subject_number import subject_number

#%%
# Read annotated original files
directory_2 = 'outputs/data/data_with_annotations/'
extension_2 = '.fif'
subject_n = '01' 
fname_7 = op.join(directory_2,'s'+ subject_n + 'annotated' + extension_2)
raw = mne.io.read_raw_fif(fname_7, preload=True) 

raw.info

if int(subject_n) < 24:
    mapping = { 1: 'rating_screen',
                2: 'video_synch',
                3: 'fixation_screen ',
                4: 'music_stim',
                5: 'fixation_screen_after',
                7: 'end_exp'}
else:
    mapping = { 1638145: 'rating_screen',
                1638149: 'fixation_screen_after ',
                1638147: 'fixation_screen',
                1638148: 'music_stim',
                1638146: 'video_synch',
                1638151: 'end_exp',
                }

events = mne.find_events(raw)

annot_from_events = mne.annotations_from_events(
    events=events, event_desc=mapping, sfreq=raw.info['sfreq'],
    orig_time=raw.info['meas_date'])

annot_from_events.append(raw.annotations.onset,
                         raw.annotations.duration,
                         raw.annotations.description)

raw.set_annotations(annot_from_events)

%matplotlib

raw.plot()

#%%
# Read annotation files and merge with files

directory_2 = 'data/'
extension_2 = '.bdf'
subject_n = '01' 
fname_7 = op.join(directory_2,'s'+ subject_n + extension_2)
raw = mne.io.read_raw_bdf(fname_7, preload=True) 


#%%
extension = '.fif'
directory = 'outputs/data/annotations_bad_no_stim+bad_resp/'
fname_5 = op.join(directory,'annotations_s'+ '01' + extension)
annot_from_file = mne.read_annotations(fname = fname_5, sfreq=512.)
raw.set_annotations(annot_from_file)

#%%
# Check BIDS eeg files
directory_3 = 'outputs/DEAP-bids/sub-30/eeg/'
extension_3 = '.bdf'
subject_n = '30' 
fname_8 = op.join(directory_3,'sub-'+ subject_n + '_task-rest_eeg' +extension_3)
raw = mne.io.read_raw_bdf(fname_8, preload=True) 

#%%
read_raw_bids(bids_path=bids_path)