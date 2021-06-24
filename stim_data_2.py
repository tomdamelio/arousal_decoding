#%%
import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold)
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from meegpowreg import make_filter_bank_regressor, make_filter_bank_transformer
from subject_number import subject_number as subjects
from numpy.lib.npyio import save
import json
from collections import defaultdict
import pathlib

DEBUG = False

if DEBUG:
    N_JOBS = 1
    subjects = subjects[12:18]

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
    annotations_root = pathlib.Path(
        "~/OneDrive/Escritorio/tomas_damelio/outputs/data/annotations_bad_no_stim").expanduser()
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

all_subs_epochs_selected = dict()

for subject in subjects:
    
    raw_path = BIDSPath(subject= subject, 
                           task='rest',
                           datatype='eeg',
                           root=derivative_path,
                           extension='.fif',
                           suffix='raw',
                           check=False)
    # Read EDA data
    raw = mne.io.read_raw_fif(raw_path)
    
    # Pick only EDA data
    raw_eda = raw.copy().pick_channels(['EDA'])
    
    # Add annotations (bad_no_stim) to data
    fname_annot = op.join(annotations_root,'sub-'+ subject + '_annotations.fif')
    annot_from_file = mne.read_annotations(fname=fname_annot, sfreq=raw.info['sfreq'])
    raw.set_annotations(annot_from_file)
    
    events = mne.make_fixed_length_events(
        raw, id=3000, start=0,
        duration=5.,
        overlap=0.)
    event_id = dict(rest=3000)
    
    metadata, _, _ = mne.epochs.make_metadata(
        events=events, event_id=event_id,
        tmin=0., tmax =5.,
        sfreq=raw.info['sfreq'])
    
    epochs = mne.Epochs(raw, events=events, event_id=event_id,
                    metadata=metadata, reject_by_annotation=True,
                    baseline=None)
    
    epochs.drop_bad()

    epochs_selected = epochs.selection
    stim_dict =  defaultdict(list)
    music_stim = 1 

    for epoch_idx in epochs_selected:
        if str(music_stim) not in stim_dict:
            stim_dict[str(music_stim)].append(epoch_idx)        
        elif len(stim_dict.get(str(music_stim))) == 1:
            stim_dict[str(music_stim)].append(epoch_idx)
        elif (epoch_idx-stim_dict.get(str(music_stim))[0]) <= 12:
            stim_dict[str(music_stim)].append(epoch_idx)    
        else:
            music_stim += 1
            
    all_subs_epochs_selected[str(subject)] = stim_dict
        
# %%
for subject in subjects:
    if len(all_subs_epochs_selected[str(subject)]) != 40:
        print('sub',subject, '->' ,len(all_subs_epochs_selected[str(subject)]))
        
#%%
for subject in subjects:
    print('sub',subject, '->' ,len(all_subs_epochs_selected[str(subject)]))
        
#%%
# Insepcciono sujetos
all_subs_epochs_selected['09']
# %%
