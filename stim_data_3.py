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
    subjects = subjects[:1]

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
            stim_dict[str(music_stim)].append(epoch_idx)  
            
    all_subs_epochs_selected[str(subject)] = stim_dict
        

# Transform dict in df
df_epochs=pd.DataFrame.from_dict(all_subs_epochs_selected,orient='index')

# set 'subject' column based on index
df_epochs['subject'] = df_epochs.index

# set 'subject' as first column in df_epochs
first_column = df_epochs.pop('subject')
df_epochs.insert(0, 'subject', first_column)

# Set stimuli as row (similar to gather in R)
df_epochs = pd.melt(df_epochs, id_vars="subject")

# Rensame column 'variable'
df_epochs = df_epochs.rename(columns={'variable': 'music_stim'})

# divide list in column 'value' to multiple  rows
df_epochs = df_epochs.explode('value')

df_epochs['id'] = df_epochs['subject'].astype(str) + '-' + df_epochs['value'].astype(str)

# Create a dataframe with filtered epochs

appended_filt_epochs = []

for subject in subjects:
    # Read EDA(y) data
    epochs_path = BIDSPath(subject= subject, 
                           task='rest',
                           datatype='eeg',
                           root=derivative_path,
                           processing='clean',
                           extension='.fif',
                           suffix='epo',
                           check=False)
    # read epochs
    epochs = mne.read_epochs(epochs_path)
   
    epochs_selected = epochs.selection
    
    df_epochs_filtered_aux = pd.DataFrame(
    [subject, idx_epoch ] for idx_epoch in epochs_selected)
    
    appended_filt_epochs.append(df_epochs_filtered_aux)
    
df_epochs_filtered = pd.concat(appended_filt_epochs)

df_epochs_filtered = df_epochs_filtered.rename(columns={0: 'subject',
                                                         1: 'value'})

df_epochs_filtered['id'] = df_epochs_filtered['subject'].astype(str) + '-' + df_epochs_filtered['value'].astype(str)
    

df_epochs_final = df_epochs.merge(df_epochs_filtered, on ='id', how='right')

  
df_epochs_final = df_epochs_final.drop(columns=['subject_y', 'value_y', 'id'])

df_epochs_final = df_epochs_final.rename(columns={'subject_x': 'Participant_id',
                                                  'value_x': 'idx_epoch',
                                                  'music_stim': 'Trial'})

# drop Nans
df_epochs_final = df_epochs_final.dropna() 

df_epochs_final.Participant_id = df_epochs_final.Participant_id.astype(int)


df_epochs_final['id_ratings'] = df_epochs_final['Participant_id'].astype(str) + '-' + df_epochs_final['Trial'].astype(str)

# read subjects' arousal self reports
data_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/data'  
fname_ratings = op.join(data_path,'participant_ratings.csv')
df_ratings = pd.read_csv(fname_ratings)  
df_ratings['id_ratings'] = df_ratings['Participant_id'].astype(str) + '-' + df_ratings['Trial'].astype(str)

# join epochs and ratings
df_epochs_ratings = df_epochs_final.merge(df_ratings, on ='id_ratings', how='left', suffixes=('', '_y'))
df_epochs_ratings.drop(df_epochs_ratings.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

# export epochs and ratings
fname_rating = op.join(derivative_path,'epochs_and_ratings.csv')
df_epochs_ratings.to_csv(fname_rating, index=False)

# %%
