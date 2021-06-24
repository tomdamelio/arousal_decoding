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

DEBUG = True

if DEBUG:
    N_JOBS = 1
    subjects = subjects[:1]

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

all_subs_epochs_selected = dict()

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
# No todos los sujetos tienen 40 epocas bien distinguidas
for subject in subjects:
    if len(all_subs_epochs_selected[str(subject)]) != 40:
        print('sub',subject, '->' ,len(all_subs_epochs_selected[str(subject)]))
        
# %%
# Insepcciono sujetos
all_subs_epochs_selected['09']
# %%
