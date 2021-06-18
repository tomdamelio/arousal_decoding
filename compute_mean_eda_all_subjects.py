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



DEBUG = False

if DEBUG:
    N_JOBS = 1
    subjects = subjects[:2]

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

mean_eda_all_subjects = dict()
for subject in subjects:
    fname = op.join(derivative_path, 'sub-' + subject, 'eeg', 'sub-' + subject + '_covariances.h5')

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

    # keep only EDA data
    eda_epochs = epochs.copy().pick_channels(['EDA'])

    y = eda_epochs.get_data().mean(axis=2)[:, 0]  

    mean_eda_all_subjects[subject] = y

#%%    
# Save dict with  mean(EDA) for every subject
fname_out = op.join(derivative_path, 'mean_eda_all_subs.npy')
np.save(fname_out, mean_eda_all_subjects)

#%%
# read dict with mean(EDA) for every subjects
mean_eda_all_subjects_npy = np.load(fname_out, allow_pickle=True)
#mean_eda_all_subjects_npy[()]['01']
# %%
