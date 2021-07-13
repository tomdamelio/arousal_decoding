import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath

from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

from meegpowreg import make_filter_bank_regressor, make_filter_bank_transformer
import  meegpowreg
from subject_number import subject_number as subjects
from numpy.lib.npyio import save
import json
from mne.decoding import SPoC
from joblib import Parallel, delayed

import DEAP_BIDS_config_emg as cfg

derivative_path = cfg.deriv_root
N_JOBS = cfg.N_JOBS

def read_bids_epochs (subject):
    # set epochs paths in BIDS format
    epochs_path = BIDSPath(subject=subject,
                            task='rest',
                            datatype='eeg',
                            root=derivative_path,
                            processing='clean',
                            extension='.fif',
                            suffix='epo',
                            check=False)
    # read epochs
    epochs = mne.read_epochs(epochs_path)
    # pick only eeg data
    epochs.pick_types(eeg=True)
    return epochs

def _compute_covs(subject, freqs):
    
    '''
    compute covariance for a single epoch
    
    Parameters
    ----------
    subject: int or list. Subject from whom it will be extracted the epochs.
    freqs: frequency bands
    
    Return
    ------
    Computed covariances in one epoch
    
    '''
    # read all epochs
    epochs_all = read_bids_epochs(subject)
    
    if DEBUG:
        epochs_all = epochs_all[:30]
        
    covs = list()
    for ii in range(len(epochs_all)):
        features, info = meegpowreg.compute_features(
            epochs_all[ii], features="covs", frequency_bands=freqs)
        covs.append([c for c in features['covs']])
        
    return covs

DEBUG = False

freqs = {'low': (0.1, 1.5),
         'delta': (1.5, 4.),
         'theta': (4., 8.),
         'alpha': (8., 15.),
         'beta_l': (15., 26.),
         'beta_h': (26., 35.),
         'gamma': (35., 49.)}

if DEBUG:
    subjects = subjects[:2]
    
out = Parallel(n_jobs=N_JOBS)(
    delayed(_compute_covs)(subject = subject, freqs=freqs)
    for subject in subjects)

for sub, dd in zip(subjects, out):
    mne.externals.h5io.write_hdf5(
        op.join(derivative_path, 'sub-' + sub , 'eeg', 'sub-' + sub + '_covariances_emg.h5'), dd,
        overwrite=True)
    

