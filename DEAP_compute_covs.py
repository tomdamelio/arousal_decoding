import numpy as np
import pandas as pd
import os.path as op
import glob
from joblib import Parallel, delayed

import mne
from mne_bids import BIDSPath
import  meegpowreg

from subject_number import subject_number as subjects

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'


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
        epochs_all = epochs_all[:3]
        
    covs = list()
    for ii in range(len(epochs_all)):
        features, info = meegpowreg.compute_features(
            epochs_all[ii], features="covs", frequency_bands=freqs)
        covs.append([c for c in features['covs']])
        
    return covs

DEBUG = False
freqs = {"low": (0.1, 1.5),
         "delta": (1.5, 4.0),
         "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0),
         "beta_low": (15.0, 26.0),
         "beta_high": (26.0, 35.0),
         "gamma_low": (35.0, 49.0)}

if DEBUG:
    N_JOBS = 1
    subjects = subjects[:3]
else:
    N_JOBS = 20
    
out = Parallel(n_jobs=N_JOBS)(
    delayed(_compute_covs)(subject = subject, freqs=freqs)
    for subject in subjects)

for sub, dd in zip(subjects, out):
    mne.externals.h5io.write_hdf5(
        op.join(derivative_path, 'sub-' + ''.join(sub), 'eeg', 'sub-' + ''.join(sub) + '_covariances.h5'), dd,
        overwrite=True)

