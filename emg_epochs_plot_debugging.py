#%%
import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.npyio import save
import json

import mne
from mne.decoding import SPoC
from mne_bids import BIDSPath

from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, Ridge

from meegpowreg import make_filter_bank_regressor, make_filter_bank_transformer
from subject_number import subject_number as subjects

measure = 'emg'

if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg

DEBUG = False

derivative_path = cfg.deriv_root
n_jobs = 15
score_name, scoring = "r2", "r2"
cv_name = '2Fold'

if DEBUG:
   n_jobs = 4
   subjects = subjects[16:]
   subject = '01'
   debug_out = '_DEBUG'
else:
   debug_out = ''


for subject in subjects:
    
    if os.name == 'nt':
        fname_epochs = derivative_path / 'clean-epo-files'
        epochs = mne.read_epochs(op.join(fname_epochs, 'sub-' + subject +
                                         '_task-rest_proc-clean_epo.fif'))

    else: 
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
    
    
    if measure == 'emg':
        picks_emg = mne.pick_types(epochs.info, emg=True)
        epochs = epochs.filter(20., 30., picks=picks_emg)
        emg_epochs = epochs.copy().pick_channels(['EXG5','EXG6'])
        y = emg_epochs.get_data().var(axis=2).mean(axis=1)
        
    # Plot the True EDA power and the EDA predicted from EEG data
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    #times = raw.times[epochs.events[:, 0] - raw.first_samp]
    times = [i for i in range(len(epochs))]
    ax.plot(times, y, color='r', label=f'True {measure}')
    ax.set_xlabel('Time (epochs)')
    if measure == 'emg':
        ax.set_ylabel(f'Var(emg)')
    else:
        ax.set_ylabel(f'Mean(eda)')
    ax.set_title(f'Subject {subject} - {measure} Prediction (SPoC) - whitout meegpowreg pipeline')
    plt.legend()
    if os.name == 'nt':
        plt.savefig(op.join(derivative_path, 'emg_plot_debugging--16-07',  'sub-' + subject +
                            f'_emg_epochs_plot_debugging.png'))
    else:
        plt.savefig(op.join(derivative_path, 'emg_plot_debugging--16-07', 'sub-' + subject +
                        f'_emg_epochs_plot_debugging.png'))
        
    epochs = epochs[600:800]
    
    if measure == 'emg':
        picks_emg = mne.pick_types(epochs.info, emg=True)
    epochs = epochs.filter(20., 30., picks=picks_emg)
    emg_epochs = epochs.copy().pick_channels(['EXG5','EXG6'])
    y = emg_epochs.get_data().var(axis=2).mean(axis=1)
        
    # Plot the True EDA power and the EDA predicted from EEG data
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    #times = raw.times[epochs.events[:, 0] - raw.first_samp]
    times = [i for i in range(len(epochs))]
    ax.plot(times, y, color='r', label=f'True {measure}')
    ax.set_xlabel('Time (epochs)')
    if measure == 'emg':
        ax.set_ylabel(f'Var(emg)')
    else:
        ax.set_ylabel(f'Mean(eda)')
    ax.set_title(f'Subject {subject} - {measure} Prediction (SPoC) - whitout meegpowreg pipeline')
    plt.legend()
    if os.name == 'nt':
        plt.savefig(op.join(derivative_path, 'emg_plot_debugging--16-07',  'sub-' + subject +
                            f'_emg_epochs_plot_debugging_200_epochs.png'))
    else:
        plt.savefig(op.join(derivative_path, 'emg_plot_debugging--16-07', 'sub-' + subject +
                        f'_emg_epochs_plot_debugging_200_epochs.png'))
    
    
