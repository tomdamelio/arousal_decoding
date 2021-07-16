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
   subjects = subjects[:3]
   subject = '01'
   debug_out = '_DEBUG'
else:
   debug_out = ''


for subject in subjects:
    
    if os.name == 'nt':
        fname_epochs = derivative_path / 'clean-epo-files'
        epochs = mne.read_epochs(op.join(fname_epochs, 'sub-' + subject + '_task-rest_proc-clean_epo.fif'))

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
    
    if DEBUG:
        epochs = epochs[:30]

    if measure == 'emg':
        picks_emg = mne.pick_types(epochs.info, emg=True)
        epochs = epochs.filter(20., 30., picks=picks_emg)
        emg_epochs = epochs.copy().pick_channels(['EXG7','EXG8'])
        y = emg_epochs.get_data().var(axis=2).mean(axis=1)
    else:
        picks_eda = mne.pick_channels(ch_names = epochs.ch_names ,include=['EDA'])       
        if int(subject) < 23:
            epochs.apply_function(fun=lambda x: x/1000, picks=picks_eda)
        else:
            epochs.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)
        eda_epochs = epochs.copy().pick_channels(['EDA'])
        
    
    picks_eeg = mne.pick_types(epochs.info, eeg=True, eog=False)    
    X = epochs.get_data(picks=picks_eeg)   

    # Classification pipeline with SPoC spatial filtering and Ridge Regression
    spoc = SPoC(n_components=2, log=True, reg='oas', rank='full')
    clf = make_pipeline(spoc, Ridge())
    # Define a two fold cross-validation
    cv = KFold(n_splits=2, shuffle=False)
    cv_name = '2Fold'
    # Run cross validaton
    y_preds = cross_val_predict(clf, X, y, cv=cv)
    scores = cross_val_score(X=X, y=y, estimator=clf, cv=cv, scoring='r2')

    # Plot the True EDA power and the EDA predicted from EEG data
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    #times = raw.times[epochs.events[:, 0] - raw.first_samp]
    times = [i for i in range(len(epochs))]
    ax.plot(times, y, color='r', label=f'True {measure}')
    ax.plot(times, y_preds, color='b', label=f'Predicted {measure}')
    ax.set_xlabel('Time (s)')
    if measure == 'emg':
        ax.set_ylabel(f'Var(emg)')
    else:
        ax.set_ylabel(f'Mean(eda)')
    ax.set_title(f'Subject {subject} - {measure} Prediction (SPoC) - whitout meegpowreg pipeline')
    plt.legend()
    if os.name == 'nt':
        np.save(op.join(derivative_path,  'sub-' + subject +
                   f'_scores_{measure}_' + score_name + '_NO_OPT_PIPELINE_spoc_' + cv_name + f'{debug_out}.npy'),
          scores)  
        plt.savefig(op.join(derivative_path,  'sub-' + subject +
                            f'_plot_NO_OPT_PIPELINE_spoc_DEAP_{measure}_{cv_name}_{debug_out}.png'))
    else:
        np.save(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                   f'_scores_{measure}_' + score_name + '_NO_OPT_PIPELINE_spoc_' + cv_name + f'{debug_out}.npy'),
          scores)
        plt.savefig(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                              f'_plot_NO_OPT_PIPELINE_spoc_DEAP_{measure}_{cv_name}_{debug_out}.png'))
