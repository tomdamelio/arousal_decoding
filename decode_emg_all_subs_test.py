#%%
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
from sklearn.linear_model import RidgeCV, Ridge

from meegpowreg import make_filter_bank_regressor, make_filter_bank_transformer
from subject_number import subject_number as subjects
from numpy.lib.npyio import save
import json
from mne.decoding import SPoC


DEBUG = True

if DEBUG:
    N_JOBS = 1
    subjects = subjects[2:]

N_JOBS = 4

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
    clean_epochs = 'emg-clean-epo-files-drago'
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

for subject in subjects:
    
    fname = f'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline/emg-clean-epo-files-drago/sub-{subject}_task-rest_proc-clean_epo.fif'
    # read epochs
    epochs = mne.read_epochs(fname)

    picks_emg = mne.pick_types(epochs.info, emg=True)
    picks_eeg = mne.pick_types(epochs.info, eeg=True, eog=False)

    epochs.filter(20., 30., picks=picks_emg)
   
#    if DEBUG:
#        epochs = epochs[:30]
    
    X = epochs.get_data(picks=picks_eeg)
    # target is emg var. We can then change last number for [1,2,3] as we have 4 emg channels.
    # We can also calculate averages on this sensors.
    y = epochs.get_data(picks=picks_emg).var(axis=2)[:, 0] 

    # Classification pipeline with SPoC spatial filtering and Ridge Regression
    spoc = SPoC(n_components=2, log=True, reg='oas', rank='full')
    clf = make_pipeline(spoc, Ridge())
    # Define a two fold cross-validation
    cv = KFold(n_splits=2, shuffle=False)
    cv_name = '2Fold'
    # Run cross validaton
    y_preds = cross_val_predict(clf, X, y, cv=cv)

    # Plot the True EDA power and the EDA predicted from EEG data
    fig, ax = plt.subplots(1, 1, figsize=[10, 4])
    #times = raw.times[epochs.events[:, 0] - raw.first_samp]
    times = [i for i in range(len(epochs))]
    ax.plot(times, y, color='r', label='True EMG')
    ax.plot(times, y_preds, color='b', label='Predicted EMG')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Var(emg)')
    ax.set_title(f'EMG Prediction (SPoC) - Subject {subject}')
    plt.legend()
    mne.viz.tight_layout()
    plt.savefig(op.join(derivative_path, clean_epochs, 'outputs',
                f'plot_NO_OPT_PIPELINE_spoc_DEAP_emg_{subject}_{cv_name}.png'))

    scores = cross_val_score(X=X, y=y, estimator=clf, cv=cv, scoring='r2')
    np.save(op.join(derivative_path, clean_epochs, 'outputs',
                f'scores_NO_OPT_PIPELINE_spoc_DEAP_emg_{subject}_{cv_name}.npy'),
                scores)
# %%
subject = '01'
x = np.load(op.join(derivative_path, clean_epochs, 'outputs',
                f'scores_NO_OPT_PIPELINE_spoc_DEAP_emg_{subject}_2Fold.npy'),
            allow_pickle=True)

# %%
