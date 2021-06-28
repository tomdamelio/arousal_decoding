#%%
import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, GroupShuffleSplit, cross_val_predict)
from meegpowreg import make_filter_bank_regressor
from subject_number import subject_number as subjects
from joblib import Parallel, delayed

import DEAP_BIDS_config_emg as cfg

derivative_path = cfg.deriv_path
if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  


DEBUG = False

n_components = np.arange(1, 31, 1)
seed = 42
n_splits = 2
n_jobs = 10
score_name, scoring = "r2", "r2"
cv_name = '2Fold'
freqs = {"beta": (15.0, 30.0)}

# Creation of the pipelines of interest
pipelines = {'riemann': make_filter_bank_regressor(
                names=freqs.keys(),
                method='riemann',
                projection_params=dict(scale='auto', reg=1.e-05, n_compo = 31),
                vectorization_params=dict(metric='riemann')),
             'spoc': make_filter_bank_regressor(
                names=freqs.keys(),
                method='spoc',
                projection_params=dict(scale='auto', reg=0,
                                       shrink=0.5, n_compo = 3),
                vectorization_params=None),
             'log_diag': make_filter_bank_regressor(
                names=freqs.keys(),
                method='log_diag',
                projection_params=None,
                vectorization_params=None),
             'upper': make_filter_bank_regressor(
                names=freqs.keys(),
                method='naive',
                projection_params=None,
                vectorization_params=None),
             'random': make_filter_bank_regressor(
                names=freqs.keys(),
                method='random',
                projection_params=None,
                vectorization_params=None)}


if DEBUG:
    N_JOBS = 1
    subjects = subjects[:2]

def run_low_rank(n_components, X, y, cv, estimators, scoring):   
    out = dict(n_components=n_components)
    for key, estimator in pipelines.items():
        this_est = estimator
        pipelines['riemann'].steps[0][1].transformers[0][1].steps[0][1].n_compo = n_components
        scores = cross_val_score(X=df_features, y=y, estimator=estimator,
                                cv=cv, n_jobs=min(n_splits, n_jobs),
                                scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
            print(scores)
        out[key] = scores
    return out

for subject in subjects:
    fname = op.join(derivative_path, 'sub-' + subject, 'eeg', 'sub-' + subject + '_covariances_emg.h5')
    covs = mne.externals.h5io.read_hdf5(fname)
 
    X_cov = np.array([cc for cc in covs])
    df_features = pd.DataFrame(
       {band: list(X_cov[:, ii]) for ii, band in
       enumerate(freqs)})
 
    # Read EMG(y) data
 
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
    
    picks_emg = mne.pick_types(epochs.info, emg=True)
    epochs.filter(20., 30., picks=picks_emg)
    
    if DEBUG:
       epochs = epochs[:30]
    
    # How are we going to model our target? -> Mean of two EMG Trapezius sensors
    emg_epochs = epochs.copy().pick_channels(['EXG7','EXG8'])
    y = emg_epochs.get_data().var(axis=2).mean(axis=1)
  
    low_rank_estimators = {k: v for k, v in pipelines.items()
                         if k in ('spoc', 'riemann')}
    
    cv = KFold(n_splits=n_splits)
    out_list = Parallel(n_jobs=n_jobs)(delayed(run_low_rank)(
                      n_components=cc, X=df_features, y=y,
                      cv=cv, estimators=low_rank_estimators, scoring='r2')
                      for cc in n_components)
    
    out_frames = list()
    for this_dict in out_list:
        this_df = pd.DataFrame({'spoc': this_dict['spoc'],
                                'riemann': this_dict['riemann']})
        this_df['n_components'] = this_dict['n_components']
        this_df['fold_idx'] = np.arange(len(this_df))
        out_frames.append(this_df)
    out_df = pd.concat(out_frames)
    out_df.to_csv("./outputs/DEAP_component_scores.csv")
 
    mean_df = out_df.groupby('n_components').mean().reset_index()
    best_components = {
       'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
       'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
    }
    
     
    pipelines[f"spoc_{best_components['spoc']}"] = make_filter_bank_regressor(
                                        names=freqs.keys(),
                                        method='spoc',
                                        projection_params=dict(scale='auto', reg=0,
                                                            shrink=0.5, n_compo = best_components['spoc']))

    pipelines[f"spoc_{best_components['riemann']}"] = make_filter_bank_regressor(
                names=freqs.keys(),
                method='riemann',
                projection_params=dict(scale='auto', reg=1.e-05, n_compo = best_components['riemann']),
                vectorization_params=dict(metric='riemann'))
 
    all_scores = dict() 
    for key, estimator in pipelines.items():
       #cv = KFold(n_splits=n_splits)
       scores = cross_val_score(X=df_features, y=y, estimator=estimator,
                               cv=cv, n_jobs=min(n_splits, n_jobs),
                               scoring=scoring)
       if scoring == 'neg_mean_absolute_error':
          scores = -scores
          print(scores)
       all_scores[key] = scores
 
 
    np.save(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                   f'_all_scores_models_DEAP_emg_{score_name}_{cv_name}.npy'),
          all_scores)
 
# %%
