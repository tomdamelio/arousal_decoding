#%%

import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from sklearn.linear_model import RidgeCV, GammaRegressor
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, GroupShuffleSplit, cross_val_predict)
from meegpowreg import make_filter_bank_regressor
from subject_number import subject_number as subjects
from joblib import Parallel, delayed

measure = 'emg'

if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg

DEBUG = True

date = '20-07'

derivative_path = cfg.deriv_root

n_components = np.arange(1, 32, 1)
seed = 42
n_splits = 10
n_jobs = 15
score_name, scoring = "r2", "r2"
cv_name = '10Fold'

freqs = {'low': (0.1, 1.5),
         'delta': (1.5, 4.),
         'theta': (4., 8.),
         'alpha': (8., 15.),
         'beta_l': (15., 26.),
         'beta_h': (26., 35.),
         'gamma': (35., 49.)}


pipelines = {'riemann': make_filter_bank_regressor(
                names=freqs.keys(),
                method='riemann',
                projection_params=dict(scale=1, reg=1.e-05, n_compo=31),
                vectorization_params=dict(metric='riemann'),
                estimator=GammaRegressor()),
             'spoc': make_filter_bank_regressor(
                names=freqs.keys(),
                method='spoc',
                projection_params=dict(scale='auto', reg=1.e-05, shrink=1, n_compo=31),
                vectorization_params=None,
                estimator=GammaRegressor()),
             'log_diag': make_filter_bank_regressor(
                names=freqs.keys(),
                method='log_diag',
                projection_params=None,
                vectorization_params=None,
                estimator=GammaRegressor()),
             'upper': make_filter_bank_regressor(
                names=freqs.keys(),
                method='naive',
                projection_params=None,
                vectorization_params=None,
                estimator=GammaRegressor()),
             'random': make_filter_bank_regressor(
                names=freqs.keys(),
                method='random',
                projection_params=None,
                vectorization_params=None,
                estimator=GammaRegressor())}


if DEBUG:
   n_jobs = 4
   # subs 8, 10, 12, 28
   subjects = ['10']
   subject = '10'
   debug_out = '_DEBUG'
else:
   debug_out = ''
   subjects = ['08', '10', '12', '28']

def run_low_rank(n_components, X, y, estimators, cv, scoring):   
    out = dict(n_components=n_components)
    for key, estimator in estimators.items():
        this_est = estimator
        for n_pipeline in range(len(freqs)):
            this_est.steps[0][1].transformers[n_pipeline][1].steps[0][1].n_compo = n_components
        scores = cross_val_score(X=df_features, y=y, estimator=estimator,
                                cv=cv, n_jobs=min(n_splits, n_jobs),
                                scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
            print(scores)
        out[key] = scores
    return out


#%%
for subject in subjects:
    if os.name == 'nt':
        fname_covs = op.join(derivative_path, f'{measure}-cov-matrices-all-freqs', 'sub-' + subject + f'_covariances_{measure}.h5')
    else:
        fname_covs = op.join(derivative_path, 'sub-' + subject, 'eeg', 'sub-' + subject + f'_covariances_{measure}.h5')
    
    covs = mne.externals.h5io.read_hdf5(fname_covs)
    
#    if DEBUG:
#       covs = covs[:30]
 
    X_cov = np.array([cc for cc in covs])    
    df_features = pd.DataFrame(
       {band: list(X_cov[:, ii]) for ii, band in
       enumerate(freqs)})

    # Read EMG or EDA (y) data
   
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
      
#    if DEBUG:
#        epochs = epochs[:30]
    
    if measure == 'emg':
        picks_emg = mne.pick_types(epochs.info, emg=True)
        epochs = epochs.filter(20., 30., picks=picks_emg)
        # How are we going to model our target? -> Mean of two EMG zygomaticus sensors
        emg_epochs = epochs.copy().pick_channels(['EXG5','EXG6'])
        
        y_stat = 'var'
        y = emg_epochs.get_data().var(axis=2).mean(axis=1)
    else: 
        picks_eda = mne.pick_channels(ch_names = epochs.ch_names ,include=['EDA'])       
        if int(subject) < 23:
            epochs.apply_function(fun=lambda x: x/1000, picks=picks_eda)
        else:
            epochs.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)
            
        eda_epochs = epochs.copy().pick_channels(['EDA'])
        
        y_stat = 'mean'
        y = eda_epochs.get_data().mean(axis=2)[:, 0] 
           
  
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
    
    out_df.to_csv(op.join(derivative_path, f'{measure}_opt--{date}-meegpowreg', 'sub-' + subject +
                            f'_DEAP_component_scores_{measure}{debug_out}.csv'))
 
    mean_df = out_df.groupby('n_components').mean().reset_index()
    best_components = {
       'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
       'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
    }
     
    pipelines[f"spoc_{best_components['spoc']}"] = make_filter_bank_regressor(
                                        names=freqs.keys(),
                                        method='spoc',
                                        projection_params=dict(scale='auto', reg=1.e-05,
                                                               shrink=1, n_compo= best_components['spoc']),
                                                               vectorization_params=None,
                                                               estimator=GammaRegressor())

    pipelines[f"riemann_{best_components['riemann']}"] = make_filter_bank_regressor(
                names=freqs.keys(),
                method='riemann',
                projection_params=dict(scale=1, reg=1.e-05, n_compo= best_components['riemann']),
                vectorization_params=dict(metric='riemann'),
                estimator=GammaRegressor())

    all_scores = dict() 
    for key, estimator in pipelines.items():
       scores = cross_val_score(X=df_features, y=y, estimator=estimator,
                               cv=cv, n_jobs=min(n_splits, n_jobs),
                               scoring=scoring)
       if scoring == 'neg_mean_absolute_error':
          scores = -scores
          print(scores)
       all_scores[key] = scores
 
    np.save(op.join(derivative_path, f'{measure}_scores--{date}-meegpowreg', 'sub-' + subject +
                f'_all_scores_models_DEAP_{measure}_' + score_name + '_' + cv_name + f'{debug_out}.npy'),
        all_scores)
    
    y_and_y_pred_opt_models = dict() 
    y_and_y_pred_opt_models['y'] = y
    for model in  ('spoc', 'riemann'):      
        if model == 'spoc':
            clf = make_filter_bank_regressor(
                                names=freqs.keys(),
                                method='spoc',
                                projection_params=dict(scale='auto', reg=1.e-05,
                                shrink=1, n_compo= best_components['spoc']),
                                vectorization_params=None,
                                estimator=GammaRegressor())   
        elif model == 'riemann':
            clf = make_filter_bank_regressor(
                                names=freqs.keys(),
                                method='riemann',
                                projection_params=dict(scale=1, reg=1.e-05,
                                                       n_compo= best_components['riemann']),
                                vectorization_params=dict(metric='riemann'),
                                estimator=GammaRegressor())

        
        # Run cross validaton
        y_preds = cross_val_predict(clf, df_features, y, cv=cv)

        # Plot the True EDA power and the EDA predicted from EEG data
        fig, ax = plt.subplots(1, 1, figsize=[30, 12])
        #times = raw.times[epochs.events[:, 0] - raw.first_samp]
        times = [i for i in range(len(epochs))]
        ax.plot(times, y, color='r', label=f'True {measure}')
        ax.plot(times, y_preds, color='b', label=f'Predicted {measure}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{measure} {y_stat}')
        ax.set_title(f'Sub {subject} - {model} model - {measure} prediction')
        plt.legend()
        plt_path = op.join(derivative_path, f'{measure}_plot--{date}-meegpowreg', 'sub-' + subject +
                            f'_DEAP_plot_prediction_{model}_{measure}{debug_out}.png')
        plt.savefig(plt_path)
        y_and_y_pred_opt_models[model] = y_preds

    np.save(op.join(derivative_path, f'{measure}_scores--{date}-meegpowreg', 'sub-' + subject +
                f'_y_and_y_pred_opt_models_{measure}_' + f'{debug_out}.npy'),
        y_and_y_pred_opt_models)
        
    del pipelines[f"spoc_{best_components['spoc']}"]
    del pipelines[f"riemann_{best_components['riemann']}"]
    
    
        
