import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from sklearn.linear_model import RidgeCV, GammaRegressor
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, GroupShuffleSplit, cross_val_predict, GridSearchCV)
from meegpowreg import make_filter_bank_regressor
from subject_number import subject_number as subjects
from joblib import Parallel, delayed

measure = 'emg'

if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg

DEBUG = False

date = '22-07_corrected'

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
   subjects = ['01']
   subject = ['01']
   debug_out = '_DEBUG'
else:
   debug_out = ''

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

def hampel(vals_orig, k=18, t0=1):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    
    #Make copy so original not edited
    vals = vals_orig.copy()
    
    #Hampel Filter
    L = 3.
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)
    
    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''
    
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx] 
    return(vals)


for subject in subjects:
    if os.name == 'nt':
        fname_covs = op.join(derivative_path, f'{measure}-cov-matrices-all-freqs', 'sub-' + subject + f'_covariances_{measure}.h5')
    else:
        fname_covs = op.join(derivative_path, 'sub-' + subject, 'eeg', 'sub-' + subject + f'_covariances_{measure}.h5')
    
    covs = mne.externals.h5io.read_hdf5(fname_covs)
    
    if DEBUG:
       covs = covs[:30]
 
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
      
    if DEBUG:
        epochs = epochs[:30]
    
    if measure == 'emg':
        # 1. Band pass filter EMG at 20 hz- 256 hz (it is not possible filter more than nfreq/2)
        picks_emg = mne.pick_types(epochs.info, emg=True)
        epochs = epochs.filter(l_freq = 20., h_freq = None, picks=picks_emg)
        emg_epochs = epochs.copy().pick_channels(['EXG5','EXG6'])

        # 2. EMG activity = EMG z1 - EMG z2
        emg = emg_epochs.get_data()
        emgz1 = emg[:,0,:]
        emgz2 = emg[:,1,:]
        emg_delta = emgz1 - emgz2
        
        # 3. Calculate RMS
        y_stat = 'rms'
        emg_delta_squared = np.square(emg_delta)
        emg_rms = np.sqrt(emg_delta_squared.mean(axis=1))
        emg_rms_series = pd.Series(emg_rms)
        
        # 4. Implement Hampel filtering to EMG data
        y = hampel(emg_rms_series)
        
        
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
                                        projection_params=dict(scale=1, reg=1.e-05,
                                                               n_compo= best_components['riemann']),
                                        vectorization_params=dict(metric='riemann'),
                                        estimator=GammaRegressor())

    all_scores = dict() 
    for key, estimator in pipelines.items():
       param_grid = {'gammaregressor__alpha': [0.01, 1., 10.]}
       search = GridSearchCV(pipelines[key], param_grid)
       search.fit(df_features, y)
       print(search.best_params_['gammaregressor__alpha'])
       estimator.steps[-1] = ('gammaregressor', GammaRegressor(alpha = search.best_params_['gammaregressor__alpha'], max_iter=1000))
       if 'spoc_' in key:
           spoc_opt_alpha = search.best_params_['gammaregressor__alpha']
       if 'riemann_' in key:
           riemann_opt_alpha = search.best_params_['gammaregressor__alpha']
           
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
                                estimator=GammaRegressor(alpha = spoc_opt_alpha, max_iter=1000))   
            score_opt = np.asarray([v for k,v in all_scores.items() if 'spoc_'in k]).mean().round(3)
        elif model == 'riemann':
            clf = make_filter_bank_regressor(
                                names=freqs.keys(),
                                method='riemann',
                                projection_params=dict(scale=1, reg=1.e-05,
                                                       n_compo= best_components['riemann']),
                                vectorization_params=dict(metric='riemann'),
                                estimator=GammaRegressor(alpha = riemann_opt_alpha, max_iter=1000))
            score_opt = np.asarray([v for k,v in all_scores.items() if 'riemann_'in k]).mean().round(3)

        # Run cross validaton
        y_preds = cross_val_predict(clf, df_features, y, cv=cv)
        y_and_y_pred_opt_models[model] = y_preds
        
        # Plot the True EDA power and the EDA predicted from EEG data
        fig, ax = plt.subplots(1, 1, figsize=[20, 8])
        times = [i for i in range(len(epochs))]
        ax.plot(times, y, color='r', alpha = 0.5, label=f'True {measure}')
        ax.plot(times, y_preds, color='b', alpha = 0.5, label=f'Predicted {measure}')
        ax.set_xlabel('Time (epochs)')
        ax.set_ylabel(f'{measure} {y_stat}')
        ax.set_title(f'Sub {subject} - {model} model - {measure} prediction\nR2 = {score_opt}')
        plt.legend()
        plt_path = op.join(derivative_path, f'{measure}_plot--{date}-meegpowreg', 'sub-' + subject +
                            f'_DEAP_plot_prediction_{model}_{measure}{debug_out}.png')
        plt.savefig(plt_path)
        

    np.save(op.join(derivative_path, f'{measure}_scores--{date}-meegpowreg', 'sub-' + subject +
                f'_y_and_y_pred_opt_models_{measure}_' + f'{debug_out}.npy'),
        y_and_y_pred_opt_models)
        
    del pipelines[f"spoc_{best_components['spoc']}"]
    del pipelines[f"riemann_{best_components['riemann']}"]
    
    
