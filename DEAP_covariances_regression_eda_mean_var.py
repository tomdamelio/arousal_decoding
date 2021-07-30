import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import mne
from mne_bids import BIDSPath
from sklearn.linear_model import RidgeCV, GammaRegressor
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, GroupShuffleSplit,
     cross_val_predict, GridSearchCV)
from coffeine import make_filter_bank_regressor
from subject_number import subject_number as subjects
from joblib import Parallel, delayed

###  SET CONFIGS ###

# eda or emg?    
measure = 'eda'
# var or mean?   
y_stat = 'var'

DEBUG = True
####################


if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg

if os.name == 'nt':
    derivative_path = cfg.deriv_root
    derivative_path_3 = cfg.deriv_root  
else:
    derivative_path = cfg.deriv_root
    derivative_path_3 = cfg.deriv_root_store3
    
n_components = np.arange(1, 32, 1)
seed = 42
n_splits = 2
n_jobs = 15
score_name, scoring = "r2", "r2"
cv_name = '2Fold'

freqs = {'low': (0.1, 1.5),
         'delta': (1.5, 4.),
         'theta': (4., 8.),
         'alpha': (8., 15.),
         'beta_l': (15., 26.),
         'beta_h': (26., 35.),
         'gamma': (35., 49.)}

if y_stat == 'mean' or measure == 'eda':
    estimator_type = RidgeCV(alphas=np.logspace(-3, 5, 100))
else:
    estimator_type = GammaRegressor()

pipelines = {'riemann': make_filter_bank_regressor(
                names=freqs.keys(),
                method='riemann',
                projection_params=dict(scale=1, reg=1.e-05, n_compo=31),
                vectorization_params=dict(metric='riemann'),
                estimator=estimator_type),
             'spoc': make_filter_bank_regressor(
                names=freqs.keys(),
                method='spoc',
                projection_params=dict(scale='auto', reg=1.e-05, shrink=1, n_compo=31),
                vectorization_params=None,
                estimator=estimator_type),
             'log_diag': make_filter_bank_regressor(
                names=freqs.keys(),
                method='log_diag',
                projection_params=None,
                vectorization_params=None,
                estimator=estimator_type),
             'upper': make_filter_bank_regressor(
                names=freqs.keys(),
                method='naive',
                projection_params=None,
                vectorization_params=None,
                estimator=estimator_type),
             'random': make_filter_bank_regressor(
                names=freqs.keys(),
                method='random',
                projection_params=None,
                vectorization_params=None,
                estimator=estimator_type)}

if DEBUG:
   n_jobs = 4
   subjects = ['01']
   subject = '01'
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
    Use to filter EMG data
    -------
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
        
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx] 
    return(vals)


for subject in subjects:
    if os.name == 'nt':
        fname_covs = op.join(derivative_path, measure + '-cov-matrices-all-freqs',
                             'sub-' + subject + '_covariances_' + measure + '.h5')
    else:
        fname_covs = op.join(derivative_path, 'sub-' + subject, 'eeg',
                             'sub-' + subject + '_covariances_' + measure + '.h5')
    
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
      
    if DEBUG:
        epochs = epochs[:30]
    
    if measure == 'emg':
        # 1. Band pass filter EMG at 20 hz
        picks_emg = mne.pick_types(epochs.info, emg=True)
        epochs = epochs.filter(l_freq = 20., h_freq = None, picks=picks_emg)
        emg_epochs = epochs.copy().pick_channels(['EXG5','EXG6'])

        # 2. EMG activity = EMG z1 - EMG z2
        emg = emg_epochs.get_data()
        emgz1 = emg[:,0,:]
        emgz2 = emg[:,1,:]
        
        emg_delta = emgz1 - emgz2
        
        if y_stat == 'mean':
            emg_delta_stat = emg_delta.mean(axis=1)
        elif y_stat == 'var':
            emg_delta_stat = emg_delta.var(axis=1)
        emg_delta_stat_series = pd.Series(emg_delta_stat)
        y = hampel(emg_delta_stat_series)   
        
    if measure == 'eda': 
        picks_eda = mne.pick_channels(ch_names = epochs.ch_names ,include=['EDA'])       
        if int(subject) > 22:
            epochs.apply_function(fun=lambda x: 10**9/x, picks=picks_eda)
            
        epochs = epochs.filter(l_freq = None, h_freq = 5., picks=picks_eda)
        
        if y_stat == 'mean':
            y = epochs.get_data().mean(axis=2)[:, 0]
        elif y_stat == 'var':
            y = epochs.get_data().var(axis=2)[:, 0]        
  
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
    
    date = datetime.datetime.now().strftime("%d-%m--%H-%M")
    opt_dir = op.join(derivative_path_3, measure + '_opt--' + date + '-' + y_stat)
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)
    
    out_df.to_csv(op.join(derivative_path_3, measure + '_opt--' + date + '-' + y_stat,
                          'sub-' + subject + '_DEAP_component_scores_' + measure + '_' +
                          y_stat + '_' + debug_out + '.csv'))
 
    mean_df = out_df.groupby('n_components').mean().reset_index()
    best_components = {
       'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
       'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
    }
     
    pipelines[f"spoc_{best_components['spoc']}"] = make_filter_bank_regressor(
                                        names=freqs.keys(),
                                        method='spoc',
                                        projection_params=dict(scale='auto', reg=1.e-05,
                                                               shrink=1,
                                                               n_compo= best_components['spoc']),
                                        vectorization_params=None,
                                        estimator=estimator_type)

    pipelines[f"riemann_{best_components['riemann']}"] = make_filter_bank_regressor(
                                        names=freqs.keys(),
                                        method='riemann',
                                        projection_params=dict(scale=1, reg=1.e-05,
                                                               n_compo= best_components['riemann']),
                                        vectorization_params=dict(metric='riemann'),
                                        estimator=estimator_type)

    all_scores = dict() 
    for key, estimator in pipelines.items():
       if measure == 'emg': 
           param_grid = {'gammaregressor__alpha': [0.01, 1., 10.]}
           search = GridSearchCV(pipelines[key], param_grid)
           search.fit(df_features, y)
           print(search.best_params_['gammaregressor__alpha'])
           estimator.steps[-1] = ('gammaregressor',
                                  GammaRegressor(alpha = search.best_params_['gammaregressor__alpha'],
                                                 max_iter=1000))
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
    
    scores_dir = op.join(derivative_path_3, measure + '_scores--' + date + '-' + y_stat)
    if not os.path.exists(scores_dir):
        os.makedirs(scores_dir)
        
    np.save(op.join(scores_dir,'sub-' + subject + '_all_scores_models_DEAP_' + measure + '_' +
                          y_stat + '_' + score_name + '_' + cv_name + debug_out + '.npy'),
            all_scores)
    
    y_and_y_pred_opt_models = dict() 
    y_and_y_pred_opt_models['y'] = y
    for model in  ('spoc', 'riemann'):      
        if model == 'spoc':
            if measure == 'emg':
                estimator_spoc_opt = GammaRegressor(alpha = spoc_opt_alpha, max_iter=1000)
            else:
                estimator_spoc_opt = None        
            clf = make_filter_bank_regressor(
                                    names=freqs.keys(),
                                    method='spoc',
                                    projection_params=dict(scale='auto', reg=1.e-05,
                                    shrink=1, n_compo= best_components['spoc']),
                                    vectorization_params=None,
                                    estimator=estimator_spoc_opt)   
            score_opt = np.asarray([v for k,v in all_scores.items() if 'spoc_'in k]).mean().round(3)
        elif model == 'riemann':
            if measure == 'emg':
                estimator_riemann_opt = GammaRegressor(alpha = riemann_opt_alpha, max_iter=1000)
            else:
                estimator_riemann_opt = None
            clf = make_filter_bank_regressor(
                                names=freqs.keys(),
                                method='riemann',
                                projection_params=dict(scale=1, reg=1.e-05,
                                                       n_compo= best_components['riemann']),
                                vectorization_params=dict(metric='riemann'),
                                estimator=estimator_riemann_opt)
            score_opt = np.asarray([
                v for k,v in all_scores.items() if 'riemann_'in k]).mean().round(3)

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
        
        plot_dir = op.join(derivative_path_3, measure + '_plot--' + date + '-' + y_stat)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        plt_path = op.join(derivative_path_3, measure + '_plot--' + date + '-' + y_stat,
                           'sub-' + subject + '_DEAP_plot_prediction_' + model + '_' +
                           measure + '_' + y_stat + '_' + debug_out + '.png')
        plt.savefig(plt_path)
        
    np.save(op.join(scores_dir, 'sub-' + subject + '_y_and_y_pred_opt_models_' +
                    measure + '_' + y_stat + '_' + debug_out + '.npy'),
            y_and_y_pred_opt_models)
        
    del pipelines[f"spoc_{best_components['spoc']}"]
    del pipelines[f"riemann_{best_components['riemann']}"]
