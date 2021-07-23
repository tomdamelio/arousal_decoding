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

DEBUG = True

date = '22-07'

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
   subjects = ['10']
   subject = ['10']
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
        
        plt.plot(y)
        plt_path = op.join(derivative_path, f'{measure}_plot--{date}-meegpowreg-ONLY_TRUE_MEASURE', 'sub-' + subject +
                            f'_DEAP_plot__{measure}_ONLY_TRUE MEASURE.png')
        plt.savefig(plt_path)
        plt.clf()
    
