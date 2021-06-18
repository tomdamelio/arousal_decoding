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

freqs = {"low": (0.1, 1.5),
         "delta": (1.5, 4.0),
         "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0),
         "beta_low": (15.0, 26.0),
         "beta_high": (26.0, 35.0),
         "gamma_low": (35.0, 49.0)}

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
                vectorization_params=None)}

DEBUG = True

if DEBUG:
    N_JOBS = 1
    sub = '01'
    subjects = subjects[:1]

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'


for subject in subjects:
   fname = op.join(derivative_path, 'sub-' + subject, 'eeg', 'sub-' + subject + '_covariances.h5')
   covs = mne.externals.h5io.read_hdf5(fname)

   X_cov = np.array([cc for cc in covs])
   df_features = pd.DataFrame(
      {band: list(X_cov[:, ii]) for ii, band in
      enumerate(freqs)})

   # Read EDA(y) data
   # change '01' to subjects then.

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
   
   picks_eda = mne.pick_channels(ch_names = epochs.ch_names ,include=['EDA'])

   if int(sub) < 23:
       epochs.apply_function(fun=lambda x: x/1000, picks=picks_eda)
   else:
       epochs.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)
   
   #whitout shift
   eda_epochs = epochs.copy().pick_channels(['EDA'])

   # With shift
   #eda_epochs = epochs.copy().pick_channels(['EDA']).shift_time(tshift= 1.5, relative=True)

   # How are we going to model our target?
   target = 'mean'

   if target == 'mean':
      y = eda_epochs.get_data().mean(axis=2)[:, 0]  
   elif target == 'delta':
      y = eda_epochs.get_data().max(axis=2)[:, 0] - eda_epochs.get_data().min(axis=2)[:, 0]
   else:
      y = eda_epochs.get_data().var(axis=2)[:, 0] 


   # Cross validation
   seed = 42
   n_splits = 2
   n_jobs = 10

   all_scores = dict()
   
   #score_name, scoring = "mae", "neg_mean_absolute_error"
   score_name, scoring = "r2", "r2"

   cv_name = '2Fold'

   cv = KFold(n_splits=n_splits) #, random_state=seed, shuffle=False)

   for key, estimator in pipelines.items():
      #cv = KFold(n_splits=n_splits)
      scores = cross_val_score(X=df_features, y=y, estimator=estimator,
                              cv=cv, n_jobs=min(n_splits, n_jobs),
                              scoring=scoring)
      if scoring == 'neg_mean_absolute_error':
         scores = -scores
         print(scores)
      all_scores[key] = scores

#%%
   np.save(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                  f'all_scores_models_DEAP_{score_name}_{cv_name}.npy'),
         all_scores)

#%%
# Obtain mean performance
all_scores_mean = dict()
for estimator, performance in all_scores.items():
   all_scores_mean[estimator] = performance.mean()
print('riemann -> ', all_scores_mean['riemann'])
print('spoc -> ', all_scores_mean['spoc'])
print('log_diag -> ', all_scores_mean['log_diag'])
print('upper -> ', all_scores_mean['upper'])

# %%
clf = make_filter_bank_regressor(
                names=freqs.keys(),
                method='riemann',
                projection_params=dict(scale='auto', reg=1.e-05, n_compo = 31),
                vectorization_params=dict(metric='riemann'))

# Run cross validaton
y_preds = cross_val_predict(clf, df_features, y, cv=cv)

# Plot the True EDA power and the EDA predicted from EEG data
fig, ax = plt.subplots(1, 1, figsize=[10, 4])
#times = raw.times[epochs.events[:, 0] - raw.first_samp]
times = [i for i in range(len(epochs))]
ax.plot(times, y, color='r', label='True EDA')
ax.plot(times, y_preds, color='b', label='Predicted EDA')
ax.set_xlabel('Time (s)')
ax.set_ylabel('EDA average')
ax.set_title('SPoC EEG Predictions')
plt.legend()
mne.viz.tight_layout()
plt.show()

# %%
