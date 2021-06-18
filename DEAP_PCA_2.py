#%%
import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from meegpowreg import make_filter_bank_regressor, make_filter_bank_transformer
from subject_number import subject_number as subjects

freqs = {"low": (0.1, 1.5),
         "delta": (1.5, 4.0),
         "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0),
         "beta_low": (15.0, 26.0),
         "beta_high": (26.0, 35.0),
         "gamma_low": (35.0, 49.0)}

def make_filter_bank_PCA(names, method='riemann',
                               projection_params=None,
                               vectorization_params=None,
                               categorical_interaction=None, scaling=None,
                               estimator=None):
    filter_bank_transformer = make_filter_bank_transformer(
    names=names, method=method, projection_params=projection_params,
    vectorization_params=vectorization_params,
    categorical_interaction=categorical_interaction
    )

    scaling_ = scaling
    if scaling_ is None:
        scaling_ = StandardScaler()

    PCA_ = PCA(n_components=1)

    filter_bank_PCA = make_pipeline(
        filter_bank_transformer,
        scaling_,
        PCA_
        )
    
    return filter_bank_PCA

# Creation of the pipelines of interest
pipelines_PCA = {'riemann': make_filter_bank_PCA(
                    names=freqs.keys(),
                    method='riemann',
                    projection_params=dict(scale='auto', reg=1.e-05, n_compo = 31),
                    vectorization_params=dict(metric='riemann')),
                'spoc': make_filter_bank_PCA(
                    names=freqs.keys(),
                    method='spoc',
                    projection_params=dict(scale='auto', reg=0,
                                        shrink=0.5, n_compo = 3),
                    vectorization_params=None),
                'log_diag': make_filter_bank_PCA(
                    names=freqs.keys(),
                    method='log_diag',
                    projection_params=None,
                    vectorization_params=None),
                'upper': make_filter_bank_PCA(
                    names=freqs.keys(),
                    method='naive',
                    projection_params=None,
                    vectorization_params=None)}


DEBUG = True

if DEBUG:
    N_JOBS = 1
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

   cv_name = 'shuffle-split'

   # cv = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

   for key, estimator in pipelines_PCA.items():
      cv = KFold(n_splits=n_splits)
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

# %%
