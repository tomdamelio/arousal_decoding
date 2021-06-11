#%%
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit)
from meegpowreg import make_filter_bank_regressor
from subject_number import subject_number as subjects

#%%
derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

DEBUG = True

if DEBUG:
    N_JOBS = 1
    subjects = subjects[1:2]

# Import covariances (X) and EDA responses (y)
fname = op.join(derivative_path, 'sub-' + '01', 'eeg', 'sub-' + '01' + '_covariances.h5')

covs = mne.externals.h5io.read_hdf5(fname)

idx_epochs = covs.keys()

covs = [covs[d] for d in covs]

#%%
X = np.array(covs)
n_epochs, n_fb, n_ch, _ = X.shape # sub01 -> (472, 7, 32, 32)
freqs = {"low": (0.1, 1.5),
         "delta": (1.5, 4.0),
         "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0),
         "beta_low": (15.0, 26.0),
         "beta_high": (26.0, 35.0),
         "gamma_low": (35.0, 49.0)}

X = list(X.transpose((1, 0, 2, 3)))
X = pd.DataFrame(dict(zip(list(freqs.keys()), map(list, X))),
                 index=idx_epochs)

#%%
# Read EDA(y) data
# change '01' to subjects then.

epochs_path = BIDSPath(subject='02', 
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

# What are we going to model?
target = 'delta'

if target == 'mean':
   y = eda_epochs.get_data().mean(axis=2)[:, 0]  
elif target == 'delta':
   y = eda_epochs.get_data().max(axis=2)[:, 0] - eda_epochs.get_data().min(axis=2)[:, 0]
else:
   y = eda_epochs.get_data().var(axis=2)[:, 0] 

#%%
# Creation of the pipelines of interest
pipelines = {'riemann': make_filter_bank_regressor(
                names=freqs.keys(),
                method='riemann',
                projection_params=dict(scale='auto', reg=1.e-05),
                vectorization_params=dict(metric='riemann')),
             'spoc': make_filter_bank_regressor(
                names=freqs.keys(),
                method='spoc',
                projection_params=dict(scale='auto', reg=0,
                                       shrink=0.5),
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

#%%
# Cross validation
seed = 42
n_splits = 10
n_jobs = 10

all_scores = dict()
score_name, scoring = "mae", "neg_mean_absolute_error"
cv_name = 'shuffle-split'

# cv = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

# scores = cross_val_score(X=X, y=y, estimator=model,
#                          cv=cv, n_jobs=min(n_splits, n_jobs),
#                          scoring=scoring)

# scores = -scores

for key, estimator in pipelines.items():
    cv = ShuffleSplit(test_size=.1, n_splits=n_splits, random_state=seed)
    scores = cross_val_score(X=X, y=y, estimator=estimator,
                             cv=cv, n_jobs=min(n_splits, n_jobs),
                             scoring='r2')
    if scoring == 'neg_mean_absolute_error':
        scores = -scores
        print(scores)
    all_scores[key] = scores

#%%
np.save(op.join(derivative_path, 'sub-' + '01', 'eeg','sub-' + '01' +
                f'all_scores_models_DEAP_{score_name}_{cv_name}.npy'),
        all_scores)

# %%
