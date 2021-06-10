import os.path as op
import pandas as pd
import numpy as np
import mne
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit)
from meegpowreg import make_filter_bank_regressor

derivative_path = '/storage/store2/work/amellot/derivatives'

# Import covariances and ages

# old_derivative = '/storage/inria/agramfor/camcan_derivatives'
fname = op.join(derivative_path, 'covariances.h5')
covs = mne.externals.h5io.read_hdf5(fname)
subjects = covs.keys()

subjects_mne = np.load(op.join('/storage/inria/agramfor/camcan_derivatives',
                               'scores_mag_models_mne_intervals_subjects.npy'),
                       allow_pickle=True)
subjects_common = [sub for sub in subjects_mne if sub in subjects]
covs = [covs[d] for d in covs if d in subjects_common]

X = np.array(covs)
n_sub, n_fb, n_ch, _ = X.shape
freqs = {"low": (0.1, 1.5),
         "delta": (1.5, 4.0),
         "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0),
         "beta_low": (15.0, 26.0),
         "beta_high": (26.0, 35.0),
         "gamma_lo": (35.0, 50.0),
         "gamma_mid": (50.0, 74.0),
         "gamma_high": (76.0, 120.0)}

X = list(X.transpose((1, 0, 2, 3)))
X = pd.DataFrame(dict(zip(list(freqs.keys()), map(list, X))),
                 index=subjects_common)

part = pd.read_csv(op.join('/storage/inria/agramfor/camcan_derivatives',
                           'participants.csv'))
y = part.set_index('Observations').age.loc[subjects_common]

# Creation of the pipelines of interest
pipelines = {'riemann': make_filter_bank_regressor(
                names=freqs.keys(),
                method='riemann',
                projection_params=dict(scale='auto', n_compo=65, reg=1.e-05),
                vectorization_params=dict(metric='riemann')),
             'spoc': make_filter_bank_regressor(
                names=freqs.keys(),
                method='spoc',
                projection_params=dict(scale='auto', n_compo=65, reg=0,
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

# Cross validation
seed = 42
n_splits = 100
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
                             scoring=scoring)
    if scoring == 'neg_mean_absolute_error':
        scores = -scores
        print(scores)
    all_scores[key] = scores

np.save(op.join(derivative_path,
                f'all_scores_models_camcan_{score_name}_{cv_name}.npy'),
        all_scores)
