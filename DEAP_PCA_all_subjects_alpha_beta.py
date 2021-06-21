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


freqs = {
#         "low": (0.1, 1.5),
#         "delta": (1.5, 4.0),
#         "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0),
         "beta_low": (15.0, 26.0),
         "beta_high": (26.0, 35.0),
#         "gamma_low": (35.0, 49.0)
         }

filter_bank_transformer = make_filter_bank_transformer(names=freqs.keys(),
                                                       method='riemann')
filter_bank_PCA = make_pipeline(
    filter_bank_transformer,
    StandardScaler(),
    PCA(n_components=3))

DEBUG = False

if DEBUG:
    N_JOBS = 1
    subjects = subjects[1:2]

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
    
    PCA_eeg = filter_bank_PCA.fit_transform(df_features)

    np.save(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                  'PCA_alpha_beta.npy'), PCA_eeg)
#%%
for subject in subjects:
    PCA_123 = np.load(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                  'PCA_alpha_beta.npy'))
    df_PCA = pd.DataFrame(PCA_123, columns = ['PCA1','PCA2','PCA3'])

