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
from sklearn import preprocessing
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

filter_bank_transformer = make_filter_bank_transformer(
                    names = freqs.keys(),
                    projection_params=dict(scale='auto', reg=1.e-05, n_compo = 31),
                    vectorization_params=dict(metric='riemann'))

filter_bank_PCA = make_pipeline(
    filter_bank_transformer,
    StandardScaler(),
    PCA(n_components=1))


DEBUG = True

if DEBUG:
    N_JOBS = 1
    subjects = subjects[5:6]

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

    if int(subject) < 23:
        epochs.apply_function(fun=lambda x: x/1000, picks=picks_eda)
    else:
        epochs.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)

    # whitout shift
    eda_epochs = epochs.copy().pick_channels(['EDA'])
    
    # How are we going to model our target?
    target = 'mean'

    if target == 'mean':
       y = eda_epochs.get_data().mean(axis=2)[:, 0]  
    elif target == 'delta':
       y = eda_epochs.get_data().max(axis=2)[:, 0] - eda_epochs.get_data().min(axis=2)[:, 0]
    else:
       y = eda_epochs.get_data().var(axis=2)[:, 0] 

df_PCA1 = filter_bank_PCA.fit_transform(df_features)

# Plot without scale
plt.plot(y, label = 'EDA')
plt.plot(df_PCA1, label='EEG')
plt.legend()
plt.show()


# Plot scaled

min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y.reshape(-1, 1))
df_PCA1_scaled = min_max_scaler.fit_transform(df_PCA1)

plt.plot(df_PCA1_scaled, label='EEG')
plt.plot(y_scaled, label = 'EDA')
plt.legend()
plt.show()
# %%
