#%%
import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mne
from mne_bids import BIDSPath
from sklearn.linear_model import RidgeCV, GammaRegressor
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, GroupShuffleSplit, cross_val_predict)
from meegpowreg import make_filter_bank_regressor
from subject_number import subject_number as subjects
from joblib import Parallel, delayed

measure = 'eda'

if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg
    
derivative_path = cfg.deriv_root

subjects = ['01']
subject = '01'

fname_covs = op.join(derivative_path, f'{measure}-cov-matrices-all-freqs', 'sub-' + subject + f'_covariances_{measure}.h5')

fname_epochs = derivative_path / 'clean-epo-files'

picks_eda = mne.pick_channels(ch_names = epochs.ch_names ,include=['EDA'])       
if int(subject) < 23:
    epochs.apply_function(fun=lambda x: x/1000, picks=picks_eda)
else:
    epochs.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)
    
eda_epochs = epochs.copy().pick_channels(['EDA'])

y_stat = 'mean'
y = eda_epochs.get_data().mean(axis=2)[:, 0]    