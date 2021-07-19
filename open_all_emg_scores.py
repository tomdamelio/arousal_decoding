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
from joblib import Parallel, delayed

measure = 'emg'

if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg

derivative_path = cfg.deriv_root

score_name, scoring = "r2", "r2"
cv_name = '2Fold'

DEBUG = False

if DEBUG:
    n_jobs = 4
    subjects = subjects[31:32]
    subject = '32'
    debug_out = '_DEBUG'
else:
    debug_out = ''

# (despues de importar librerias conomo numpy)
all_scores = dict()
for subject in subjects:
    score_sub = np.load(op.join(derivative_path, f'{measure}_scores--16-07-meegpowreg', 'sub-' + subject +
                        f'_all_scores_models_DEAP_{measure}_' + score_name + '_' + cv_name + f'{debug_out}.npy'),
                        allow_pickle=True)
    all_scores[subject] = score_sub
# %%
