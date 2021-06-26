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
from numpy.lib.npyio import save
import json
from collections import defaultdict
import pathlib
from pyEDA.main import *

DEBUG = True

if DEBUG:
    N_JOBS = 1
    subjects = subjects[:1]

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

for subject in subjects:
    
    raw_path = BIDSPath(subject= subject, 
                           task='rest',
                           datatype='eeg',
                           root=derivative_path,
                           extension='.fif',
                           suffix='raw',
                           check=False)
    # Read EDA data
    raw = mne.io.read_raw_fif(raw_path)
    
    # Pick only EDA data
    raw_eda = raw.copy().pick_channels(['EDA'])
    
    eda = raw_eda.get_data()[0]
    
#%%
# extracts clean EDA
m, wd, eda_clean = process_statistical(eda, use_scipy=True, sample_rate=512, new_sample_rate=256, segment_width=5, segment_overlap=0)


# %%
# NOT WORKING YET
# Extract Automatic Features

prepare_automatic(eda, sample_rate=512, new_sample_rate=512, k=32, epochs=100, batch_size=10)
# k number of automatic features to extract
#%%
automatic_features = process_automatic(eda)
# %%
