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


DEBUG = True

if DEBUG:
    N_JOBS = 1
    subjects = subjects[:1]

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

y_all_features = dict()
for subject in subjects:
    # Read raw EDA(y) data
    raw_path = BIDSPath(subject= subject, 
                           task='rest',
                           datatype='eeg',
                           root=derivative_path,
                           processing='filt',
                           extension='.fif',
                           suffix='raw',
                           check=False)
    # read raw
    raw = mne.io.read_raw_fif(raw_path)

    ### FILTER SIGNAL IN ALL POSSIBLE WAYS ###
    ### agregarlos como canales distintos  ###
    ###           al raw file.             ###
    
    # scypy.signal.svgol_filter
    
    # 512 points moving average raw signal 
    
    # filt_05_5_y
    
    # svgol_filt_05_5_y 
    
    # filt_05_3_y
    
    ### EPOCHEAR DE MODO ANALOGO A COMO SE HIZO EN MNE BIDS PIPELINE  ###
    ###       HAY QUE OBTENER FINALMENTE LAS MISMAS EPOCAS            ###


    ### EXTRACT INDIXES OF EPOCHS TO REJECT ###

    epochs_path = BIDSPath(subject= subject, 
                           task='rest',
                           datatype='eeg',
                           root=derivative_path,
                           processing='clean',
                           extension='.fif',
                           suffix='epo',
                           check=False)
    
    epochs_aux = mne.read_epochs(epochs_path)
    
    ### REJECTEAR LAS EPOCAS EXTRAIDAS DE 'epochs_aux' DE MI SEÑAL###
    
    ### CREAR UN ARCHIVO NPY CON TODAS LAS MEDIDAS PARA CADA UNO DE  ###
    ###        DE LOS SUJETOS (EN SUS RESPECTIVAS CAPRETAS)          ###



#%%
    # keep only EDA data
    eda_epochs = epochs.copy().pick_channels(['EDA'])

    y = eda_epochs.get_data().mean(axis=2)[:, 0]  

    mean_eda_all_subjects[subject] = y

#%%    
# Save dict with  mean(EDA) for every subject
fname_out = op.join(derivative_path, 'mean_eda_all_subs.npy')
np.save(fname_out, mean_eda_all_subjects)

#%%
# read dict with mean(EDA) for every subjects
mean_eda_all_subjects_npy = np.load(fname_out, allow_pickle=True)
#mean_eda_all_subjects_npy[()]['01']
# %%
