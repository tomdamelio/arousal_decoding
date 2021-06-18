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

'''
Visualize EDA signal through MNE BIDS Pipeline outcome epochs.
Compare with the mean (target in out model)
'''

# Read EDA(y) data
subject = '01'

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'


raw_path = BIDSPath(subject= subject, 
                        task='rest',
                        datatype='eeg',
                        root=derivative_path,
                        processing=None,
                        extension='.fif',
                        suffix='raw',
                        check=False)
# read epochs
raw = mne.io.read_raw_fif(raw_path)

#%%
# Function no get  channel and channel types from a raw file
def ch_and_ch_type(raw):
    channels_info = raw.info
    channels = channels_info.ch_names
    channels_info = mne.pick_types(channels_info, eeg = True, stim = True, eog = True, emg = True, misc = True, resp = True)
    channel_types = [mne.io.pick.channel_type(info, ch) for ch in channels_info]
    channel_and_channel_type = []
    for channel, channel_type in zip(channels, channel_types):
         out = str(channel+ '-> '+ channel_type)
         channel_and_channel_type.append(out)
    return channel_and_channel_type

#%%
picks_eda = mne.pick_channels(ch_names = epochs.ch_names ,include=['EDA'])

#if int(subject) < 23:
#    epochs.apply_function(fun=lambda x: x/1000, picks=picks_eda)
#else:
#    epochs.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)

eda_epochs = epochs.copy().pick_channels(['EDA'])

# How are we going to model our target?
y = eda_epochs.get_data().mean(axis=2)[:, 0]

# Create y_epoched_flat (epoched EDA signal)
y_epoched_aux = eda_epochs.get_data()
y_epoched_flat = y_epoched_aux.reshape(y_epoched_aux.shape[0]*y_epoched_aux.shape[2],)


# Plot EDA epochs -> Problems when trying to plot
%matplotlib
epochs.plot(picks=picks_eda)

# %%
# Plot y_epoched_flat (epoched EDA signal)
plt.plot(y, label = 'mean EDA')
plt.legend()
plt.show()
# %%
# Plot y (mean EDA) 
%matplotlib
plt.plot(y_epoched_flat, label='EDA')
plt.legend()
plt.show()

#%%
# Visualize raw data subseting by epochs
# Change the number of the epoch to visualize a particular epoch
epochs = 1
plt.plot(
    y_epoched_flat[y_epoched_aux.shape[2]*(epochs-1):y_epoched_aux.shape[2]*epochs],
    label='EDA_first_epochs'
    )
# %%
# Ploteo la sprimeras 60 epocas del sujeto 1, porque ahi visualizo que hay epocas
# con valores muy altos seguidos de valores muy bajos.
# Que significa eso?
# Visualizar la epoca 36 a ver por que se produce un picon en la media de las epocas

epochs = 104
plt.plot(
    y_epoched_flat[y_epoched_aux.shape[2]*(epochs-1):y_epoched_aux.shape[2]*epochs],
    label='EDA_first_epochs'
    )


# %%
