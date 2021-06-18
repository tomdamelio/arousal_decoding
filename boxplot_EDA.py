#Boxplot EDA - all subjects in
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

for subject in subjects:
    if os.name == 'nt':
        derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
    else:
        derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'


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

    eda_epochs = epochs.copy().pick_channels(['EDA'])

    # How are we going to model our target?
    y = eda_epochs.get_data().mean(axis=2)[:, 0]
    
    plt.clf()
    plt.boxplot(y)
    plt.xlabel("EDA")
    plt.title("subject {}".format(subject))
    #plt.show()
    directory = 'outputs/figures/boxplot_mean_EDA/'
    fname = op.join(directory, 'boxplot_mean_EDA_sub{}.png'.format(subject))
    plt.savefig(fname, dpi = 300)
# %%
