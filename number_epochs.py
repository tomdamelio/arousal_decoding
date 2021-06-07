#%%
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import config as cfg
import pathlib

from subject_number import subject_number

directory = 'outputs/DEAP-bids'
extension = '.fif'
process = 'proc-clean_epo'

n_epochs = np.zeros(32)
ii = 0
for subject in subject_number:
    fname = op.join(directory, 'derivatives/mne-bids-pipeline', 'sub-'+ subject, 'eeg', 'sub-'+
                    subject + '_task-rest_' + process + extension)
    epochs = mne.read_epochs(fname, preload=True)
    n_epoch = len(epochs)
    n_epochs[ii] = n_epoch
    ii += 1

#%%
%matplotlib
plt.hist(n_epochs,
         edgecolor = 'salmon',
         color = 'salmon',
         bins = 32)
plt.xlabel('Epochs')
plt.ylabel('Number of subjects') 
plt.title('Epochs: 5 seconds (no overlap)')
plt.legend()
plt.show()
# %%
