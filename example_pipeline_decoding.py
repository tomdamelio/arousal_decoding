# Author: Alexandre Barachant <alexandre.barachant@gmail.com>
#         Jean-Remi King <jeanremi.king@gmail.com>
#
# License: BSD (3-clause)
# Link https://mne.tools/dev/auto_examples/decoding/plot_decoding_spoc_CMC.html#sphx-glr-auto-examples-decoding-plot-decoding-spoc-cmc-py

#%%

import matplotlib.pyplot as plt

import mne
from mne import Epochs
from mne.decoding import SPoC
from mne.datasets.fieldtrip_cmc import data_path

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from my_functions import extract_signal

# Define parameters
number_subject = '01'

#Extract signal
raw = extract_signal(directory = 'data', number_subject=number_subject,
                     extension = '.bdf')

# Rename channel EDA and set GSR as channel type
mne.rename_channels(info= raw.info , mapping={'GSR1':'EDA'})
raw.set_channel_types({'EDA': 'misc'})

# Filter muscular activity to only keep high frequencies
eda = raw.copy().pick_channels(['EDA']).load_data()

#emg.filter(20., None, fir_design='firwin')
eda.filter(0.05, 5., fir_design='firwin', picks=['EDA'])

# Select and filter EEG data 
raw.pick_types(meg=False, ref_meg=False, eeg=True, eog=True).load_data()
raw.filter(None, 50., fir_design='firwin')

#%%
# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=1, duration=10)


#%%
# Epoch length is 1.5 second
meg_epochs = Epochs(raw, events, tmin=0., tmax=1.500, baseline=None,
                    detrend=1, decim=8)
emg_epochs = Epochs(emg, events, tmin=0., tmax=1.500, baseline=None)


#%%
# Prepare classification
X = meg_epochs.get_data()
y = emg_epochs.get_data().var(axis=2)[:, 0]  # target is EMG power

# Classification pipeline with SPoC spatial filtering and Ridge Regression
spoc = SPoC(n_components=2, log=True, reg='oas', rank='full')
clf = make_pipeline(spoc, Ridge())
# Define a two fold cross-validation
cv = KFold(n_splits=2, shuffle=False)

# Run cross validaton
y_preds = cross_val_predict(clf, X, y, cv=cv)

# Plot the True EMG power and the EMG power predicted from MEG data
fig, ax = plt.subplots(1, 1, figsize=[10, 4])
times = raw.times[meg_epochs.events[:, 0] - raw.first_samp]
ax.plot(times, y_preds, color='b', label='Predicted EMG')
ax.plot(times, y, color='r', label='True EMG')
ax.set_xlabel('Time (s)')
ax.set_ylabel('EMG Power')
ax.set_title('SPoC MEG Predictions')
plt.legend()
mne.viz.tight_layout()
plt.show()
# %%
