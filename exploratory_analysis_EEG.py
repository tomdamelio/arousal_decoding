#%%
import os
import mne
from EEG_channel_name import channels_geneva, channels_twente 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import numpy as np

# Read bdf
number_subject = '10'
path = os.path.join('data', 's'+ number_subject + '.bdf')
s1 = mne.io.read_raw_bdf(path, preload=True)
# Print info of the subject 1 signal
print(s1.info)

# Select only EEG data
s1_temp2 = s1.copy()
print('Number of channels in s1_temp:')
print(len(s1_temp2.ch_names), end=' → pick only EEG → ')
if int(number_subject) < 23:
    s1_temp2.pick_channels(channels_twente)
else:
    s1_temp2.pick_channels(channels_geneva)

print(len(s1_temp2.ch_names))
# Plot EEG 
s1_temp2.plot()
# Plot the EEG power spectral density 
s1_temp2.plot_psd()
# Create dataframe EEG subject 1
df_s1_EEG = s1_temp2.to_data_frame()

# %%
