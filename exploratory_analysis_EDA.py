#%%
import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from subject_number import subject_number

#for i in  subject_number:
i = '01'
# Read bdf
path = os.path.join('data', 's'+ i + '.bdf')
s1 = mne.io.read_raw_bdf(path, preload=True)
# Print info of subject's signal
print(s1.info)

# Select EDA data
s1_temp = s1.copy()
print('Number of channels in s1_temp:')
print(len(s1_temp.ch_names), end=' → pick only EDA → ')
s1_temp.pick_channels(['GSR1'])
print(len(s1_temp.ch_names))

# Create dataframe of EDA subject 1
df_s1_EDA = s1_temp.to_data_frame()
# Filter signal
s1_filtered = s1_temp.filter(0.05, 5., fir_design='firwin')

### CONTINUE PIPELINE IN MNE ###




### ------------------------ ###
