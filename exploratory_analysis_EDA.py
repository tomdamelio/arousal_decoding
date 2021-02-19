#%%
import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from subject_number import subject_number
from my_functions import extract_signal

#trial = ['01','02','03','04']
#for i in  trial:
# Read bdf
extract_signal()

#path = os.path.join('data', 's'+ i + '.bdf')
#s1 = mne.io.read_raw_bdf(path, preload=True)
# Print info of subject's signal
#print(s1.info)

# Select EDA data
#s1_temp = s1.copy()
#print('Number of channels in s1_temp:')
#print(len(s1_temp.ch_names), end=' → pick only EDA → ')
#s1_temp.pick_channels(['GSR1'])
#print(len(s1_temp.ch_names))


# Filter signal
s1_filtered = subject_n_temp.filter(0.05, 5., fir_design='firwin')

# Create dataframe of EDA subject 1 (filtered)
df_s1_EDA = s1_filtered.to_data_frame()


### CONTINUE PIPELINE WITH DATAFRAMES (PANDAS) ###
#Rename column
df_s1_EDA.rename(columns={'GSR1': 'EDA'}, inplace=True)

# Transform EDA (participant 23-32 in Geneva) --> GSR geneva = 10**9 / GSR twente
if int(i) < 23:
    df_s1_EDA["EDA"] = (df_s1_EDA["EDA"])/10**9
else:
    df_s1_EDA["EDA"] = (10**9/df_s1_EDA["EDA"])*1000  
    
# Create column "time_min"
df_s1_EDA['time_min'] = (df_s1_EDA.time/1000)/60
# Create column "time_sec"
df_s1_EDA['time_sec'] = df_s1_EDA.time/1000

# plot signal
t = df_s1_EDA['time_min']
#x = df_s1_EDA['EDA']
x_detrended = df_s1_EDA['EDA']

#plt.figure(figsize=(5, 4))
plt.plot(t, x_detrended, label="S {}".format(i), linewidth=1)

plt.xlabel("Time(min)")
plt.ylabel("Skin conductance(µS)")
plt.legend(loc='best')
plt.show()


### ------------------------------------------ ###

#%%
### CONTINUE PIPELINE IN MNE ###




### ------------------------ ###