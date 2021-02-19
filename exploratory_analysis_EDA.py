#%%
import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from subject_number import subject_number
from my_functions import extract_signal

# subject number
i = '01'

# Extract EDA signal
subject_n_temp = extract_signal(signal = 'EDA', number_subject = i)

# Filter signal
s1_filtered = subject_n_temp.filter(0.05, 5., fir_design='firwin')

### CONTINUE PIPELINE IN MNE ###




### ------------------------ ###


# Create dataframe of EDA subject 1 (filtered)
df_s1_EDA = s1_filtered.to_data_frame()

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
x_detrended = df_s1_EDA['EDA']

plt.plot(t, x_detrended, label="S {}".format(i), linewidth=1)

plt.xlabel("Time(min)")
plt.ylabel("Skin conductance(µS)")
plt.legend(loc='best')
plt.show()
# %%
