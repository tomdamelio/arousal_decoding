#%%
import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from subject_number import subject_number
from my_functions import extract_signal

number_subject = '01'

#Extract signal
raw = extract_signal(directory = 'data', number_subject=number_subject,
                     extension = '.bdf')

# Pick EDA signal
#raw.pick_channels(['GSR1'])

# Rename channel EDA and set GSR as channel type
mne.rename_channels(info= raw.info , mapping={'GSR1':'EDA'})
raw.set_channel_types({'EDA': 'misc'})

# Creat numpy array
#INTENTO 1
#EDA_index = mne.pick_types(raw.info, misc=True)
#EDA_array = raw.get_data('EDA_index',return_times=True) #[mne.pick_types(raw.info, misc=True)]
#INTENTO 2
EDA_index = mne.pick_types(raw.info, misc=True)
EDA_array = raw.get_data()[EDA_index]#(return_times=True)#[EDA_index]


#%%

# 1)  Transform EDA (depending on recording procedure) --> IN PROGRESS (results doesn't match what I have obtained with DFs)
if int(number_subject) < 23:
    EDA_array_transformed = EDA_array / 10**9
else:
    EDA_array_transformed = (10**9/EDA_array)*1000  
    
# 2) Clean signals
#    -  Negative values            ==> 01 02 03 08 14 15
#    -  Out-of-range values        ==> 26
#    -  Sudden jumps in the signal ==> 31

# 3) Change variable names (and x and y labels in plots)


# Return to Raw data
#raw = mne.io.RawArray(data=EDA_array_transformed, info= raw.info)

# Filter signal
#raw_filtered = raw.filter(0.05, 5., fir_design='firwin')



#%%
#####  Continue working with DataFrames ######

# Create dataframe of EDA subject 1 (filtered)
df_s1_EDA = raw_filtered.to_data_frame()


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
