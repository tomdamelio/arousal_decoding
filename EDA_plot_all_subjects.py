#%%
import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from subject_number import subject_number
from preprocessing import extract_signal

subset_1_subjects = subject_number[0:10]
subset_2_subjects = subject_number[10:20]
subset_3_subjects = subject_number[20:27]
 
for i in subset_1_subjects:
#i = '01'
# Extract signal
    raw = extract_signal(directory = 'data', number_subject=i,
                     extension = '.bdf')

    # Filter signal
    #raw = raw.filter(0.05, 5., fir_design='firwin')
    
    #raw = mne.pick_channels(ch_names = raw.ch_names ,include=['EDA'])
    ### CONTINUE PIPELINE WITH DATAFRAMES (PANDAS) ###
    
    # Create dataframe of EDA subject 1 (filtered)
    df_s1_EDA = raw.to_data_frame()[40] # EDA channel

    # Select first part of the signal
    df_s1_EDA = df_s1_EDA.head(20000)
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
    t = df_s1_EDA['time_sec']
    #x = df_s1_EDA['EDA']
    x_detrended = df_s1_EDA['EDA']

    #plt.figure(figsize=(5, 4))
    plt.plot(t, x_detrended, label= i ,linewidth=0.4)

    plt.xlabel("Time(sec)")
    plt.ylabel("Skin conductance(µS)")
    plt.legend(loc='best')
    
plt.show()


### ------------------------------------------ ###

#%%
### CONTINUE PIPELINE IN MNE ###




### ------------------------ ###