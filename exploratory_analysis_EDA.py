#%%
import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from subject_number import subject_number

#os.mkdir('subject_plots')
#subject_number_sample = ['01','02']
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

df_s1_filtered_EDA = s1_filtered.to_data_frame()

# Plot  EDA. Plot only first part of the signal
#%matplotlib qt
#s1_filtered.plot(title='EDA' , scalings='auto')

# Plot the EDA power spectral density 
#s1_filtered.plot_psd(fmin=0.01, fmax=5, area_mode='range', picks=['GSR1'], average=False)


#%%                          
#Rename column
df_s1_EDA.rename(columns={'GSR1': 'EDA'}, inplace=True)

# Transform EDA (participant 23-32 in Geneva) --> GSR geneva = 10**9 / GSR twente
if int(i) < 23:
    df_s1_EDA["EDA"] = (df_s1_EDA["EDA"])/10**9
else:
    df_s1_EDA["EDA"] = (10**9/df_s1_EDA["EDA"])*1000  
    
#Rename column
df_s1_filtered_EDA.rename(columns={'GSR1': 'EDA'}, inplace=True)

# Transform EDA (participant 23-32 in Geneva) --> GSR geneva = 10**9 / GSR twente
if int(i) < 23:
    df_s1_filtered_EDA["EDA"] = (df_s1_filtered_EDA["EDA"])/10**9
else:
    df_s1_filtered_EDA["EDA"] = (10**9/df_s1_filtered_EDA["EDA"])*1000  

#%%  
# Replace zeros with NaNs
#df_s1_EDA = df_s1_EDA.loc[df_s1_EDA['EDA'] > 0]

#%%
# Create column "time_min"
df_s1_EDA['time_min'] = (df_s1_EDA.time/1000)/60
# Create column "time_sec"
df_s1_EDA['time_sec'] = df_s1_EDA.time/1000

# Create column "time_min"
df_s1_filtered_EDA['time_min'] = (df_s1_filtered_EDA.time/1000)/60
# Create column "time_sec"
df_s1_filtered_EDA['time_sec'] = df_s1_filtered_EDA.time/1000

#df_s1_EDA = df_s1_EDA.head(100000)
#df_s1_filtered_EDA = df_s1_filtered_EDA.head(100000)


#ax = df_s1_EDA.plot.line(ylim=(0,20), x='time_min', y='EDA')
#ax.set_xlabel("Time(min)")
#ax.set_ylabel("Skin conductance(µS)")
#ax.set_title('Subject {}'.format(i))

#plt.savefig('/data/clean_EDA/Subject_{}_raw_EDA_last_seconds.png'.format(i))

# Plot all signals
#for df in testdf.groupby(by='subject'):
#    plt.plot(sub_df['Visit Number'], sub_df['dB'], label=subject)

# plot detrended signal v. 1
#df_s1_EDA['EDA_detreneded'] = signal.detrend(df_s1_EDA["EDA"])

t = df_s1_EDA['time_min']
x = df_s1_EDA['EDA']
x_detrended = df_s1_filtered_EDA['EDA']

plt.figure(figsize=(5, 4))
plt.plot(t, x, label="EDA")
plt.plot(t, x_detrended, label="EDA_detreneded")
plt.xlabel("Time(min)")
plt.ylabel("Skin conductance(µS)")
plt.legend(loc='best')
plt.show()


# plot detrended signal v. 2
# Sample rate and desired cutoff frequencies (in Hz).
#sensor_data = np.array(x)
#time = t
#plt.plot(time, sensor_data)
#plt.show()

#filtered_signal = bandPassFilter(sensor_data)
#plt.plot(time, filtered_signal)
#plt.show()

# Plot EDA: whole data v.2
#ax = df_s1_EDA.plot.line(title= 'subject'+number_subject, x='time_min', y='EDA')
##ax.set_xlabel("Time(min)")
#ax.set_ylabel("Skin conductance(µS)")
## ylim=(0,20) # mantain commented

#PSD EDA
#s1_temp.plot_psd(n_fft=512, fmin=0, fmax=10)



# %%
