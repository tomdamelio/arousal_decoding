import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import freqz
from filters import butter_bandpass, butter_bandpass_filter

# Read bdf
number_subject = '06' # Insert subject number
path = os.path.join('data', 's'+ number_subject + '.bdf')
s1 = mne.io.read_raw_bdf(path, preload=True)
# Print info of subject's signal
print(s1.info)

# Select EDA data
s1_temp = s1.copy()
print('Number of channels in s1_temp:')
print(len(s1_temp.ch_names), end=' → pick only EDA → ')
s1_temp.pick_channels(['GSR1'])
print(len(s1_temp.ch_names))

# Plot  EDA. Plot only first part of the signal
#%matplotlib qt
#s1_temp.plot(title='EDA' , scalings='auto')

# Plot the EDA power spectral density 
#s1_temp.plot_psd()

#Create dataframe of EDA subject 1
df_s1_EDA = s1_temp.to_data_frame()

#Rename column
df_s1_EDA.rename(columns={'GSR1': 'EDA'}, inplace=True)

# Transform EDA (participant 23-32 in Geneva) --> GSR geneva = 10**9 / GSR twente
if int(number_subject) < 23:
    df_s1_EDA["EDA"] = (df_s1_EDA["EDA"])/10**9
else:
    df_s1_EDA["EDA"] = (10**9/df_s1_EDA["EDA"])*1000  
    
# Create column "time_min"
df_s1_EDA['time_min'] = (df_s1_EDA.time/1000)/60
# Create column "time_sed"
df_s1_EDA['time_sec'] = df_s1_EDA.time/1000

# Detrend signal
df_s1_EDA['EDA_detreneded'] = signal.detrend(df_s1_EDA["EDA"])
# Plot detrended signal
t = df_s1_EDA['time_min']
x = df_s1_EDA['EDA']
x_detrended = df_s1_EDA['EDA_detreneded']

plt.figure(figsize=(5, 4))
plt.plot(t, x, label="EDA")
plt.plot(t, x_detrended, label="EDA_detreneded")
plt.legend(loc='best')
plt.show()

# plot detrended signal v. 2
# Sample rate and desired cutoff frequencies (in Hz).
fs = 512
lowcut = 0.05
highcut = 5.0

# Filter a noisy signal.
plt.figure()
plt.clf()
plt.plot(t, x, label='EDA')

y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
plt.plot(t, y, label='Filtered EDA')
plt.xlabel('time (minutes)')
plt.grid(True)
plt.axis('tight')
plt.legend(loc='upper left')

plt.show()

# Plot EDA: whole data v.2
#ax = df_s1_EDA.plot.line(title= 'subject'+number_subject, x='time_min', y='EDA')
##ax.set_xlabel("Time(min)")
#ax.set_ylabel("Skin conductance(µS)")
## ylim=(0,20) # mantain commented

#PSD EDA
#s1_temp.plot_psd(n_fft=512, fmin=0, fmax=10)


