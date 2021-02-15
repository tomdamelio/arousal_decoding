import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Read bdf
number_subject = '05' # Insert subject number
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

# Filter signal
##  Samples per second
dt = 512
## Time
t = df_s1_EDA.loc[:,'time_sec']
## Signal to filter
f = df_s1_EDA.loc[:,'EDA']
## Lenght signal (in data points)
n = df_s1_EDA.shape[0]
## Compute the FFT
fhat = np.fft.fft(f, n)
## PSD
PSD = fhat * np.conj(fhat) / n
## Frequencies
freq = 1/(dt*n) * np.arange(n)

L = np.arange(1,np.floor(n/2),dtype='int')

fig, axs = plt.subplots (2,1)

plt.sca(axs[0])
plt.plot(t,f,color='c', LineWidth=1.5, label='EDA signal')
plt.xlim (t.iloc[0], t.iloc[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color='c', LineWidth=2, label = 'EDA signal')
plt.xlim(freq[L[0], freq[L[-1]]])
plt.legend()
    
# Plot EDA: whole data v.2
#ax = df_s1_EDA.plot.line(title= 'subject'+number_subject, x='time_min', y='EDA')
##ax.set_xlabel("Time(min)")
#ax.set_ylabel("Skin conductance(µS)")
## ylim=(0,20) # mantain commented

#PSD EDA
#s1_temp.plot_psd(n_fft=512, fmin=0, fmax=10)


