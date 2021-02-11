import mne
from EEG_channels_twente import channels 

#Read bdf
s1 = mne.io.read_raw_bdf("data/s01.bdf")
print(s1.info)

# Select only EDA data
s1_temp = s1.copy()
print('Number of channels in s1_temp:')
print(len(s1_temp.ch_names), end=' → pick only EDA → ')
s1_temp.pick_channels(['GSR1'])
print(len(s1_temp.ch_names))
# Plot  EDA. Doesn't work well.
s1_temp.plot()
# Plot the EDA power spectral density 
s1_temp.plot_psd()
# Create dataframe of EDA subject 1
df_s1_EDA = s1_temp.to_data_frame()

# Select only EEG data
s1_temp2 = s1.copy()
print('Number of channels in s1_temp:')
print(len(s1_temp2.ch_names), end=' → pick only EEG → ')
s1_temp2.pick_channels(channels)
print(len(s1_temp2.ch_names))
# Plot EEG 
s1_temp2.plot()
# Plot the EEG power spectral density 
s1_temp2.plot_psd()
# Create dataframe EEG subject 1
df_s1_EEG = s1_temp2.to_data_frame()

