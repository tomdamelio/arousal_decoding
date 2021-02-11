import mne
s1 = mne.io.read_raw_bdf("data/s01.bdf")
print(s1.info)
s1.plot(duration=5, n_channels=48)