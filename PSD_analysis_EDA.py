#%%
import os
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from subject_number import subject_number
from my_functions import extract_signal

for i in subject_number:
    # Extract signal
    subject_n_temp = extract_signal(signal = 'EDA', number_subject = i)

    # Filter signal
    s1_filtered = subject_n_temp.filter(0.05, 5., fir_design='firwin')

    # PSD
    fig = s1_filtered.plot_psd(n_fft=512, fmin=0.01, fmax=5)#, xscale = 'log')

    # Guardar en 
    fig.savefig('subject_plots/PSD/plot_S{}.png'.format(i))
# %%
