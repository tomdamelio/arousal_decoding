
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import Epochs
from mne.decoding import SPoC
from mne.datasets.fieldtrip_cmc import data_path
from mne import pick_types

from autoreject import get_rejection_threshold
from autoreject.autoreject import _GlobalAutoReject
from autoreject.bayesopt import expected_improvement, bayes_opt

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, check_cv

from my_functions import extract_signal, transform_negative_to_zero, out_of_range, get_rejection_threshold
from channel_names import channels_geneva, channels_twente 

from subject_number import subject_number

reject_values = []
epochs_rejected_values = []
subject_number_reduced = ['01','08', '16', '24', '32']
for number_subject in subject_number_reduced:
    #Extract signal
    raw = extract_signal(directory = 'data', number_subject=number_subject,
                        extension = '.bdf')

    # Rename channel EDA and set GSR as channel type
    raw.rename_channels(mapping={'GSR1':'EDA'})
    # raw.ch_names # Reutrn list of channels

    raw.set_channel_types({'EXG1': 'eog',
                        'EXG2': 'eog',
                        'EXG3': 'eog',
                        'EXG4': 'eog',
                        'EXG5': 'emg',
                        'EXG6': 'emg',
                        'EXG7': 'emg',
                        'EXG8': 'emg',
                        'EDA' : 'misc',
                        'GSR2': 'misc',
                        'Erg1': 'misc',
                        'Erg2': 'misc',
                        'Resp': 'misc',
                        'Plet': 'misc',
                        'Temp': 'misc'})

    raw.drop_channels(['GSR2', 'EXG5', 'EXG6', 'EXG7', 'EXG8',
                       'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp' ])
    
    # Pick EDA and EEG
    picks_eda = mne.pick_channels(ch_names = raw.ch_names ,include=['EDA'])
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False)

    # Clean data 

    # 1)  Transform EDA (depending on recording procedure) --> 
    #     http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html

    if int(number_subject) < 23:
        raw.apply_function(fun=lambda x: x/1000, picks=picks_eda)
    else:
        raw.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)

    # 2) Clean signals --> 
    #    -  Negative values            ==> subjects 01 02 03 08 14 15
    raw.apply_function(fun=transform_negative_to_zero, picks=picks_eda)

    #    -  Out-of-range values        ==> 26
    #raw.apply_function(fun=out_of_range, picks=picks_eda)

    # Filter EDA 
    raw.filter(0.05, 5., fir_design='firwin', picks=picks_eda)

    # FIlter EEG
    raw.filter(0., 50., fir_design='firwin', picks=picks_eeg)

    # Downsample to 250 Hz 
    #raw.resample(250.) 

    # Build epochs as sliding windows over the continuous raw file
    events = mne.make_fixed_length_events(raw, id=1, duration=5., overlap= 2.)


    epochs = Epochs(raw, events, tmin=0., tmax=10., baseline=None)
    #eda_epochs = Epochs(raw=raw_eda, events=events, tmin=0., tmax=0., baseline=None)

    # Autoreject 
    reject = get_rejection_threshold(epochs, decim=1, verbose=False)

    reject.update({'misc': 10.}) # 3 times typical phasic incrase in conductance (Boucsein, 2012)

    reject_values.append(reject)
    
    # Reject bad epochs
    epochs.drop_bad(reject=reject)
    epochs_rejected_values.append([events.shape[0], epochs.__len__()])
    
print(reject_values)

# n epochs total
epochs_rejected_values_total = [item[0] for item in epochs_rejected_values]

# n epochs after rejection
epochs_rejected_values_rejected = [item[1] for item in epochs_rejected_values]

x = np.arange(len(subject_number_reduced))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, epochs_rejected_values_total, width, label='Total')
rects2 = ax.bar(x + width/2, epochs_rejected_values_rejected, width, label='After rejection')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Epochs')
ax.set_title('Number of epochs (total and after rejection)')
ax.set_xticks(x)
ax.set_xticklabels(subject_number_reduced)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)

fig.set_size_inches(10., 5.)

plt.show()


#with open("epochs_rejected_values_rejected.txt", "w") as output:
#    output.write(str(epochs_rejected_values_rejected))
    
#with open("epochs_rejected_values_total.txt", "w") as output:
#    output.write(str(epochs_rejected_values_total))

#with open("reject_values.txt", "w") as output:
#    output.write(str(reject_values))

#%%
#with open("epochs_rejected_values_rejected.txt") as f:
#    epochs_rejected_values_rejected = f.read().splitlines()

#with open("epochs_rejected_values_total.txt") as f:
#    epochs_rejected_values_total = f.read().splitlines()

#with open("reject_values.txt") as f:
#    reject_values = f.read().splitlines()

#misc_values = []
#eog_values = []
#eeg_values = []
#for i in reject_values:
#    eeg_values.append(list(i.values())[0])
#    eog_values.append(list(i.values())[1])
#    misc_values.append(list(i.values())[2])

#plt.hist(misc_values)
