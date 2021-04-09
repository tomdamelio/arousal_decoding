# Mark and then remove all the events that don't correspond to stimulus
# presentation (e.g. inter-block interval, self-reporting, etc).

'''
Steps:
1) Plot one subject with stims
2) Understand code of stims
3) mark bad epochs as:
    a)  bad_inter_block_inteval
    b)  bad_self_report
    c)  bad_eda_artifact
4) Save in other folder this step
5) Insert in function run_all_subjects clean subjects

'''
#%%
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

from preprocessing import extract_signal, transform_negative_to_zero, out_of_range, get_rejection_threshold
from channel_names import channels_geneva, channels_twente 
import os.path as op
from subject_number import subject_number

subject_number = '32'

# Read bdf files (without annotations)
directory = 'data/'
number_subject = subject_number
extension = '.bdf'

#Read bdf
fname_bdf = op.join(directory + 's'+ number_subject + extension)
raw = mne.io.read_raw_bdf(fname_bdf, preload=True) 

#%%

mne.rename_channels(info= raw.info , mapping={'GSR1':'EDA'})

# DESCOMENTAR ESTO CUANDO ANALIZO SUJETOS 24 PARA ADELANTE
# Del sujeto 24 a 28, el valor a mappear es un espacio ' '.
# Del sujeto 29 a 32, el valor a mappear es un espacio '-1' y tengo que
# eliminar el canal '-0'
raw.rename_channels(mapping={'-1': 'Status'} )

raw.set_channel_types({ 'EXG1'  : 'eog',
                        'EXG2'  : 'eog',
                        'EXG3'  : 'eog',
                        'EXG4'  : 'eog',
                        'EXG5'  : 'emg',
                        'EXG6'  : 'emg',
                        'EXG7'  : 'emg',
                        'EXG8'  : 'emg',
                        'EDA'   : 'emg', # select as emg to make it easier
                        'GSR2'  : 'misc',
                        'Erg1'  : 'misc',
                        'Erg2'  : 'misc',
                        'Resp'  : 'misc',
                        'Plet'  : 'misc',
                        'Temp'  : 'misc',
                        'Status': 'stim' })

raw = raw.pick_types(stim = True, eeg = True, eog = True,
                              misc = False, emg = False)

#%matplotlib
#raw.plot()

#%%
# print unique values from "possible stim" channel ('' channel)
import pandas as pd
status_array = raw.get_data()[-1] #/raw.info['sfreq']
print(np.where(status_array == 1703680))

#%%
print(dict(pd.Series(status_array).value_counts()))
    
#%%
# Create events based on stim channel
events = mne.find_events(raw, stim_channel='Status')

mapping = { 1638145: '1638145',
            1638149: '1638149 ',
            1638147: '1638147',
            1638148: '1638148',
            1703680: '1703680',
            1638146: '1638146',
            1638151: '1638151',
            }


annot_from_events = mne.annotations_from_events(
    events=events, event_desc=mapping, sfreq=raw.info['sfreq'],
    orig_time=raw.info['meas_date'])
raw.set_annotations(annot_from_events)

%matplotlib
raw.plot()
# %%
