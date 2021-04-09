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

subject_number = ['22']

for i in subject_number: 
    # Read .fif files (with respiration annotations)
    directory = 'outputs/data/EDA+Resp_with_resp_annotations/'
    number_subject = i
    extension = '.fif'
    fname = op.join(directory + 's'+ number_subject + extension)

    raw_fif = mne.io.read_raw_fif(fname, preload=True) 


    # Read bdf files (without annotations) --> all channels
    extension = '.bdf'
    directory = 'data/'
    #Read bdf
    fname_bdf = op.join(directory + 's'+ number_subject + extension)
    raw_bdf = mne.io.read_raw_bdf(fname_bdf, preload=True) 

    mne.rename_channels(info= raw_bdf.info , mapping={'GSR1':'EDA'})

    raw_bdf.set_channel_types({ 'EXG1': 'eog',
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


    # Pick STIM, EEG and EOG from .bdf
    raw_bdf2 = raw_bdf.pick_types(stim = True, eeg = True, eog = True,
                                misc = False, emg = False)

    # Add channels from raw_bdf (stim, EEG and EOG) to raw_fif (resp and EDA)
    raw_fif.add_channels([raw_bdf2])

    # Save .fif file
    extension = '.fif'
    directory = 'outputs/data/EDA+EEG+bad_resp/'
    fname_2 = op.join(directory,'s'+ number_subject + extension)
        
    #%matplotlib
    #raw_fif.plot(order=[1,0], scalings=dict(misc='1e-1', emg='1e-1'))
    #raw_fif.plot()

    # Decomment in case in want to resave EDA + EEG + bad_resp annotations
    # raw_fif.save(fname = fname_2, overwrite=True)

    if int(i) > 28:
        raw_fif.rename_channels(mapping={'-1': 'Status'} )
        raw_fif.drop_channels('-0')
    
    elif int(i) > 23:
        raw_fif.rename_channels(mapping={'': 'Status'} )
    
    # Create events based on stim channel
    events = mne.find_events(raw_fif, stim_channel='Status')

    # Select events with stim value == 4 --> start music stimulus
    #rows=np.where(events[:,2]==4)
    #events_4 = events[rows]
    
    if int(i) < 24:
        mapping = { 1: 'rating_screen',
                    2: 'video_synch',
                    3: 'fixation_screen ',
                    4: 'music_stim',
                    5: 'fixation_screen_after',
                    7: 'end_exp'}
    else:
        mapping = { 1638145: 'rating_screen',
                    1638149: 'fixation_screen_after ',
                    1638147: 'fixation_screen',
                    1638148: 'music_stim',
                    1638146: 'video_synch',
                    1638151: 'end_exp',
                    }

    annot_from_events = mne.annotations_from_events(
        events=events, event_desc=mapping, sfreq=raw_fif.info['sfreq'],
        orig_time=raw_fif.info['meas_date'])
    raw_fif.set_annotations(annot_from_events)
    
    %matplotlib
    raw_fif.plot()

    ##### Programmatically annotate bad signal ####
    # bad signal annotate as 'bad_no_stim':
    # - Time before the first trial
    # - Time between trials 
    # - Time between blocks (only one bad_no_stim in each subject)
    # - Time after the experimental task
    
    # Select events with stim value == 5 [fix after] or 3 [fixation]
    # Between this two events -->  music stimulus
    
    # Select events of fixation screen after trial [5]
    rows_rating = np.where(events[:,2] == 5)
    events_rating = events[rows_rating]
    onset_music_stim = events_rating[:,0]/raw_fif.info['sfreq'] 
    
    # Delete first two values from onset_stim_rating
    # (has nothing to do with rating)
    #onset_stim_rating = onset_stim_rating[2:]

    # As we have 4 rating screens per music stimulus, we will select 1
    # value per every 4 rating screen values
    #row_music_stim = np.arange(0, len(onset_stim_rating),4)

    # Select events of fixation screen before beginning of trial [3]
    rows_fix_before = np.where(events[:,2] == 3)
    events_fix_before = events[rows_fix_before]
    onset_stim_fix_before = events_fix_before[:,0]/raw_fif.info['sfreq']
    onset_stim_fix_before_2 = onset_stim_fix_before[1:]
    onset_stim_fix_before_2 = np.append(onset_stim_fix_before_2, (len(raw_fif)/raw_fif.info['sfreq']))

    diff_onset_music_stim = onset_stim_fix_before_2 - onset_music_stim

    # Select events pre first music stimulus
    diff_onset_music_stim = np.append(onset_stim_fix_before[0], diff_onset_music_stim)
    onset_music_stim= np.append(0, onset_music_stim)

    # Agregar annotations por respiration
    #raw_2 = mne.io.read_raw_fif(fname, preload=True) 
    #raw_2.plot(order=[1,0], scalings=dict(misc='1e-1', emg='1e-1'))


    # Extraer onset (dividiendo por frecuencia)
    #resp_onset = raw_2.annotations.onset
    # Extraer frecuencia (dividiendo por frecuencia)
    #resp_duration = raw_2.annotations.duration
    # Appendearlo a las listas correspondientes abajo

    # Correr set_annotations y plot

    # https://mne.tools/dev/auto_tutorials/raw/plot_30_annotate_raw.html
    later_annot = mne.Annotations(onset=onset_music_stim,
                                duration=diff_onset_music_stim,
                                description=['bad_no_stim']*41)

    raw2 = raw_fif.copy().set_annotations(later_annot)
    raw2.plot()

#%%

    extension = '.fif'
    directory = 'outputs/data/EDA+EEG+bad_no_stim/'
    fname_3 = op.join(directory,'s'+ number_subject + extension)
    raw2.save(fname = fname_3, overwrite=True)

#%%
######################################

# Making multiple events per annotation
# Crear multiples eventos 
# https://mne.tools/stable/auto_tutorials/intro/plot_20_events_from_raw.html#tut-events-vs-annotations

# (music_events,
# music_event_dict) = mne.events_from_annotations(raw_fif, chunk_duration=10.)


# Epoche considering this epochs, from -5 secs (fix) to 63 sec (stim + postfix)
# epochs = Epochs(raw=raw_fif, events=music_events, tmin=0., tmax=68., baseline=None)

#events_from_annot, event_dict = mne.events_from_annotations(raw_bdf)
#print(event_dict)
#print(events_from_annot[:5])

# Find events with status code == 4
#events = mne.find_events(raw_bdf, stim_channel='Status')
#epochs = mne.Epochs(raw_bdf, events=events, event_id=None, tmin=-5., tmax=60.)
#epochs['4']



