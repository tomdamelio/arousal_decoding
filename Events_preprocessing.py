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

subject_number = ['05']

for i in subject_number: 
    # Read .fif files (with respiration annotations)
    directory = 'outputs/data/EDA+Resp_with_resp_annotations/'
    number_subject = i
    events_label = '_without_ITI'
    extension = '.fif'
    fname = op.join(directory + 's'+ number_subject + extension)

    raw_fif = mne.io.read_raw_fif(fname, preload=True) 


    # Read bdf files (without annotations)
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
    #print(raw_fif.ch_names)


    # DESCOMENTAR ESTO SI QUIERO VOLVER A GUARDAR TODO 
    # raw_fif.save(fname = fname_2, overwrite=True)


    # Create events based on stim channel
    events = mne.find_events(raw_fif, stim_channel='Status')

    # Select events with stim value == 4 --> start music stimulus
    rows=np.where(events[:,2]==4)
    events_4 = events[rows]

    mapping = { 1: 'rating_screen',
                2: 'video_synch',
                3: 'fixation_screen ',
                4: 'music_stim',
                5: 'fixation_screen_after',
                7: 'end_exp'}


    annot_from_events = mne.annotations_from_events(
        events=events, event_desc=mapping, sfreq=raw_fif.info['sfreq'],
        orig_time=raw_fif.info['meas_date'])
    raw_fif.set_annotations(annot_from_events)
    
    #%matplotlib
    #raw_fif.plot()


    # Crear una lista con los valores de la primera columna del ndarray
    # dividos por la freq. de sampleo -->  raw_fif.info['sfreq']
    # Cambiar el parametro de 'onset' por 5 segundos antes
    # Esto devuelve una lista con los valores en segundos de los stim=4

    #list_onset_stim = ((events_4[:,0]/raw_fif.info['sfreq'])-5).tolist()


    # Crear las annotations que correspondan con la presentacion
    # de estimulos musicales (40 stimuli)
    # Cambiar el parametro de 'onset' por 5 segundos antes y 3 despues de los
    # 40 segundos que dura los estimulos.

    #rem_annot = mne.Annotations(onset= list_onset_stim,
    #                            duration=[68.],
    #                            description=['STIM'] * 40)
    #raw_fif.set_annotations(rem_annot)


    # Save output fif file
    directory = 'outputs/data/EDA+EEG'
    fname2 = op.join(directory + 's'+ number_subject + events_label + extension)
    #raw_2 = mne.io.read_raw_fif(fname2, preload=True) 

    #%matplotlib
    #raw_2.plot()


    #events_label = '_without_ITI_prueba'
    #fname3 = op.join(directory + 's'+ number_subject + events_label + extension)
    #raw_fif.save(fname = fname3, overwrite=True)   

    # Intentar anotar todas las bad_epochs de forma programatica
    # Select events with stim value == 1 o 3 --> start music stimulus

    rows_1 = np.where(events[:,2] == 1)
    events_1 = events[rows_1]
    onset_stim_1 = events_1[:,0]/raw_fif.info['sfreq'] 
    
    # Elimino los los dos primeros valores del onset stim
    # (que no tiene que ver con el autoreporte)
    onset_stim_1 = onset_stim_1[2:]

    row_stim_1 = np.arange(0, len(onset_stim_1),4)
    onset_stim_1_unique = onset_stim_1[row_stim_1]

    rows_3 = np.where(events[:,2] == 3)
    events_3 = events[rows_3]
    onset_stim_3 = events_3[:,0]/raw_fif.info['sfreq']
    onset_stim_3 = onset_stim_3[1:]
    onset_stim_3 = np.append(onset_stim_3, (len(raw_fif)/raw_fif.info['sfreq']))

    diff_onset_stim_1_3 = onset_stim_3 - onset_stim_1_unique


    # Marcar tiempo pre-baseline
    # entre timpo 0 y rating screen (1)
    events_1_first = events[rows_1]
    onset_stim_1_first = events_1_first[:,0]/raw_fif.info['sfreq']
    # Elimino los los dos primeros valores del onset stim
    # (que no tiene que ver con el autoreporte)
    #onset_stim_1_first = onset_stim_1_first[0]

    diff_onset_stim_1_3 = np.append(onset_stim_1_first[0], diff_onset_stim_1_3)
    onset_stim_1_unique = np.append(0, onset_stim_1_unique)


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
    later_annot = mne.Annotations(onset=onset_stim_1_unique,
                                duration=diff_onset_stim_1_3,
                                description=['bad_no_stim']*41)
    # 39 estimulos, porque el ultimo coincide con la ultima parte del experimento
    raw2 = raw_fif.copy().set_annotations(later_annot)
    raw2.plot()

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



