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


def save_bad_resp_and_no_stim_annotations (subject_number  = subject_number,
                                           save_EDA_EEG_bad_resp = False,
                                           plot_EDA_EEG_bad_resp = False,
                                           plot_events = False,
                                           plot_EDA_EEG_bad_no_stim = False,
                                           save_EDA_EEG_bad_no_stim = False,
                                           save_just_no_stim_annotations = False,
                                           plot_all_annotations  = False,
                                           save_all_annotations = False,
                                           save_just_annotations = False,
                                           add_annotations_to_original_files = False):
    
    """ First, join EDA, EEG and bad respiration annotations.
    Then, mark and  remove all the events that do not correspond to stimulus
    presentation (e.g. pre-experiement, self-reporting, inter-block interval,
    after last stimulus).
    Possibility of save raw files with annotations, or just save annotations/
    This last option is going to be what we need to include this process into 
    BIDS pipeline (by including afterwards the annotations files into de raw
    'BIDS organized' data).
    
    Parameters
    ----------

    ### subject_number:  list
        Subjects to analyze.
        Default = subject_number --> all subjects
    ### save_EDA_EEG_bad_resp: Boolean
        Save EDA, EEG and bad respiraition annotations .fif file
        Default = False
    ### plot_EDA_EEG_bad_resp: Boolean
        Plot EDA and EEG with bad respiraition annotations.
        Default = False
    ### plot_events: Boolean
        Plot all channels with events marks.
        Default = False
    ### plot_EDA_EEG_bad_no_stim: Boolean
        Plot all channles with bad_no_stim annotations.
        Default = False
    ### save_EDA_EEG_bad_no_stim: Boolean
        Save all channels with bad_no_stim annotations. 
        Default = False
    ### plot_all_annotations: Boolean
        Plot all channles with bad_no_stim and resp annotations.
        Default = False
    ### save_all_annotations: Boolean
        Save all channels with bad_no_stim and resp annotations.
        Default = False
    ### save_just_annotations: Boolean
        Save onlye the annotations (bad_no_stim and resp annotations)
        without saving Raw data of subjects.
        Default = False
    ### add_annotations_to_original_files: Boolean
        Add annotations to original files
        Default = False
        
   Returns
    -------
   This would change depending on the paramters' setting.
   Possibilities: plots, raw data with annotations, or just the annotations.
   The most important output would be to obtain the annotations to add after
   to original files (if 'save_just_annotations' parameter is set as True).
        
                            
    """ 
    for i in subject_number: 
        
        # Read .fif files (already with respiration annotations).
        # This annotations were made manually (by visual inspection)
        directory = 'outputs/data/EDA+Resp_with_resp_annotations/'
        
        number_subject = i
        extension = '.fif'
        fname = op.join(directory + 's'+ number_subject + extension)

        raw_fif = mne.io.read_raw_fif(fname, preload=True) 

        # Read bdf files (without annotations) --> all channels
        extension = '.bdf'
        directory = 'data/'
        
        #Read bdf
        fname_bdf = op.join(directory + 's' + number_subject + extension)
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
        raw_fif.filter(0.1, 1, picks='emg')

        # plot EDA and resp
        if plot_EDA_EEG_bad_resp == True:  
            fig_plot_EDA_EEG_bad_resp = raw_fif.plot(order=[1,0], scalings=dict(misc='1e-1', emg='1e-2'), start=394)
            return fig_plot_EDA_EEG_bad_resp
        
        # In case in want to resave EDA + EEG + bad_resp annotations
        if save_EDA_EEG_bad_resp == True:
            extension = '.fif'
            directory = 'outputs/data/EDA+EEG+bad_resp/'
            fname_2 = op.join(directory,'s'+ number_subject + extension)
            raw_fif.save(fname = fname_2, overwrite=True)
            
        #######################################################################

        # Some subjects had problems with channel names and channel types
        # that is necessary to fix
        if int(i) > 28:
            raw_fif.rename_channels(mapping={'-1': 'Status'} )
            raw_fif.drop_channels('-0')
        
        elif int(i) > 23:
            raw_fif.rename_channels(mapping={'': 'Status'} )
        
        # Create events based on stim channel
        events = mne.find_events(raw_fif, stim_channel='Status')
        
        
        # ID status assigning based on DEAP documentation (subject 01 to 23)
        if int(i) < 24:
            event_id = { 1: 'rating_screen',
                        2: 'video_synch',
                        3: 'fixation_screen ',
                        4: 'music_stim',
                        5: 'fixation_screen_after',
                        7: 'end_exp'}
        
        # ID status assigning based on visual inspection and comparison with subjects'
        # ID status (subject 24 to 32)
        else:
            event_id = { 1638145: 'rating_screen',
                        1638149: 'fixation_screen_after ',
                        1638147: 'fixation_screen',
                        1638148: 'music_stim',
                        1638146: 'video_synch',
                        1638151: 'end_exp',
                        }
        
        # Create Annotation object based on events' ID    
        annot_from_events = mne.annotations_from_events(
            events=events, event_desc=event_id, sfreq=raw_fif.info['sfreq'],
            orig_time=raw_fif.info['meas_date'])
        
        # Set annotations' iD in raw files
        raw_fif.set_annotations(annot_from_events)
        
        # Plot events' annotations
        if plot_events == True:
            raw2.plot()

        # Programmatically annotate bad signal (signal to drop when epoching)
        """
        bad signal annotate as 'bad_no_stim':
         - Time before the first trial
         - Time between trials 
         - Time between blocks (only one bad_no_stim in each subject)
         - Time after the experimental task
        
        Select:
        - events with stim value == 5 [fix after] or 3 [fixation] for subject 01 to 23
        - events with stim value == 1638149 [fix after] or 1638147 [fixation] for subject 24 to 32
        Between this two events -->  music stimulus
        
        """
        # Select indices of 'fixation screen after trial' events 
        if int(i) < 24:
            # 5 -> fixation screen after trial ID for subject 01 to 23
            rows_fix_after = np.where(events[:,2] == 5)
        else:
            # 1638149 -> fixation screen after trial ID for subject 24 to 32
            rows_fix_after = np.where(events[:,2] == 1638149)
        
        # Select events of fixation screen after trial  based on indices       
        events_fix_after= events[rows_fix_after]
        # As we have these events in datapoints, we have to leverage considering
        # sample frequency
        onset_fix_after = events_fix_after[:,0]/raw_fif.info['sfreq']     

        # Select indices of 'fixation screen before beginning of trial' events 
        if int(i) < 24:
            rows_fix_before = np.where(events[:,2] == 3)
        else:
            rows_fix_before = np.where(events[:,2] == 1638147)
        
        # Select events of fixation screen before trial based on indices    
        events_fix_before = events[rows_fix_before]
        # As we have these events in datapoints, we have to leverage considering
        # sample frequency
        onset_stim_fix_before = events_fix_before[:,0]/raw_fif.info['sfreq']
        
        # Select from the 2nd event, becasue then we are going to substract
        # the onset of the fixation cross before stimuli 2 and the onset of the
        # fixation cross of stimuli 1, to obtain duration of bad annotations.
        onset_stim_fix_before_2 = onset_stim_fix_before[1:]
        
        # For subjects 23 to 32, there are more annotations at the beginning
        # of the register. We have to remove them, to make the algorithm work.
        if int(i) > 22:
            # Delete an extra value at the beginning of  onset_stim_fix_before_2 
            onset_stim_fix_before_2 = onset_stim_fix_before_2[1:]
            # Delete two  extra value at the beginning of onset_stim_fix_after_2 
            onset_fix_after = onset_fix_after[2:]
            
        # Delete index 22 from onset_fix_after, because instead of 'videosync' event
        # there is an extra fixation mark at index '22' in subject 28
        if int (i) == 28:
            onset_stim_fix_before_2 = np.delete(onset_stim_fix_before_2, 22)

        # Timestamp (in secs) of the onset of 'fixation cross before stimulus' event     
        onset_stim_fix_before_2 = np.append(onset_stim_fix_before_2,
                                           (len(raw_fif)/raw_fif.info['sfreq']))

        # substract the onset of the fixation cross before stimuli 2 and the onset of the
        # fixation cross of stimuli 1, to obtain duration of bad annotations
        # (interstimulus interval)
        diff_onset_music_stim = onset_stim_fix_before_2 - onset_fix_after

        # Select event before first music stimulus. The onset of the fixation cross before
        # the first stimulus correspond to the durecion of the first segment of be marked as 'bad'
        if int(i) > 22:
            # In subjects 23 to 32 there is one extra fixation cross that we want to dismiss
            diff_onset_music_stim = np.append(onset_stim_fix_before[1], diff_onset_music_stim)
        else:
            diff_onset_music_stim = np.append(onset_stim_fix_before[0], diff_onset_music_stim)

        # Add '0' as first event, because we want to annotate from the beginning of the registration
        # to the start of the first music stimulus as bad segment    
        onset_music_stim= np.append(0, onset_fix_after)
        
        if int(i) == 28:
            # 37 stimuli + 1 events (beginning of the registration)
            n_stims = 37+1
        else:
            # 40 stimuli + 1 events (beginning of the registration)
            n_stims = 40+1
        
        # Create annotations corresponding to moments in which no stimuli are presented
        later_annot = mne.Annotations(onset=onset_music_stim,
                                    duration=diff_onset_music_stim,
                                    description=['bad_no_stim']*n_stims)

        # Create raw file with 'bad_no_stim' annotations
        raw2 = raw_fif.copy().set_annotations(later_annot)
        
        # Plot EDA, EEG and 'bad_no_stim' annotations
        if plot_EDA_EEG_bad_no_stim == True:
             fig_plot_EDA_EEG_bad_no_stim = raw2.plot(start=2488)
             return fig_plot_EDA_EEG_bad_no_stim
            
        # Save EDA, EEG and 'bad_no_stim' file with annotations
        if save_EDA_EEG_bad_no_stim ==True:
            extension = '.fif'
            directory = 'outputs/data/EDA+EEG+bad_no_stim/'
            fname_3 = op.join(directory,'s'+ number_subject + extension)
            raw2.save(fname = fname_3, overwrite=True)
            
        # # Save 'bad_no_stim' annotations
        if save_just_no_stim_annotations == True:
            extension = '.fif'
            directory = 'outputs/data/annotations_bad_no_stim/'
            fname_5 = op.join(directory,'sub-'+ number_subject + '_annotations' + extension)
            later_annot.save(fname = fname_5, overwrite=True)
                
        # Create raw file with EEG, EDA and 'bad_respirations' annotations
        # based on previous manual annotation (with visual inspection) 
        directory = 'outputs/data/EDA+Resp_with_resp_annotations'
        number_subject = i
        extension = '.fif'
        fname = op.join(directory,'s'+ number_subject + extension)
        raw_fif = mne.io.read_raw_fif(fname, preload=True) 
        later_annot.append(raw_fif.annotations.onset,
                           raw_fif.annotations.duration,
                           raw_fif.annotations.description)
        raw2.set_annotations(later_annot)
        
        if plot_all_annotations == True:
            # Plot annotations: a) respiration, b) no stimuli presentation
            raw2.plot()

        if save_all_annotations == True:
            # Save raw files with EEG, EDA and annotations (respiration and no stimuli annotations)
            extension = '.fif'
            directory = 'outputs/data/EDA+EEG+bad_no_stim+bad_resp/'
            fname_4 = op.join(directory,'s'+ number_subject + extension)
            raw2.save(fname = fname_4, overwrite=True)
        
        if save_just_annotations == True:
            # Save only the annotations (respiration and no stimuli annotations) 
            extension = '.fif'
            directory = 'outputs/data/annotations_bad_no_stim+bad_resp/'
            fname_5 = op.join(directory,'sub-'+ number_subject + '_annotations' + extension)
            later_annot.save(fname = fname_5, overwrite=True)
        
        
        if add_annotations_to_original_files == True:
            # add annotations to original files
            extension = '.bdf'
            directory = 'data/'
            fname_bdf = op.join(directory + 's'+ number_subject + extension)
            raw_bdf = mne.io.read_raw_bdf(fname_bdf, preload=True)  
            raw_bdf.set_annotations(later_annot)
            # Save annotated original files
            directory_2 = 'outputs/data/data_with_annotations/'
            extension_2 = '.fif'
            # e.g. save as 'sub-01_annotations.fif' raw + annotations from subject 1
            fname_7 = op.join(directory_2,'sub-'+ number_subject + 'data_and_annotations' + extension_2)
            raw_bdf.save(fname = fname_7, overwrite=True)

                                                    
save_bad_resp_and_no_stim_annotations(save_just_no_stim_annotations= True)


# %%
