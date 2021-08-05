'''
El objetivo de este script es extraer un array de dos dimensiones de
epochs x stimuli -> len = (epochs_per_trial, 40)

'''

import os
import os.path as op
import pandas as pd
import numpy as np
import pathlib
from subject_number import subject_number as subjects
import mne
from mne_bids import BIDSPath

###  SET CONFIGS ###

# eda or emg?    
measure = 'eda'
# var or mean?   
y_stat = 'var'

DEBUG = False
####################

if os.name == 'nt':
    data_root = pathlib.Path(
        '~/OneDrive/Escritorio/tomas_damelio/data').expanduser()
else:
    data_root = pathlib.Path(
        "/storage/store2/work/tdamelio/tomas_damelio/data")

fname_ratings = op.join(data_root, 'participant_ratings.csv')

ratings = pd.read_csv(fname_ratings)

# S28  has 37 stimuli -> not possible to map
ratings.drop(ratings[ratings.Participant_id == 28].index, inplace=True)
subjects.remove('28')

if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg

if os.name == 'nt':
    derivative_path = cfg.deriv_root
    derivative_path_3 = cfg.deriv_root  
    eda_root = derivative_path_3 / 'eda_var_y_and_y_pred_scores'
else:
    derivative_path = cfg.deriv_root
    derivative_path_3 = cfg.deriv_root_store3
    eda_root = derivative_path_3 / f'{measure}_scores--30-07--05-36-{y_stat}'
    
ratings['y'] = np.nan
ratings['y_pred'] = np.nan

for subject in subjects:
    if os.name == 'nt':
        fname_epochs = derivative_path / 'clean-epo-files-eda'
        epochs = mne.read_epochs(op.join(fname_epochs, 'sub-' + subject +
                                    '_task-rest_proc-clean_epo.fif'))

    else: 
        epochs_path = BIDSPath(subject= subject, 
                                task='rest',
                                datatype='eeg',
                                root=derivative_path,
                                processing='clean',
                                extension='.fif',
                                suffix='epo',
                                check=False)
        # read epochs
        epochs = mne.read_epochs(epochs_path)

    epochs_and_annots = pd.DataFrame(epochs.drop_log)

    # Rename column
    epochs_and_annots.columns = ['annotation']
    # Set index as column
    epochs_and_annots['n_epoch'] = epochs_and_annots.index
    # Check number of Null values (final epochs after rejection)
    #epochs_and_annots['annotation'].isnull().sum() #596 -> equal to number of epochs!
    
    # Generar una nueva columna en epochs_and_annots (stimuli) con rango de 1 a 40 (numero de estimulos).
    # Va a incrementar el valor de esa columna (partiendo de 0) cuando pase de bad_no_stim a algo que no sea 
    # bad_no_stim
    # En cualquier otro caso el valor queda igual
    j = 0
    epochs_and_annots['stimulus'] = np.nan
    for i in range(len(epochs_and_annots)-1):
        if epochs_and_annots.loc[i, 'annotation'] == 'bad_no_stim' and epochs_and_annots.loc[i+1, 'annotation'] != 'bad_no_stim':
            j += 1
        epochs_and_annots.loc[i, 'stimulus'] = j
        
    # Substear para quedarme unicamente con los valores de annotation que sea null
    epochs_and_annots = epochs_and_annots[epochs_and_annots.annotation.isnull()]
    epochs_and_annots.drop('annotation', axis=1, inplace=True)
    epochs_and_annots['idx'] = list(range(0,len(epochs)))

    # Crear resumen de la señal de EDA real (media de la varianza)
    eda_scores =  np.load(op.join(eda_root,
                                'sub-' + subject + '_y_and_y_pred_opt_models_eda_var_.npy'), allow_pickle=True)

    # create variables y and y_pred (Riemann)
    y = eda_scores.item()['y']
    y_pred = eda_scores.item()['riemann']

    # create dict (1,40) with number of epochs that correspond to each stimuli
    epochs_dict = dict()
    for ii in range(1, 41):
        idx = epochs_and_annots.stimulus == ii
        epochs_dict[ii] = list(epochs_and_annots.loc[idx, "idx"])

    for key, value in epochs_dict.items():
        ratings.loc[(ratings.Trial == key) & (ratings.Participant_id == int(subject)), 'y'] = np.mean(y[value])
        ratings.loc[(ratings.Trial == key) & (ratings.Participant_id == int(subject)), 'y_pred'] = np.mean(y_pred[value])

ratings.to_csv(op.join(eda_root, 'sub-' + subject + 'ratings_and_y.csv'))

    
