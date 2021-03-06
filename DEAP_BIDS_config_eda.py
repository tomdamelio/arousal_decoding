import importlib
import pathlib
import functools
import os
import pdb
import traceback
import sys
import copy
import logging

from typing import Optional, Union, Iterable, List, Tuple, Dict, Callable 

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import coloredlogs
import numpy as np
import mne
from mne_bids.path import get_entity_vals

study_name = 'DEAP'  

if os.name == 'nt':
    bids_root = pathlib.Path(
        "~/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids").expanduser()
    annotations_root = pathlib.Path(
        "~/OneDrive/Escritorio/tomas_damelio/outputs/data/annotations_bad_no_stim").expanduser()
    deriv_root = pathlib.Path(
        '~/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline-eda').expanduser()
    N_JOBS = 4

else:
    bids_root = pathlib.Path(
        "/storage/store2/derivatives/DEAP-bids")
    deriv_root = pathlib.Path(
        "/storage/store2/work/tdamelio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline-eda")
    deriv_root_store3 = pathlib.Path(
        "/storage/store3/work/tdamelio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline-eda")
    annotations_root = pathlib.Path("/storage/store2/work/tdamelio/tomas_damelio/outputs/data/annotations_bad_no_stim")
    N_JOBS = 15

interactive = False

subjects = 'all' 

task = 'rest' 

ch_types = ['eeg']

data_type = 'eeg'

drop_channels = ['-1', '-0', 'GSR2', 'Erg1', 'Erg2']

eeg_reference = [] 

eeg_template_montage: 'biosemi32'

analyze_channels = 'all'

conditions = ['rest']

filter_freq = dict(eeg=(0.1, 49.), misc=(0.05, 5.))

epochs_tmin = 0.

epochs_tmax = 5.

fixed_length_epochs_duration = 5.

fixed_length_epochs_overlap = 1.

baseline =  None 

spatial_filter = 'ssp'

n_proj_ecg_grad = 0

n_proj_ecg_mag = 0

n_proj_ecg_eeg = 0

n_proj_eog_grad = 0

n_proj_eog_mag = 0

n_proj_eog_eeg = 1

ssp_reject_eog = 'auto'

ssp_autoreject_decim = 5

reject = 'local'

reject_tmin = None

reject_tmax: Optional[float] = None

autoreject_decim = 4

decode = False

interpolate_bads_grand_average = True

run_source_estimation = False

l_trans_bandwidth = 'auto'

h_trans_bandwidth = 'auto'

random_state = 42

shortest_event = 1

log_level = 'info'

mne_log_level = 'error'

on_error = 'continue'
