#%%
import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold)
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from meegpowreg import make_filter_bank_regressor, make_filter_bank_transformer
from subject_number import subject_number as subjects
from numpy.lib.npyio import save


DEBUG = True

if DEBUG:
    N_JOBS = 1
    subjects = ['01']

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

for subject in subjects:
    PCA_123 = np.load(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                  'PCA.npy'))
    df_PCA = pd.DataFrame(PCA_123, columns = ['PCA1','PCA2','PCA3'])

    



# %%

array = np.load(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                  'PCA.npy'))
#%%



#%%
#%%
report = mne.Report(verbose=True)

for subject in ['01'] #, '02', '03']: #subject_number:
        # Create container for two types of plots
        globals()[f'plots_results_{proj}'] = []
        # Run function to preprocess signal depeding avg_reference
        raw, _ = global_preprocessing(number_subject = subject,
                                      annotations_resp = True,
                                      annotations_no_stim = True,
                                      avg_reference = avg_reference,
                                      crop = False,
                                      project_eog = False,
                                      apply_autoreject = False)

        # Crop data
        eeg_raw = raw.copy().crop(tmin=1000, tmax=1010).pick_types(eeg=True)
        # Create times to insert in sliders
        times = eeg_raw.times[::512]
        # for loop to create plot with sliders to insert in report
        for t in times:
            # create plot with time t
            globals()[f'fig_proj_{proj}'] =  eeg_raw.plot(start = t, duration = 1, butterfly=True,
                                                        proj=proj, show=False)
            globals()[f'fig_proj_{proj}'].subplots_adjust(top=0.9)
            globals()[f'fig_proj_{proj}'].suptitle(f'proj={proj}', size='xx-large', weight='bold')
            # append this plot with time t in 'plots_results_{proj}'
            globals()[f'plots_results_{proj}'].append(globals()[f'fig_proj_{proj}']) 

        report.add_slider_to_section(globals()[f'plots_results_{proj}'],
                                      times, f'S{subject}',
                                      title = f'EEG reference Proj = {proj}',
                                      image_format='png')  

report.save('report_EEG_reference_Proj.html', overwrite=True)