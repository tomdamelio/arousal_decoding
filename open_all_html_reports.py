#%%
import os
import webbrowser
from subject_number import subject_number


def open_files_in_browser(directory = 'outputs/DEAP-bids/derivatives/mne-bids-pipeline'):
    extension = '.html'
    process = 'report'
    for subject in subject_number:
        webbrowser.open_new_tab(os.path.join(directory, 'sub-'+ subject,
                                             'eeg', 'sub-'+ subject + '_task-rest_' + process + extension))

open_files_in_browser()
# %%
