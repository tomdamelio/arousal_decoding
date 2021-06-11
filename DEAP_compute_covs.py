#%%
import os.path as op
import glob
import mne
from joblib import Parallel, delayed
from mne_bids import BIDSPath

from subject_number import subject_number as subjects
from meegpowreg.power_features import _compute_covs_epochs

derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'


#%%

def read_bids_epochs (subject):
    # set epochs paths in BIDS format
    epochs_path = BIDSPath(subject=subject,
                            task='rest',
                            datatype='eeg',
                            root=derivative_path,
                            processing='clean',
                            extension='.fif',
                            suffix='epo',
                            check=False)
    # read epochs
    epochs = mne.read_epochs(epochs_path)
    # pick only eeg data
    epochs.pick_types(eeg=True)
    return epochs


def _compute_covs(subject, freqs):
    
    '''
    compute covariance for a single epoch
    
    Parameters
    ----------
    subject: int or list. Subject from whom it will be extracted the epochs.
    freqs: frequency bands
    
    Return
    ------
    Computed covariances in one epoch
    
    '''
    # read all epochs
    epochs_all = read_bids_epochs(subject)
    if DEBUG:
        epochs_all = epochs_all[:3]
    covs = {}
    for n_epochs in range(len(epochs_all)):
        # subset one epoch based on index (n_epochs)
        epochs = epochs_all[n_epochs:n_epochs+1]
        # compute covariance matrix
        covs[str(n_epochs+1)] = _compute_covs_epochs(epochs, frequency_bands=freqs)
    return covs


def _run_all(subject, freqs):
    ''' Run '_compute_covs' to compute cov matrices for single epochs
        and handle errors''' 
    mne.utils.set_log_level('warning')
    # mne.utils.set_log_level('info')
    print(subject)
    error = 'None'
    if not DEBUG:
        try:
            out = _compute_covs(subject, freqs)
        except Exception as err:
            error = repr(err)
            print(error)
    else:
        out = _compute_covs(subject, freqs)

    if error != 'None':
        out = {band: None for _, _, band in freqs}
        out['error'] = error
    return out


N_JOBS = 20
DEBUG = False
freqs = {"low": (0.1, 1.5),
         "delta": (1.5, 4.0),
         "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0),
         "beta_low": (15.0, 26.0),
         "beta_high": (26.0, 35.0),
         "gamma_low": (35.0, 49.0)}

if DEBUG:
    N_JOBS = 1
    subjects = subjects[:3]

out = Parallel(n_jobs=N_JOBS)(
    delayed(_run_all)(subject = subject, freqs=freqs)
    for subject in subjects)

for sub, dd in zip(subjects, out):
    if 'error' not in dd:
        mne.externals.h5io.write_hdf5(
            op.join(derivative_path, 'sub-' + ''.join(sub), 'eeg', 'sub-' + ''.join(sub) + '_covariances.h5'), dd,
            overwrite=True)

covs = mne.externals.h5io.read_hdf5(op.join(derivative_path, 'sub-' + '01', 'eeg',
                                            'sub-' + '01' + '_covariances.h5'))
# %%
