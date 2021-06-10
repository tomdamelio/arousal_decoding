#%%
import os.path as op
import glob
import mne
from joblib import Parallel, delayed
from mne_bids import BIDSPath

from subject_number import subject_number as subjects
from meegpowreg.power_features import _compute_covs_epochs

#%%

derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

def _compute_covs(subject, kind, freqs):
    epochs_path = BIDSPath(subject=subject,
                           task=kind,
                           datatype='eeg',
                           root=derivative_path,
                           processing='clean',
                           extension='.fif',
                           suffix='epo',
                           check=False)
    epochs = mne.read_epochs(epochs_path)
    epochs.pick_types(eeg=True)
    covs = list()
    for i in range(len(epochs)):
        for _, fb in freqs.items():
#           _compute_covs_epochs(epoch, freqs)
            ec = epochs[i:i+1].copy().load_data().filter(fb[0], fb[1])
            cov = mne.compute_covariance(ec, method='oas', rank=None)
            covs.append(cov.data)
    #covs_event = [_compute_covs_epochs(epoch, frequency_bands=freqs) for epoch in epochs]
    return covs


def _run_all(subject, freqs, kind='rest'):
    mne.utils.set_log_level('warning')
    # mne.utils.set_log_level('info')
    print(subject)
    error = 'None'
    if not DEBUG:
        try:
            out = _compute_covs(subject, kind, freqs)
        except Exception as err:
            error = repr(err)
            print(error)
    else:
        out = _compute_covs(subject, kind, freqs)

    if error != 'None':
        out = {band: None for _, _, band in freqs}
        out['error'] = error
    return out


#%%
N_JOBS = 20
DEBUG = True
freqs = {"low": (0.1, 1.5),
         "delta": (1.5, 4.0),
         "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0),
         "beta_low": (15.0, 26.0),
         "beta_high": (26.0, 35.0),
         "gamma_low": (35.0, 49.0)}


if DEBUG:
    N_JOBS = 1
    subjects = subjects[:1]

out = Parallel(n_jobs=N_JOBS)(
    delayed(_run_all)(subject=subject, freqs=freqs, kind='rest')
    for subject in subjects)

for sub, dd in zip(subjects, out):
    if 'error' not in dd:
        mne.externals.h5io.write_hdf5(
            op.join(derivative_path, 'sub-' + ''.join(sub), 'eeg', 'sub-' + ''.join(sub) + '_covariances.h5'), dd,
            overwrite=True)

#%%
covs = mne.externals.h5io.read_hdf5(op.join(derivative_path, 'sub-' + '01', 'eeg',
                                            'sub-' + '01' + '_covariances.h5'))
# %%
