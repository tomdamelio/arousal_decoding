#%%
import glob
import re
import os.path as op
from os import listdir
from os.path import isfile, join

import numpy as np
from joblib import Parallel, delayed
import mne
from autoreject import get_rejection_threshold

import config as cfg
import library.preprocessing_david as pp

#%%
def rawfile_of(subject):
    'return list of rawfile' #(?) return path of rawfile of a particular subject
    rawfiles = [f for f in bdfs if subject in f]
    return rawfiles[0]  # few subjects have multiple sessions

#def age_of(subject, print_header=False):
    # TNTLFreiburg/brainfeatures/blob/master/brainfeatures/utils/file_util.py
    # teuniz.net/edfbrowser/edf%20format%20description.html
#    fp = rawfile_of(subject)
#    assert op.exists(fp), "file not found {}".format(fp) # confirms that the path exists.
#    f = open(fp, 'rb') # Use the 'rb' mode in the open() function to read a binary files
#    content = f.read(88)
#    f.close()
#    patient_id = content[8:88].decode('ascii')
#    print(patient_id if print_header else None)
#    [age] = re.findall("Age:(\\d+)", patient_id)
#    return int(age)


def preprocess_raw(subject):
    raw_file = rawfile_of(subject)
    raw = mne.io.read_raw_bdf(raw_file)
    #raw.crop(tmin=60, tmax=540)  # 8mn of signal to be comparable with CAM-can
    raw.load_data().pick_channels(list(common_chs))
    #raw.resample(250)  # max common sfreq --> not necessary

    # autoreject global (instead of clip at +-800uV proposed by Freiburg)
    duration = 5.
    events = mne.make_fixed_length_events(raw, id=1, duration=duration, overlap=0.)
    epochs = mne.Epochs(raw, events, tmin=0., tmax=duration, baseline=None)
    reject = get_rejection_threshold(epochs, decim=1)
    return raw, reject


def _compute_cov(subject): # estimation of between-sensor covariance (spatial covariance)
    rawc, reject = preprocess_raw(subject)

    events = mne.make_fixed_length_events(
        rawc, id=3000, duration=pp.duration, overlap=2.0)
    epochs = mne.Epochs(
        rawc, events, event_id=3000, tmin=0, tmax=pp.duration, proj=True,
        baseline=None, reject=reject, preload=False, decim=1)
    epochs.drop_bad()
    clean_events = events[epochs.selection]

    #  picks = mne.pick_types(rawc.info, meg=False, eeg=True)
    covs = []
    for fb in pp.fbands: #fb = frequency bands
        rf = rawc.copy().load_data().filter(fb[0], fb[1]) #rf = raw filtered
        ec = mne.Epochs(
            rf, clean_events, event_id=3000, tmin=0, tmax=pp.duration,
            proj=True, baseline=None, reject=None, preload=False, decim=1,
            picks=None)
        cov = mne.compute_covariance(ec, method='oas', rank=None)
        covs.append(cov.data)
    out = dict(subject=subject, n_events=len(events),
               n_events_good=len(clean_events),
               covs=np.array(covs))#, age=age_of(subject))
    return out


def _run_all(subject):
    mne.utils.set_log_level('warning')
    print(subject)
    error = 'None'
    result = dict()
    try:
        result = _compute_cov(subject)
    except Exception as err:
        error = repr(err)
        print(error)
    out = dict(error=error)
    out.update(result)
    return out
#%%
# edf files are stored in root_dir/
# edf/{eval|train}/normal/01_tcp_ar/103/00010307/s001_2013_05_29/00010307_s001_t000.edf'
# '01_tcp_ar': the only type of channel configuration used in this corpus
# '103': header of patient id to make folders size reasonnable
# '00010307': patient id
# 's001_2013_01_09': session & archive date (~record date from EEG header)
# '00010194_s001_t001.edf': patient id, session number and token number of EEG
# segment

#root_dir = 'data'
#bdfs = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) & f.endswith(".bdf")]
bdfs = glob.glob("data/*.bdf")           
                    
subjects = sorted(list(set([bdf.split('\\')[-1] for bdf in bdfs])))

#%%
raw = mne.io.read_raw_bdf(rawfile_of(subjects[0]))
common_chs = set(raw.info['ch_names'])

#%%
for sub in subjects[1:]:
    raw = mne.io.read_raw_bdf(rawfile_of(sub))
    chs = set(raw.info['ch_names'])
    common_chs = common_chs.intersection(chs)
    
#%%
common_chs -= {'EXG5', 'EXG6', 'EXG7', 'EXG8'
               'GSR2', 'Erg1', 'Erg2', 'Resp'
               'Plet', 'Temp'}

#%%
out = Parallel(n_jobs=1)(
    delayed(_run_all)(subject=subject)
    for subject in subjects)

#%%
fname_covs = op.join(cfg.derivative_path, 'covs_tuh_oas.h5')
mne.externals.h5io.write_hdf5(fname_covs, out, overwrite=True)

#  age = np.array([age_of(subject) for subject in subjects])
#  import matplotlib.pyplot as plt
#  plt.close('all')
#  plt.hist(age, bins=20)
#  plt.title('Age histogram of TUH Abnormal dataset')
#  plt.xlabel('Age')
#  plt.savefig(op.join(cfg.path_outputs, 'fig_tuh_hist_age.png'), dpi=300)