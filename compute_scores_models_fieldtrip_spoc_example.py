import os.path as op

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupShuffleSplit, KFold, cross_val_predict

# from sklearn.model_selection import cross_val_score, KFold, GroupShuffleSplit
# from sklearn.model_selection import GridSearchCV
# from sklearn.base import clone
import mne
import pandas as pd

from mne import Epochs
from mne.datasets.fieldtrip_cmc import data_path

import config as cfg
from library.spfiltering import (
    ProjIdentitySpace, ProjCommonSpace, ProjSPoCSpace)
from library.featuring import Riemann, LogDiag, NaiveVec
import library.preprocessing_david as pp

from joblib import Parallel, delayed

##############################################################################
n_compo = 151
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 10
n_jobs = 20

##############################################################################

# Define parameters
fname = op.join('data', 's01.bdf')
# may need mkdir ~/mne_data
# import locale; locale.setlocale(locale.LC_ALL, "en_US.utf8")

raw = mne.io.read_raw_bdf(fname)
#raw.crop(450., 650.).load_data()  # crop for memory purposes
raw.load_data()  # crop for memory purposes
# (0-400s=lft=> crop 50-250, 400-800=rgt => crop 450-650)

# Filter muscular activity to only keep high frequencies
#emg = raw.copy().pick_channels(['EMGrgt'])
eda = raw.copy().pick_channels(['GSR1'])
eda.filter(5., None, fir_design='firwin')

# Common channels
common_chs = set(raw.info['ch_names'])
common_chs -= {'EXG1', 'EXG2', 'EXG3', 'EXG4',
               'EXG5', 'EXG6', 'EXG7', 'EXG8',
               'GSR2', 'Erg1', 'Erg2', 'Resp',
               'Plet', 'Temp', 'GSR1', 'Status'}

# Filter MEG data to focus on beta band, no ref channels (!)
raw.pick_channels(list(common_chs))
raw.filter(None, 120., fir_design='firwin')

# Build epochs as sliding windows over the continuous raw file
events = mne.make_fixed_length_events(raw, id=3000, duration=pp.duration, overlap=2.0)

#eeg_epochs = Epochs(raw, events,  event_id=3000, tmin=0, tmax=pp.duration, proj=True,
#        baseline=None, preload=True, decim=1)
eda_epochs = Epochs(eda, events,  event_id=3000, tmin=0, tmax=pp.duration, proj=True,
        baseline=None, preload=True, decim=1)

# Prepare data
#X = np.empty(shape=(483, len(pp.fbands), 32, 32), dtype=int)
X = []
for fb in pp.fbands:
    rf = raw.copy().load_data().filter(fb[0], fb[1])
    events = mne.make_fixed_length_events(rf,
                                          id=3000,
                                          duration=pp.duration,
                                          overlap=2.0)
    
    ec = mne.Epochs(rf, events,
            event_id=3000, tmin=0, tmax=pp.duration,
            proj=True, baseline=None, reject=None, preload=True, decim=1,
            picks=None)
    X.append([mne.compute_covariance(
                ec[ii], method='oas')['data'][None]
                for ii in range(len(ec))])   

X = np.array(X)   
# Delete one axis     
X = np.squeeze(X)
# order axis of ndarray (first n_sub, then n_fb)
X = X.transpose(1,0,2,3)
            
y = eda_epochs.get_data().var(axis=2)[:, 0]  # target is EDA power

n_sub, n_fb, n_ch, _ = X.shape

#%%
##############################################################################

#ridge_shrinkage = np.logspace(-3, 5, 100)
#spoc_shrinkage = np.linspace(0, 1, 5)
#common_shrinkage = np.logspace(-7, -3, 5)

#riemann_model = make_pipeline(    
#    ProjCommonSpace(scale=scale, n_compo=n_compo,
#                    reg=1.e-05),
#    Riemann(n_fb=n_fb, metric=metric), #n_fb=n_fb 
#    StandardScaler(),
#    RidgeCV(alphas=ridge_shrinkage))

#cv = KFold(n_splits=2, shuffle=False)

# Run cross validaton
#y_preds = cross_val_predict(riemann_model, X, y, cv=cv)

#%%
# Plot the True EDA power and the EDA predicted from EEG data
#fig, ax = plt.subplots(1, 1, figsize=[10, 4])
#times = raw.times[epochs.events[:, 0] - raw.first_samp]
#ax.plot(times, y_preds, color='b', label='Predicted EDA')
#ax.plot(times, y, color='r', label='True EDA')
#ax.set_xlabel('Time (s)')
#ax.set_ylabel('EDA average')
#ax.set_title('SPoC EEG Predictions')
#plt.legend()
#mne.viz.tight_layout()
#plt.show()

#%%
pipelines = {
    'dummy':  make_pipeline(
        ProjIdentitySpace(), LogDiag(), StandardScaler(), DummyRegressor()),
    'naive': make_pipeline(ProjIdentitySpace(), NaiveVec(method='upper'),
                           StandardScaler(),
                           RidgeCV(alphas=ridge_shrinkage)),
    'log-diag': make_pipeline(ProjIdentitySpace(), LogDiag(),
                              StandardScaler(),
                              RidgeCV(alphas=ridge_shrinkage)),
    'spoc': make_pipeline(
            ProjSPoCSpace(n_compo=n_compo,
                          scale=scale, reg=0, shrink=0.5),
            LogDiag(),
            StandardScaler(),
            RidgeCV(alphas=ridge_shrinkage)),
    'riemann':
        make_pipeline(
            ProjCommonSpace(scale=scale, n_compo=n_compo, reg=1.e-05),
            Riemann(n_fb=n_fb, metric=metric),
            StandardScaler(),
            RidgeCV(alphas=ridge_shrinkage))
}

#%%
n_components = np.arange(1, 152, 1)
# now let's do group shuffle split
splits = np.array_split(np.arange(len(y)), n_splits)
groups = np.zeros(len(y), dtype=np.int)
for val, inds in enumerate(splits):
    groups[inds] = val

#%%
def run_low_rank(n_components, X, y, cv, estimators, scoring, groups):
    out = dict(n_components=n_components)
    for name, est in estimators.items():
        print(name)
        this_est = est
        this_est.steps[0][1].n_compo = n_components
        scores = cross_val_score(
            X=X, y=y, cv=cv, estimator=this_est, n_jobs=1,
            groups=groups,
            scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        print(np.mean(scores), f"+/-{np.std(scores)}")
        out[name] = scores
    return out


low_rank_estimators = {k: v for k, v in pipelines.items()
                       if k in ('spoc', 'riemann')}

out_list = Parallel(n_jobs=n_jobs)(delayed(run_low_rank)(
                    n_components=cc, X=X, y=y,
                    groups=groups,
                    cv=GroupShuffleSplit(
                        n_splits=10, train_size=.8, test_size=.2),
                    estimators=low_rank_estimators, scoring='r2')
                    for cc in n_components)
out_frames = list()
for this_dict in out_list:
    this_df = pd.DataFrame({'spoc': this_dict['spoc'],
                           'riemann': this_dict['riemann']})
    this_df['n_components'] = this_dict['n_components']
    this_df['fold_idx'] = np.arange(len(this_df))
    out_frames.append(this_df)
out_df = pd.concat(out_frames)
#%%
out_df.to_csv("./DEAP_component_scores.csv")

mean_df = out_df.groupby('n_components').mean().reset_index()

#%%
best_components = {
    'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
    'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
}

#%%
pipelines[f"spoc_{best_components['spoc']}"] = make_pipeline(
    ProjSPoCSpace(n_compo=best_components['spoc'],
                  scale=scale, reg=0, shrink=0.5),
    LogDiag(),
    StandardScaler(),
    RidgeCV(alphas=ridge_shrinkage))

pipelines[f"riemann_{best_components['riemann']}"] = make_pipeline(
    ProjCommonSpace(scale=scale, n_compo=best_components['riemann'],
                    reg=1.e-05),
    Riemann(n_fb=n_fb, metric=metric),
    StandardScaler(),
    RidgeCV(alphas=ridge_shrinkage))

#%%

for scoring in ("r2", "neg_mean_absolute_error"):
    # now regular buisiness
    all_scores = dict()
    for key, estimator in pipelines.items():
        cv = GroupShuffleSplit(n_splits=n_splits, train_size=.8, test_size=.2)
        scores = cross_val_score(X=X, y=y, estimator=estimator,
                                 cv=cv, n_jobs=min(n_splits, n_jobs),
                                 groups=groups,
                                 scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        all_scores[key] = scores
    score_name = scoring if scoring == 'r2' else 'mae'
    np.save(op.join('data',
                    f'all_scores_models_fieldtrip_spoc_{score_name}.npy'),
            all_scores)


