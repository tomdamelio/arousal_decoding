#%%
import os.path as op

import numpy as np
import matplotlib.pyplot as plt

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

eda.filter(None, 5, fir_design='firwin')

# Common channels
common_chs = set(raw.info['ch_names'])
common_chs -= {'EXG1', 'EXG2', 'EXG3', 'EXG4',
               'EXG5', 'EXG6', 'EXG7', 'EXG8',
               'GSR2', 'Erg1', 'Erg2', 'Resp',
               'Plet', 'Temp', 'GSR1', 'Status'}

# Filter EEG data to focus on beta band, no ref channels (!)
raw.pick_channels(list(common_chs))
raw.filter(None, 120., fir_design='firwin')

#%%
# Prepare data
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
            
n_sub, n_fb, n_ch, _ = X.shape

#%%
ridge_shrinkage = np.logspace(-3, 5, 100)
spoc_shrinkage = np.linspace(0, 1, 5)
common_shrinkage = np.logspace(-7, -3, 5)

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
                       if k in ('riemann')} #'spoc', 

out_list = Parallel(n_jobs=n_jobs)(delayed(run_low_rank)(
                    n_components=cc, X=X, y=y,
                    groups=groups,
                    cv=GroupShuffleSplit(
                        n_splits=10, train_size=.8, test_size=.2),
                    estimators=low_rank_estimators, scoring='r2')
                    for cc in n_components)
out_frames = list()
for this_dict in out_list:
    this_df = pd.DataFrame({#'spoc': this_dict['spoc'],
                           'riemann': this_dict['riemann']})
    this_df['n_components'] = this_dict['n_components']
    this_df['fold_idx'] = np.arange(len(this_df))
    out_frames.append(this_df)
out_df = pd.concat(out_frames)

out_df.to_csv("./DEAP_component_scores.csv")

mean_df = out_df.groupby('n_components').mean().reset_index()

best_components = {
    #'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
    'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
}

#%%
riemann_model = make_pipeline(
    ProjCommonSpace(scale=scale, n_compo=best_components['riemann'],
                    reg=1.e-05),
    Riemann(n_fb=n_fb, metric=metric),
    StandardScaler(),
    RidgeCV(alphas=ridge_shrinkage))

cv = KFold(n_splits=2, shuffle=False)


#%%
# Optimize duration epoch --> SEGUIR DESDE ACA. ADAPTAR TODO ESTO PARA OPTIMIZAR EL START DE LAS EPOCAS DE Y    
    
def run_low_rank(n_components, start_shift, X, y, cv, estimators, scoring, groups):
    out = dict(start_shift=start_shift)
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
                       if k in ('riemann')} #'spoc', 

out_list = Parallel(n_jobs=n_jobs)(delayed(run_low_rank)(
                    n_components=cc, X=X, y=y,
                    groups=groups,
                    cv=GroupShuffleSplit(
                        n_splits=10, train_size=.8, test_size=.2),
                    estimators=low_rank_estimators, scoring='r2')
                    for cc in n_components)
out_frames = list()
for this_dict in out_list:
    this_df = pd.DataFrame({#'spoc': this_dict['spoc'],
                           'riemann': this_dict['riemann']})
    this_df['n_components'] = this_dict['n_components']
    this_df['fold_idx'] = np.arange(len(this_df))
    out_frames.append(this_df)
out_df = pd.concat(out_frames)

out_df.to_csv("./DEAP_component_scores.csv")

mean_df = out_df.groupby('n_components').mean().reset_index()

best_components = {
    #'spoc': mean_df['n_components'][mean_df['spoc'].argmax()],
    'riemann': mean_df['n_components'][mean_df['riemann'].argmax()]
}