#%%
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import config as cfg

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import RidgeCV, GammaRegressor, BayesianRidge, TweedieRegressor, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupShuffleSplit, KFold, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from itertools import product

from library.spfiltering import (
    ProjIdentitySpace, ProjCommonSpace, ProjSPoCSpace)
from library.featuring import Riemann, LogDiag, NaiveVec
import library.preprocessing_david as pp
from subject_number import subject_number

#from preprocessing import get_rejection_threshold as get_rejection_threshold
from autoreject import get_rejection_threshold

############################################################################
n_compo = 32
scale = 'auto'
metric = 'riemann'
seed = 42
n_splits = 2
n_jobs = 20
############################################################################

def run_low_rank(n_components, X, y, cv, estimators, scoring, groups):
    out = dict(n_components=n_components)
    for name, est in estimators.items(): #e.g. name --> riemann // est --> make_pipeline(...)
        print(name)
        this_est = est # est --> make_pipeline(...)
        this_est.steps[0][1].n_compo = n_components #151 -> n_components inside Riemann inside pipeline
        scores = cross_val_score(
            X=X, y=y, cv=cv, estimator=this_est, n_jobs=n_jobs,
            groups=groups,
            scoring=scoring)
        if scoring == 'neg_mean_absolute_error':
            scores = -scores
        print(np.mean(scores), f"+/-{np.std(scores)}")
        out[name] = scores
    return out


def _get_global_reject_ssp(raw, decim=5):
    # generate epochs around EOG artifact events
    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    if len(eog_epochs) >= 5: #  (?) Why?
        # Reject eog artifacts epochs with autoreject
        reject_eog = get_rejection_threshold(eog_epochs, decim=decim)
        del reject_eog['eog']
    else:
        reject_eog = None

    return reject_eog


def _compute_add_ssp_exg(raw):
    reject_eog = _get_global_reject_ssp(raw)
    if 'eog' in raw:
        proj_eog, _ = mne.preprocessing.compute_proj_eog(
            raw, average=True, reject=reject_eog, n_mag=0, n_grad=0, n_eeg=1)
    else:
        proj_eog = None
    if proj_eog is not None:
        raw.add_proj(proj_eog)


def global_run (number_subject      =   subject_number,
                annotations_resp    =   True,
                annotations_no_stim =   True,                
                crop                =   True,
                eog_rejection       =   True,
                eeg_rejection       =   True,
                high_pass_filter    =   0.01,
                baseline            =   -0.2,
                shift_EDA           =   1.5,
                tune_components     =   False,
                target              =   'delta',
                scores_prediction   =   False,
                estimator           =   TweedieRegressor,
                power               =   2,
                alpha               =   1,
                link                =   'log'
                ):

    """

    Reads the all subject files in 'data', preprocess signals, predicts EDA from EEG,
    and outputs a .csv file per subject in 'outputs'

    :number_subject:        String or list. '01' to '32' (DEAP database)
                            Default = 'subject_number' --> all subjects
    :annotations_resp:      Boolean. Respirations annotations (bad_resp)
    :annotations_no_stim:   Boolean. No stimuli annotations (bad_no_stim)
    :crop:                  Boolean. Work with crop data (between 0 and 500 secs)                  
    :eog_rejection:         Boolean. EOG Correction with SSP.
                            Default = True
    :eeg_rejection:         Boolean. EEG autoreject.
                            Default = True
    :high_pass_filter:      Float. High-pass filter (in Hz). No filter = None
                            Default = 0.01        
    :baseline:              Float.    Onset baseline when epoching (in seconds)
                            Default = 0.2       
    :shift_EDA:             Float.    Length shift between EEG and EDA epochs (in seconds)
                            Default = 1.5
    :tune_components:       Boolean. Tune n_components (rank of cov matrices)
                            Default = True                    
    :target:                String.   'Y' used in our model 
                            Default = 'delta' --> difference between min-max of the epoch
    :scores:                Return performance of EDA prediction (R2)
    :estimator:             String.   Model used to predict EDA
                            Default = TweedieRegressor()
    :power:                 Int. The power determines the underlying target distribution
                            Default = 2 --> Gamma
    :alpha:                 Int. Constant that multiplies the penalty term and thus determines
                            the regularization strength.
                            Default = 1
    :link:                  String. Link function of the GLM
                            Default = 'log'    
                            
    
    :return:             Dictionary with 'key' = parameters and arguments I tested
                         (e.g. subject 1, shif_EDA = 1.5), and ''value": eeg raw data, EDA data,
                         epochs, true EDA (e.g. true EDA var) and predicted EDA (e.g. predicted EDA var)
    
    """ 
    # container
    exp = []   

    # Making the function work with only one subject as input
    if type(number_subject) == str:
        number_subject = [number_subject]
 
    for i in number_subject:   
        
        if annotations_resp == True and annotations_no_stim == True:
            directory = 'outputs/data/EDA+EEG+bad_no_stim+bad_resp/'
            extension = '.fif'
            fname = op.join(directory, 's'+ i + extension)
            raw = mne.io.read_raw_fif(fname, preload=True)
            raw.set_channel_types({ 'Resp': 'misc'})

        elif annotations_resp == True and annotations_no_stim == False:
            directory = 'outputs/data/EDA+EEG+bad_resp/'
            extension = '.fif'
            fname = op.join(directory, 's'+ i + extension)
            raw = mne.io.read_raw_fif(fname, preload=True)
            raw.set_channel_types({ 'EDA': 'misc'})
        
        elif annotations_resp == False and annotations_no_stim == True:
            directory = 'outputs/data/EDA+EEG+bad_no_stim/'
            extension = '.fif'
            fname = op.join(directory, 's'+ i + extension)
            raw = mne.io.read_raw_fif(fname, preload=True)
            raw.set_channel_types({ 'EDA': 'misc'})
        
        else:
            directory = 'data/'
            extension = '.bdf'
            fname = op.join(directory, 's'+ i + extension)
            raw = mne.io.read_raw_bdf(fname, preload=True)
            #(?) Is it necessary to preload or better to load_data? Why?
            
            # Rename EDA channels
            raw.rename_channels(mapping={'GSR1':'EDA'})
            
            raw.set_channel_types({ 'EXG1': 'eog',
                                    'EXG2': 'eog',
                                    'EXG3': 'eog',
                                    'EXG4': 'eog',
                                    'EDA' : 'misc',
                                    'Erg1': 'misc',
                                    'Erg2': 'misc',
                                    'Resp': 'misc'})
            
            # Common channels
            common_chs = set(raw.info['ch_names'])
            # Discard non-relevant channels 
            common_chs -= {'EXG5', 'EXG6', 'EXG7', 'EXG8',
                           'GSR2', 'Erg1', 'Erg2', 'Plet', 'Temp'}
            
            common_chs = set(raw.info['ch_names'])
            raw.pick_channels(list(common_chs))
            
            picks_eda = mne.pick_channels(ch_names = raw.ch_names ,include=['EDA'])
            
            if int(i) < 23:
                raw.apply_function(fun=lambda x: x/1000, picks=picks_eda)
            else:
                raw.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)


        # crop for memory purposes (use to test things)
        if crop == True:
            raw = raw.crop(0., 500.)
            
        if eog_rejection == True:
            _compute_add_ssp_exg(raw)
            
        if eeg_rejection == True:
            events_reject = mne.make_fixed_length_events(raw, id=1, duration=5.,
                                                         overlap=0.)

            epochs_reject = mne.Epochs(raw, events_reject, tmin=0., tmax=5.,
                                   baseline=None)
            #eda_epochs = Epochs(raw=raw_eda, events=events, tmin=0., tmax=0., baseline=None)

            # Autoreject 
            reject = get_rejection_threshold(epochs_reject, ch_types=['eeg'],decim=4)
        
        
        raw.filter(high_pass_filter, 5, fir_design='firwin', picks='EDA') 

        # EEG Band-pass filter
        raw.filter(None, 120., fir_design='firwin')
        
        
        events = mne.make_fixed_length_events(  raw,
                                                id=3000,
                                                duration=pp.duration,
                                                overlap=2.0)
        if eeg_rejection == False:
            reject = None    
            
        ec = mne.Epochs(raw, events,
                        event_id=3000, tmin=baseline, tmax=pp.duration, proj=True,
                        baseline=(baseline, 0), reject=reject, preload=True,
                        decim=1)
        
        return ec
#%%
        X = []

        for fb in pp.fbands:
            
            ec_fbands = ec.copy().load_data().pick_types(eeg=True).filter(fb[0], fb[1])

            X.append([mne.compute_covariance(
                                            ec_fbands[ii], method='oas')['data'][None]
                                            for ii in range(len(ec_fbands))])   

        X = np.array(X)   
        # Delete one axis     
        X = np.squeeze(X)
        # order axis of ndarray (first n_sub, then n_fb)
        X = X.transpose(1,0,2,3)
        n_sub, n_fb, n_ch, _ = X.shape
        
        
        eda_epochs = ec.copy().pick_channels(['EDA']).shift_time(tshift= shift_EDA, relative=True)
        
        if target == 'mean':
            y = eda_epochs.get_data().mean(axis=2)[:, 0]  
        elif target == 'delta':
            y = eda_epochs.get_data().max(axis=2)[:, 0] - eda_epochs.get_data().min(axis=2)[:, 0]
        else:
            y = eda_epochs.get_data().var(axis=2)[:, 0]     
 
        n_components = np.arange(1, 32, 1) # max components --> 32 --> 32 EEG channels
        # now let's do group shuffle split
        splits = np.array_split(np.arange(len(y)), n_splits)
        groups = np.zeros(len(y), dtype=np.int)
        for val, inds in enumerate(splits):
            groups[inds] = val

        ##################################################################
        ridge_shrinkage = np.logspace(-3, 5, 100)
        spoc_shrinkage = np.linspace(0, 1, 5)
        common_shrinkage = np.logspace(-7, -3, 5)
        ##################################################################

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
                    RidgeCV(alphas=ridge_shrinkage)),
            'riemann_gamma': #GammaRegressor
                make_pipeline(
                    ProjCommonSpace(scale=scale, n_compo=n_compo, reg=1.e-05),
                    Riemann(n_fb=n_fb, metric=metric),
                    StandardScaler(),
                    GammaRegressor())
            }    
        
        if tune_components == True:
            
            low_rank_estimators = {k: v for k, v in pipelines.items()
                                if k in ('riemann')} #'spoc', 

            out_list = Parallel(n_jobs=n_jobs)(delayed(run_low_rank)(
                                n_components=cc, X=X, y=y,
                                groups=groups,
                                cv=GroupShuffleSplit(
                                    n_splits=2, train_size=.5, test_size=.5),
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

            riemann_model = make_pipeline(
            ProjCommonSpace(scale=scale, n_compo=best_components['riemann'],
                            reg=1.e-05),
            Riemann(n_fb=n_fb, metric=metric),
            StandardScaler(),
            estimator(power=power, alpha =alpha, link=link))
        
        else:
            riemann_model = make_pipeline(
            ProjCommonSpace(scale=scale, n_compo= 32,
                            reg=1.e-05),
            Riemann(n_fb=n_fb, metric=metric),
            StandardScaler(),
            estimator())
        
        cv = KFold(n_splits=2, shuffle=False)
        
        y_preds = cross_val_predict(riemann_model, X, y, cv=cv)
           
        r2_riemann_model = cross_val_score(riemann_model, X, y, cv=cv,  groups=groups)
        print("mean of R2 cross validation Riemannian Model : ", np.mean(r2_riemann_model))
        
        if scores_prediction == True:
            for scoring in ("r2", "neg_mean_absolute_error"):
                all_scores = dict()
                for key, estimator in pipelines.items():
                    cv = GroupShuffleSplit(n_splits=2, train_size=.5, test_size=.5)
                    scores = cross_val_score(X=X, y=y, estimator=estimator,
                                            cv=cv, n_jobs=min(2, n_jobs),
                                            groups=groups,
                                            scoring=scoring)
                    if scoring == 'neg_mean_absolute_error':
                        scores = -scores
                    all_scores[key] = scores
                score_name = scoring if scoring == 'r2' else 'mae'
            #return all_scores

            exp.append([{'raw':raw}, {'eda':eda}, {'ec':ec}, {'y': y},
                      {'y_pred': y_preds}, {'all_scores': all_scores}])
        
        else:
            exp.append([{'raw':raw}, {'eda':eda}, {'ec':ec}, {'y': y},
                      {'y_pred': y_preds}])
        
    return exp

#%%      
all_subjects ={}
for i in ['01']:
    experiment_results = global_run(number_subject=i, crop = False,
                                    eog_rejection       =   True  ,
                                    eeg_rejection       =   True  ,
                                    annotations_resp    =   True  ,
                                    annotations_no_stim =   True  )
    all_subjects[i] = [experiment_results]
   
#   X = experiment_results[0]
#   y = experiment_results[1]
#    ec = experiment_results [0]
#    reject = experiment_results [1]

#%%    
alpha = list(np.logspace(-3, 5, 100))
# boilerplate function to print kwargs
def print_kwargs(**kwargs):
    print(kwargs)

subject_number = ['01', '02']    
# Set combinations of paramaters
dynamic_params = {
                     'number_subject'     : subject_number,
#                    'high_pass_filter'   : [None, .001,.01, .05, .1, .2, 0.5],
#                    'shift_EDA'          : [0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.],
#                    'target'             : ['mean', 'var', 'delta'],
#                    'estimator'          : ['TweedieRegressor'],
#                    'power'              : [2, 3], # 2: Gamma / 3: Inverse Gaussian
#                    'alpha'              : alpha
                }

# Select which parameter we are going to test in this run
keys_to_extract = ["number_subject"]#, "shift_EDA_EEG"]

param_subset = {key: dynamic_params[key] for key in keys_to_extract}

param_names = list(param_subset.keys())
# zip with parameter names in order to get original property
param_values = (zip(param_names, x) for x in product(*param_subset.values()))

total_param = []
results_group_by_parameters = {}

for paramset in param_values:
    # use the dict from iterator of tuples constructor
    kwargs = dict(paramset)
    print_kwargs(**kwargs)
    total_param.append(kwargs)
    experiment_results = global_run(**kwargs)
    key_kwargs = frozenset(kwargs.items())
    results_group_by_parameters[key_kwargs] = [experiment_results]


#%%





















    
#%%
#################     TEST 1: NAME CHANNELS      ###################
def openfiles_raw(subject = '01',
                  annotations_resp = True,
                  annotations_no_stim = True):
    
    if annotations_resp == True and annotations_no_stim == True:
        directory = 'outputs/data/EDA+EEG+bad_no_stim+bad_resp/'
        extension = '.fif'
        fname = op.join(directory, 's'+ subject + extension)
        raw = mne.io.read_raw_fif(fname, preload=True)
        
    elif annotations_resp == True and annotations_no_stim == False:
        directory = 'outputs/data/EDA+EEG+bad_resp/'
        extension = '.fif'
        fname = op.join(directory, 's'+ subject + extension)
        raw = mne.io.read_raw_fif(fname, preload=True)
    
    elif annotations_resp == False and annotations_no_stim == True:
        directory = 'outputs/data/EDA+EEG+bad_no_stim/'
        extension = '.fif'
        fname = op.join(directory, 's'+ subject + extension)
        raw = mne.io.read_raw_fif(fname, preload=True)
    
    else:
        directory = 'data/'
        extension = '.bdf'
        fname = op.join(directory, 's'+ subject + extension)
        raw = mne.io.read_raw_bdf(fname, preload=True)
    
    return raw
        
#%%
raw = openfiles_raw(subject = '01', 
                    annotations_resp = True, annotations_no_stim = True)
# EDA stim --> EDA 
# chs: 1 MISC, 1 EMG, 32 EEG, 4 EOG, 1 STIM
# crop for memory purposes (use to test things) 
raw.set_channel_types({ 'Resp': 'misc'})      
eog_epochs = mne.preprocessing.create_eog_epochs(raw)
if len(eog_epochs) >= 5: #  (?) Why?
    # Reject eog artifacts epochs with autoreject
    reject_eog = get_rejection_threshold(eog_epochs, ch_types=['eeg'], decim=5)
    del reject_eog['eog']
else:
    reject_eog = None
#%%
# %%
raw = openfiles_raw(annotations_resp = True, annotations_no_stim = False)
# EDA stim --> EDA
# chs: 1 MISC, 1 EMG, 32 EEG, 4 EOG, 1 STIM
#%%
raw = openfiles_raw(annotations_resp = False, annotations_no_stim = True)
# EDA stim --> EDA
# chs: 1 MISC, 1 EMG, 32 EEG, 4 EOG, 1 STIM
#%%
raw = openfiles_raw(subject = '01',
                    annotations_resp = False, annotations_no_stim = False)
# EDA stim --> GSR1
# chs: 47 EEG, 1 STIM
raw.rename_channels(mapping={'GSR1':'EDA'})
raw.set_channel_types({ 'EXG1': 'eog',
                        'EXG2': 'eog',
                        'EXG3': 'eog',
                        'EXG4': 'eog',
                        'EDA' : 'misc',
                        'Erg1': 'misc',
                        'Erg2': 'misc',
                        'Resp': 'misc'})

# %%
