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
    #  Cannot have number of splits n_splits=5 greater than the number of samples: n_samples=3.
    if len(eog_epochs) >= 5: 
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


def global_preprocessing(number_subject = subject_number,
                         annotations_resp = True,
                         annotations_no_stim = True,                
                         crop = True,
                         avg_reference = True,
                         project_eog = True,
                         apply_autoreject  = True,
                         high_pass_filter =  0.1,
                         baseline = -0.2,
                         shift_EDA = 1.5,
                         target = 'delta'):

    """ Reads all subjects' files and preprocess signals.
    
    Parameters
    ----------

    number_subject : str or list
        '01' to '32' (DEAP database)
        Default = 'subject_number' --> all subjects
    annotations_resp : bool
        include respiration annotations (bad_resp)
        Default = True
    annotations_no_stim : bool
        include 'non stimuli' annotations (bad_no_stim)
        Default = True
    crop : bool
        Work with crop data (between 0 and 500 secs)
        Default = True  
    avg_reference: bool
        EEG average reference (proj)
        Default = True                
    project_eog : bool 
        EOG Correction with SSP.
        Default = True
    apply_autoreject : bool 
        EEG autoreject.
        Default = True
    high_pass_filter : float
        High-pass filter (in Hz).
        No filter --> None
        Default = 0.01      
    baseline : float
        Onset baseline when epoching (in seconds)
        Default = -0.2       
    shift_EDA : float
        Shift's length between EEG and EDA epochs (in seconds)
        Default = 1.5
    target : str
        'Y' used in our model.
        Default = 'delta' --> difference between min-max of the epoch 
                            
    Returns
    -------               
    X : epochs x fbands x EEG cov matrices 
    y : EDA target       
    """
    # Making the function work with only one subject as input
    if type(number_subject) == str:
        number_subject = [number_subject]
 
    for subject in number_subject:   
        
        if annotations_resp == True and annotations_no_stim == True:
            directory = 'outputs/data/EDA+EEG+bad_no_stim+bad_resp/'
            extension = '.fif'
            fname = op.join(directory, 's'+ subject + extension)
            raw = mne.io.read_raw_fif(fname, preload=True)
            raw.set_channel_types({ 'Resp': 'misc'})


        elif annotations_resp == True and annotations_no_stim == False:
            directory = 'outputs/data/EDA+EEG+bad_resp/'
            extension = '.fif'
            fname = op.join(directory, 's'+ subject + extension)
            raw = mne.io.read_raw_fif(fname, preload=True)
            raw.set_channel_types({'EDA' : 'misc',
                                   'Resp': 'misc'})
        
        elif annotations_resp == False and annotations_no_stim == True:
            directory = 'outputs/data/EDA+EEG+bad_no_stim/'
            extension = '.fif'
            fname = op.join(directory, 's'+ subject + extension)
            raw = mne.io.read_raw_fif(fname, preload=True)
            raw.set_channel_types({'EDA' : 'misc',
                                   'Resp': 'misc'})
        
        else:
            directory = 'data/'
            extension = '.bdf'
            fname = op.join(directory, 's'+ subject + extension)
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
                                    'Resp': 'misc',
                                    'Plet': 'misc'})
            
            # Common channels
            common_chs = set(raw.info['ch_names'])
            # Discard non-relevant channels 
            common_chs -= {'EXG5', 'EXG6', 'EXG7', 'EXG8',
                           'GSR2', 'Erg1', 'Erg2', 'Plet', 'Temp'}
            
            raw.pick_channels(list(common_chs))
            
            picks_eda = mne.pick_channels(ch_names = raw.ch_names ,include=['EDA'])
            
            if int(subject) < 23:
                raw.apply_function(fun=lambda x: x/1000, picks=picks_eda)
            else:
                raw.apply_function(fun=lambda x: (10**9/x)/1000, picks=picks_eda)

        # Fix problems with ch_names and ch_types
        if int(subject) > 28 and annotations_no_stim == False:
            raw.rename_channels(mapping={'-1': 'Status'} )
            raw.drop_channels('-0')            

        elif int(subject) > 23 and annotations_no_stim == False:
            raw.rename_channels(mapping={'': 'Status'} )
            
        raw.set_channel_types({ 'Status': 'stim'})

        # Crop for memory purposes (use to test things)
        if crop == True:
            raw = raw.crop(0., 500.)
        
        raw.filter(high_pass_filter, 5, fir_design='firwin', picks='EDA') 

        # EEG Band-pass filter
        raw.filter(0.1, 40., fir_design='firwin')
               
        # Set the EEG reference
        if avg_reference == True:
            raw, _ = mne.set_eeg_reference(raw, 'average', projection=True, ch_type='eeg')
          
        if project_eog == True:
            _compute_add_ssp_exg(raw)
            
        if apply_autoreject == True:
            events_reject = mne.make_fixed_length_events(raw, id=1, duration=5.,
                                                         overlap=0.)

            epochs_reject = mne.Epochs(raw, events_reject, tmin=0., tmax=5.,
                                   baseline=None)
            #eda_epochs = Epochs(raw=raw_eda, events=events, tmin=0., tmax=0., baseline=None)

            # Autoreject 
            reject = get_rejection_threshold(epochs_reject, ch_types=['eeg'], decim=4)
        
        events = mne.make_fixed_length_events(  raw,
                                                id=3000,
                                                duration=pp.duration,
                                                overlap=2.0)
        if apply_autoreject == False:
            reject = None    
            
        ec = mne.Epochs(raw, events,
                        event_id=3000, tmin=baseline, tmax=pp.duration, proj=True,
                        baseline=(baseline, 0), reject=reject, preload=True,
                        decim=1)
        
        return raw, ec
#%%        
#        X = []

#        for fb in pp.fbands:
            
#            ec_fbands = ec.copy().load_data().pick_types(eeg=True).filter(fb[0], fb[1])

#            X.append([mne.compute_covariance(
#                                            ec_fbands[ii], method='oas')['data'][None]
#                                            for ii in range(len(ec_fbands))])   

#        X = np.array(X)   
        # Delete one axis     
#        X = np.squeeze(X)
        # order axis of ndarray (first n_sub, then n_fb)
#        X = X.transpose(1,0,2,3)
#        n_sub, n_fb, n_ch, _ = X.shape
        
#        eda_epochs = ec.copy().pick_channels(['EDA']).shift_time(tshift= shift_EDA, relative=True)
        
#        if target == 'mean':
#            y = eda_epochs.get_data().mean(axis=2)[:, 0]  
#        elif target == 'delta':
#            y = eda_epochs.get_data().max(axis=2)[:, 0] - eda_epochs.get_data().min(axis=2)[:, 0]
#        else:
#            y = eda_epochs.get_data().var(axis=2)[:, 0]     
            
#        return raw, X, y


#%%
#### CREATE EOG PROJ ####
eog_epochs = mne.preprocessing.create_eog_epochs(raw)

reject_eog = get_rejection_threshold(eog_epochs, decim=5)
del reject_eog['eog']
    
# Compute SSP (signal-space projection) vectors for EOG artifacts.
proj_eog, _ = mne.preprocessing.compute_proj_eog(
    raw, average=True, reject=reject_eog, n_mag=0, n_grad=0, n_eeg=1, verbose = True)

if proj_eog is not None:
    eog_epochs.add_proj(proj_eog)

eog_epochs.average().plot(proj=True)


    
#%%
#### VISUALIZING EFECT OF EOG PROJ ####
report = mne.Report(verbose=True)

# Help to insert report name later
annotations_resp = True
annotations_no_stim = True
for subject in subject_number:
    globals()[f'plots_results_{subject}'] = []
    # Run function to preprocess signal depeding project_eog
    for project_eog in (False, True):
        raw, _ = global_preprocessing(number_subject = subject,
                                      annotations_resp = annotations_resp,
                                      annotations_no_stim = annotations_no_stim,
                                      avg_reference = True,
                                      crop = False,
                                      project_eog = False,
                                      apply_autoreject = False)
        
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)
        
        if len(eog_epochs) >= 5: 
                        
            if project_eog == False:
                plot_eog_avg = eog_epochs.average().plot(proj=True, show=False)
                plot_lims = plot_eog_avg.axes[0].get_ylim()
                globals()[f'plots_results_{subject}'].append(plot_eog_avg)
                
            else:
                reject_eog = get_rejection_threshold(eog_epochs, decim=5)
                del reject_eog['eog']
                    
                # Compute SSP (signal-space projection) vectors for EOG artifacts.
                proj_eog, _ = mne.preprocessing.compute_proj_eog(
                    raw, average=True, reject=reject_eog, n_mag=0, n_grad=0, n_eeg=1, verbose = True)

                if proj_eog is not None:
                    eog_epochs.add_proj(proj_eog)

                globals()[f'plots_results_{subject}'].append(eog_epochs.average().plot(proj=True, show=False,
                                                                                    ylim = dict(eeg = list(plot_lims))))
    
    if len(eog_epochs) >= 5:
        # add the custom plots to the report:
        report.add_figs_to_section(figs = globals()[f'plots_results_{subject}'],
                                captions = [f'SSP = False / EEG (32 channels)',
                                            f'SSP = True / EEG (32 channels)'],
                                section=f'S{subject}')
    
report.save(f'KEEP_report_all_subjects.html', overwrite=True)
#report.save(f'report_{subject}_annotations_{annotations_resp}_{annotations_no_stim}.html', overwrite=True)    
        
#%%
report.save(f'KEEP_report_{subject}_annotations_{annotations_resp}_{annotations_no_stim}.html', overwrite=True)      
        

#%%
bads = []
for subject in subject_number:
    raw, _ = global_preprocessing(number_subject = subject,
                                annotations_resp = False,
                                annotations_no_stim = False,
                                avg_reference = True,
                                crop = False,
                                project_eog = True,
                                apply_autoreject = False)
    bads.append(raw.info['bads'])

#%%
def _simplify_info(info):
    """Return a simplified info structure to speed up picking."""
    chs = [{key: ch[key]
            for key in ('ch_name', 'kind', 'unit', 'coil_type', 'loc', 'cal')}
           for ch in info['chs']]
    sub_info = mne.Info(chs=chs)
    return sub_info


#%% 
def global_modeling(number_subject = subject_number,
                    tune_components = False,
                    scores_prediction = False,
                    estimator = TweedieRegressor,
                    power = 2,
                    alpha = 1,
                    link = 'log'):
    
    """ Predicts EDA from EEG and outputs a dictionary with predictions.
    
    Parameters
    ----------
    tune_components : bool
        Tune n_components (rank of cov matrices)
        Default = False                    
    scores : bool
        Return performance of EDA prediction (R2)
        Default = False
    estimator : str
        Model used to predict EDA
        Default = TweedieRegressor()
    power : int
        Determines the underlying target distribution
        Default = 2 --> Gamma
    alpha : int
        Constant that multiplies the penalty term and thus determines
        the regularization strength.
        Default = 1
    link : str
        Link function of the GLM
        Default = 'log'    
                            
    Returns
    ------- 
    exp: dictionary with -->
        'key': parameters and arguments I tested
            (e.g. subject 1, shif_EDA = 1.5)
        'value':
            - EEG raw data
            - EDA data
            - Epochs
            - True EDA (e.g. true EDA var)
            - Predicted EDA (e.g. predicted EDA var)
    
    """ 
    # container
    exp = []   
    for subject in number_subject:
        X, y = global_preprocessing(number_subject = subject_number)
        
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










#############################################################
#################         TESTS         #####################
#############################################################
#%%
#### VISUALIZING EFECT OF EEG REFERENCE PROJ ####
# https://mne.tools/dev/auto_tutorials/preprocessing/45_projectors_background.html

report = mne.Report(verbose=True)

for subject in ['01', '02', '03']: #subject_number:
    for proj, avg_reference in ([False, False], [True, True]):
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
    
#%%
#### TEST EOG EPOCHS REMOVAL ####
# https://mne.tools/dev/auto_tutorials/preprocessing/10_preprocessing_overview.html
raw, _ = global_preprocessing(number_subject = '01',
                              annotations_resp = True,
                              annotations_no_stim = True,
                              crop = False,
                              project_eog = False,
                              apply_autoreject = False)

eog_epochs = mne.preprocessing.create_eog_epochs(raw)
eog_epochs.plot_image(combine='mean', picks='eeg')
eog_epochs.average().plot()

