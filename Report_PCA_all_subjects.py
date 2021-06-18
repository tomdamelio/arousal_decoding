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
from scipy.signal import detrend


report = mne.Report(verbose=True)

DEBUG = False

if DEBUG:
    N_JOBS = 1
    subjects = ['01', '02']

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

PCA_all_subjects = dict()
for subject in subjects:
    PCA_123 = np.load(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                  'PCA.npy'))
    df_PCA = pd.DataFrame(PCA_123, columns = ['PCA1','PCA2','PCA3'])
    PCA_all_subjects[subject] = PCA_123
    
# read dict with mean(EDA) for every subjects
fname_out = op.join(derivative_path, 'mean_eda_all_subs.npy')
mean_eda_all_subjects_npy = np.load(fname_out, allow_pickle=True)

# Define plotting function
def fig_plot(mean_eda_all_subjects_npy = mean_eda_all_subjects_npy,
             PCA_all_subjects = PCA_all_subjects,
             PCA_dimension = 1,
             subject = '01'):
    
    y = mean_eda_all_subjects_npy[()][subject]
    idx_ndarray_PCA = PCA_dimension-1
    PCA = PCA_all_subjects[subject][:,[idx_ndarray_PCA]]
    # Scale
    scaler = preprocessing.StandardScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    PCA_scaled = scaler.fit_transform(PCA)
    mid_point = int(len(PCA_scaled)/2)
    last_point = int(len(PCA_scaled))
    
    # Create a new figure
    fig_aux = plt.figure(figsize= [17.2, 4.8])
    plt.plot(y_scaled, label = 'EDA')
    plt.plot(PCA_scaled, label='EEG')
    y_lim_value = plt.gca().get_ylim()
    
    
    fig = plt.figure(figsize= [17.2, 4.8])
    plt.plot(y_scaled[0:int(mid_point/2)], label = 'EDA')
    plt.plot(PCA_scaled[0:int(mid_point/2)], label='EEG')
    plt.ylim(y_lim_value)
    plt.legend()
    
    fig2 = plt.figure(figsize= [17.2, 4.8])
    plt.plot([i for i in range(int(mid_point/2),mid_point)], y_scaled[int(mid_point/2):mid_point], label = 'EDA')
    plt.plot([i for i in range(int(mid_point/2),mid_point)], PCA_scaled[int(mid_point/2):mid_point], label='EEG')
    plt.ylim(y_lim_value)
    plt.legend()
    
    fig3 = plt.figure(figsize= [17.2, 4.8])
    plt.plot([i for i in range(mid_point,int(mid_point*1.5))], y_scaled[mid_point:int(mid_point*1.5)],label ='EDA')
    plt.plot([i for i in range(mid_point,int(mid_point*1.5))], PCA_scaled[mid_point:int(mid_point*1.5)],label='EEG')
    plt.ylim(y_lim_value)
    plt.legend()
    
    fig4 = plt.figure(figsize= [17.2, 4.8])
    plt.plot([i for i in range(int(mid_point*1.5),last_point)], y_scaled[int(mid_point*1.5):last_point], label = 'EDA')
    plt.plot([i for i in range(int(mid_point*1.5),last_point)], PCA_scaled[int(mid_point*1.5):last_point], label='EEG')
    plt.ylim(y_lim_value)
    plt.legend()
    
    return fig, fig2, fig3, fig4

for dimension in [1,2,3]:
    for subject in subjects:
        plot1, plot2, plot3, plot4 = fig_plot(subject=subject, PCA_dimension = dimension)
        # add the custom plots to the report:
        report.add_figs_to_section([plot1, plot2, plot3, plot4],
                                captions=['PCA {} EEG and EDA comparison - Subject {}'.format(dimension, subject),
                                          'PCA {} EEG and EDA comparison - Subject {}'.format(dimension, subject),
                                          'PCA {} EEG and EDA comparison - Subject {}'.format(dimension, subject),
                                          'PCA {} EEG and EDA comparison - Subject {}'.format(dimension, subject)],
                                section= '{} PCA dimension'.format(dimension))

report.save('Report_PCA_all_subjects.html', overwrite=True)

#%%
