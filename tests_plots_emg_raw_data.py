#%%
import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mne
from mne_bids import BIDSPath
from sklearn.linear_model import RidgeCV, GammaRegressor
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, GroupShuffleSplit, cross_val_predict)
from meegpowreg import make_filter_bank_regressor
from subject_number import subject_number as subjects
from joblib import Parallel, delayed

measure = 'emg'

if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg

DEBUG = True

date = '20-07'

derivative_path = cfg.deriv_root

n_components = np.arange(1, 32, 1)
seed = 42
n_splits = 10
n_jobs = 15
score_name, scoring = "r2", "r2"
cv_name = '10Fold'

freqs = {'low': (0.1, 1.5),
         'delta': (1.5, 4.),
         'theta': (4., 8.),
         'alpha': (8., 15.),
         'beta_l': (15., 26.),
         'beta_h': (26., 35.),
         'gamma': (35., 49.)}

n_jobs = 4
# subs 8, 10, 12, 28
subjects = ['10']
subject = '10'
debug_out = '_DEBUG'

fname_covs = op.join(derivative_path, f'{measure}-cov-matrices-all-freqs', 'sub-' + subject + f'_covariances_{measure}.h5')

fname_epochs = derivative_path / 'clean-epo-files'
epochs = mne.read_epochs(op.join(fname_epochs, 'sub-' + subject + '_task-rest_proc-clean_epo.fif'))


picks_emg = mne.pick_types(epochs.info, emg=True)
epochs = epochs.filter(20., 30., picks=picks_emg)
# How are we going to model our target? -> Mean of two EMG zygomaticus sensors
emg_epochs = epochs.copy().pick_channels(['EXG5','EXG6'])

y = emg_epochs.get_data().var(axis=2).mean(axis=1)

#%%
y_aux = emg_epochs.get_data().mean(axis=1)
_ , aux_x = y_aux.shape
y_aux_aux = y_aux.reshape(len(y)*aux_x)
plt.plot(y_aux_aux)

# %%
# Para quedarme con la señal raw, deberia subsamplear la señal para quedarme con una epoca cada 6.
# Esto es asi porque tengo epocas de 1500 ms, cada 250 ms. 1500/260 = 6.
# Deberia quedarme con las epocas 1, 7, 13, etc.
epoch_aux = emg_epochs[0]
epoch_aux = epoch_aux.get_data().mean(axis=1)
epoch_aux = np.squeeze(epoch_aux)

# Resample inside and epoch
epoch_aux = epoch_aux[::6]

# %%
# plot raw data
epochs = emg_epochs.get_data().mean(axis=1)
epochs = epochs[::6,:]
epochs_shape_1, epochs_shape_2 = epochs.shape
epochs = epochs.reshape(epochs_shape_1*epochs_shape_2)
plt.plot(epochs)
# %%
# Plot subsample
# Un valor cada medio segundo
%matplotlib
epochs_sub = epochs[::256]
plt.plot(epochs_sub)

#%%
import os.path as op
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import mne
from mne_bids import BIDSPath
from sklearn.linear_model import RidgeCV, GammaRegressor
from sklearn.model_selection import (
     cross_val_score, ShuffleSplit, KFold, GroupShuffleSplit, cross_val_predict)
from meegpowreg import make_filter_bank_regressor
from subject_number import subject_number as subjects
from joblib import Parallel, delayed

measure = 'emg'

if measure == 'emg':
    import DEAP_BIDS_config_emg as cfg
else:
    import DEAP_BIDS_config_eda as cfg
    
derivative_path = cfg.deriv_root

subjects = ['10']
subject = '10'

fname_covs = op.join(derivative_path, f'{measure}-cov-matrices-all-freqs', 'sub-' + subject + f'_covariances_{measure}.h5')

fname_epochs = derivative_path / 'clean-epo-files'

#%%

# Desarrollo pipeline para anlisis de datos de EMG
# 1. Band pass filter EMG at 20 hz- 256 hz
# (no se puede 512 hz. Maximo nfreq/2 segun documentacion de MNE)

epochs = mne.read_epochs(op.join(fname_epochs, 'sub-' + subject + '_task-rest_proc-clean_epo.fif'))
picks_emg = mne.pick_types(epochs.info, emg=True)
epochs = epochs.filter(l_freq = 20., h_freq = None, picks=picks_emg)
emg_epochs = epochs.copy().pick_channels(['EXG5','EXG6'])

# 2. EMG activity = EMG z1 - EMG z2

emg = emg_epochs.get_data()
emgz1 = emg[:,0,:]
emgz2 = emg[:,1,:]
#%%
# 3. Divide EMG activity in 100 ms bins (15 bins in 1.5 seconds)
#    (para lograr esto, descarto ultimo 4 valores de cada epoca, asi es divisible por 15 [100 ms * 15]])

emg_delta = emgz1 - emgz2

emg_delta  = emg_delta[:,:-4]
emg_delta_reshaped = emg_delta.reshape(len(emg_epochs), 15, 51)

# 4. Calculate  root-mean-square (over EMG activity bins)

# Eleva al cuadrado todos las restas ebtre emgz1 y emgz2 (emg delta)
emg_delta_squared = np.square(emg_delta_reshaped)
# Calcula el promedio de cada bin (15 promedios por epoca)
emg_rms = np.sqrt(emg_delta_squared.mean(axis=2))

# 5. Calculate Z-score of every bin

emg_rms_zscore = stats.zscore(emg_rms, axis=1)

# 6. Check if there are differences over 3.5 SD in a given bin

emg_rms_zscore_diff = np.diff(emg_rms_zscore)
emg_to_reject = np.logical_or(emg_rms_zscore_diff > 3.5, emg_rms_zscore_diff < - 3.5)
emg_to_reject_indices = np.unique(np.where(emg_to_reject == True)[0])

emg_all_indices = np.arange(0,len(emg_epochs))

indices_to_take = []
for item in emg_all_indices:
    if item not in emg_to_reject_indices:
        indices_to_take.append(item)

# 7. filter emg data based on rejections
emg_delta_filt = np.take(emg_delta, indices_to_take, 0)

#%%
# SEGUIR DESDE ACA -> 
# 1. Calculo de RMS para todas las epocas (1.5 secs)
emg_delta = emgz1 - emgz2
emg_delta_squared = np.square(emg_delta)
emg_rms = np.sqrt(emg_delta_squared.mean(axis=1))
emg_rms_series = pd.Series(emg_rms)

# 2. Implementacion de Hempel filtering
# SEGUIR DESDE ACA -> ESTOY DEBUGEANDO ESTA FUNCION. VOY POR DIFF.

def hampel(vals_orig, k=3, t0=1):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    
    #Make copy so original not edited
    vals = vals_orig.copy()
    
    #Hampel Filter
    L = 3.
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)
    
    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''
    
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx] 
    return(vals)

emg_rms_filtered = hampel(emg_rms_series)


#%%
# Calculo los RMSE luego de descartar epocas (RMSE finales)
emg = emg_epochs.get_data()
emgz1 = emg[:,0,:]
emgz2 = emg[:,1,:]
emg_delta = emgz1 - emgz2

emg_delta_squared = np.square(emg_delta)
emg_rms = np.sqrt(emg_delta_squared.mean(axis=1))

plt.plot(emg_rms[:3000]) 
plt.xticks(emg_to_reject_indices)

#%%
emg_rms_filt = np.take(emg_rms, indices_to_take, 0)
