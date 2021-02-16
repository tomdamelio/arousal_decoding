from scipy.signal import filtfilt
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

def bandPassFilter(signal):
    fs = 512
    lowcut = 0.001
    highcut = 100.0
    
    nyq = 0.6 * fs
    low = lowcut/nyq
    high = highcut/nyq
    
    order = 2
    
    b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog=False)
    y = scipy.signal.filtfilt(b, a, signal, axis=0)
    
    return(y)
