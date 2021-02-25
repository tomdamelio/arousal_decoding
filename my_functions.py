def extract_signal(directory = 'data', number_subject='01',
                   extension = '.bdf'):
    """
    extract_signal reads the biosignals of a given file format and extracts the signal
    we want to use for further analysis.

    :signal:         'EDA' (electrodermal activity, GSR); or 'EEG'
    :directory:      directory in which files are stored. Default=data
    :number_subject: '01' to '32' (DEAP database)
    :extension:      file extension. Default --> .bdf
    :info:           True or False. Print signal information.
    
    :return:         extracted signal (EEG or EDA) --> subject_n_temp
    """ 
    import os
    import mne
    path = os.path.join(directory, 's'+ number_subject + extension)
    raw = mne.io.read_raw_bdf(path, preload=True)
    return raw
    
def transform_negative_to_zero(x):
    x[x<0] = 0
    return x

def out_of_range(x):
    x[x>40] = 0 # set 40 uS ad hoc
    return x

def get_rejection_threshold(epochs, decim=1, random_state=None,
                            ch_types=None, cv=5, verbose=True):
    """Compute global rejection thresholds.
    Parameters
    ----------
    epochs : mne.Epochs object
        The epochs from which to estimate the epochs dictionary
    decim : int
        The decimation factor: Increment for selecting every nth time slice.
    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use.
    ch_types : str | list of str | None
        The channel types for which to find the rejection dictionary.
        e.g., ['mag', 'grad']. If None, the rejection dictionary
        will have keys ['mag', 'grad', 'eeg', 'eog'].
    cv : a scikit-learn cross-validation object
        Defaults to cv=5
    verbose : bool
        If False, suppress all output messages.
    Returns
    -------
    reject : dict
        The rejection dictionary with keys as specified by ch_types.
    Note
    ----
    Sensors marked as bad by user will be excluded when estimating the
    rejection dictionary.
    """
    from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, check_cv
    from autoreject import get_rejection_threshold
    from autoreject.autoreject import _GlobalAutoReject
    from autoreject.bayesopt import expected_improvement, bayes_opt
    from mne import Epochs
    from mne import pick_types
    reject = dict()

    if ch_types is not None and not isinstance(ch_types, (list, str)):
        raise ValueError('ch_types must be of type None, list,'
                         'or str. Got %s' % type(ch_types))

    if ch_types is None:
        ch_types = ['mag', 'grad', 'eeg', 'eog', 'misc']
    elif isinstance(ch_types, str):
        ch_types = [ch_types]

    if decim > 1:
        epochs = epochs.copy()
        epochs.decimate(decim=decim)

    cv = check_cv(cv)

    for ch_type in ch_types:
        if ch_type not in epochs:
            continue

        if ch_type == 'mag':
            picks = pick_types(epochs.info, meg='mag', eeg=False)
        elif ch_type == 'eeg':
            picks = pick_types(epochs.info, meg=False, eeg=True)
        elif ch_type == 'eog':
            picks = pick_types(epochs.info, meg=False, eog=True)
        elif ch_type == 'grad':
            picks = pick_types(epochs.info, meg='grad', eeg=False)
        elif ch_type == 'misc':
            picks = pick_types(epochs.info, meg=False, eeg=False, misc=True)

        X = epochs.get_data()[:, picks, :]
        n_epochs, n_channels, n_times = X.shape
        deltas = np.array([np.ptp(d, axis=1) for d in X])
        all_threshes = np.sort(deltas.max(axis=1))

        if verbose:
            print('Estimating rejection dictionary for %s' % ch_type)
        cache = dict()
        est = _GlobalAutoReject(n_channels=n_channels, n_times=n_times)

        def func(thresh):
            idx = np.where(thresh - all_threshes >= 0)[0][-1]
            thresh = all_threshes[idx]
            if thresh not in cache:
                est.set_params(thresh=thresh)
                obj = -np.mean(cross_val_score(est, X, cv=cv))
                cache.update({thresh: obj})
            return cache[thresh]

        n_epochs = all_threshes.shape[0]
        idx = np.concatenate((
            np.linspace(0, n_epochs, 5, endpoint=False, dtype=int),
            [n_epochs - 1]))  # ensure last point is in init
        idx = np.unique(idx)  # linspace may be non-unique if n_epochs < 5
        initial_x = all_threshes[idx]
        best_thresh, _ = bayes_opt(func, initial_x,
                                   all_threshes,
                                   expected_improvement,
                                   max_iter=10, debug=False,
                                   random_state=random_state)
        reject[ch_type] = best_thresh

    return reject