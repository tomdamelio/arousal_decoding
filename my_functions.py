def extract_signal(signal = 'EDA', directory = 'data', number_subject='01',
                   extension = '.bdf', info=False):
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
    from EEG_channel_name import channels_geneva, channels_twente 
    path = os.path.join(directory, 's'+ number_subject + extension)
    raw = mne.io.read_raw_bdf(path, preload=True)
    
    if signal == 'EDA':
        print('Number of channels in s1_temp:')
        print(len(raw.ch_names), end=' → pick only EDA → ')
        raw.pick_channels(['GSR1'])
        if info == True:
            print(len(raw.ch_names))
            print(raw.info)
    elif signal == 'EEG':
        print('Number of channels in s1_temp:')
        print(len(raw.ch_names), end=' → pick only EEG → ')
        if int(number_subject) < 23:
            raw.pick_channels(channels_twente)
        else:
            raw.pick_channels(channels_geneva)
        if info == True:
            print(len(raw.ch_names))
            print(raw.info)
    return raw
    
    