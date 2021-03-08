## channel_names.py
- input   --> -
- process --> channels' order according to location (twente or geneva)
- output  -->
    -     channels_twente --> list. Order of channels according to twente configutarion
    -     channels_geneva --> list. Order of channels according to geneva configutarion

## decoding_EDA_with_EEG.py
- input   --> one subject (fixed on subject 26)
- process --> EDA prediction from EEG
- output  --> Plot the True EDA power and the EDA predicted from EEG data 
- comment --> DEPRECATED. ValueError. I was trying to predict the signal and not a variance.

## decoding_EDA_with_error.py
- input   --> one subject (fixed on subject 01)
- process --> EDA prediction from EEG
- output  --> Plot the True EDA mean and the EDA predicted from EEG data 
- comment -->
    -    89 events after rejection (raw = 483 events).
    -    Predict EDA's mean. 
    -    Noisy plot of True EDA. Underfitted predicted EDA.

## decoding_EDA.py
- input   --> one subject (fixed on subject 02)
- process --> EDA prediction from EEG
- output  --> Plot the True EDA power and the EDA predicted from EEG data
- comment --> 
    -    DEPRECATED. ValueError. expected square "a" matrix
    -    51 events after rejection (raw = 462 events).

## EDA_plot_all_subjects.py
- input   --> 
- process --> 
- output  --> 
- comment --> 


## example_pipeline_decoding.py
- input   --> one subject (fixed on subject 01)
- process --> adaptation of https://mne.tools/dev/auto_examples/decoding/plot_decoding_spoc_CMC.html#sphx-glr-auto-examples-decoding-plot-decoding-spoc-cmc-py
- output  --> Plot the True EDA power and the EDA predicted from EEG data
- comment --> 
    -    DEPRECATED. ValueError: y must have at least two distinct values.


## exploratory_analysis_EDA.py
- input   --> one subject (fixed on subject 01)
- process --> exploratory data analysis
- output  --> plot raw EDA subt 01
- comment --> 
    -    without data cleaning


## exploratory_analysis_EEG.py
- input   --> one subject (fixed on subject 10)
- process --> read one subject a save variable df_s1_EEG (dataframe of one subject's EEG signal)
- output  --> df EEG
    - comment --> DEPRECATED. Because we don't want to separate EEG and EDA

## filters.py
- input   --> -
- process --> Create variable band pass filter
- output  --> def variable bandPassFilter
- comment --> DEPRECATED. We are going to use band pass filter from MNE 

## Plot_comparison_detrended_not_detrended_EDA.py
- input   --> one subject (fixed on subject 01)
- process --> Plot comparison between signal and signal detrended of a subject
- output  --> plot EDA and EDA detrended (matplotlib) of subject 01

## plot_EDA_value_autoreject.py
- input   --> data of some subjects (defined in 'subject_number_reduced' list)
- process --> Plot to know how many events where discarded after event rejection
- output  --> Barplot of EDA events of each subject before and after rejection with autoreject

## preprocessing.py
- input   --> -
- process --> define variables for preprocessing
- output  --> def variable bandPassFilter
- comment --> DEPRECATED. We are going to use band pass filter from MNE 

## PSD_analysis_EDA.py
- input   --> -
- process --> plot EDA PSD of all subjects 
- output  --> plots saved in 'subject_plots/PSD'
- comment --> DEPRECATED. We are going to use band pass filter from MNE 

## subject_number.py
 -input   --> -
- process --> create list with number of the subjects
- output  --> list 'subject_number'. Lenght: 32

