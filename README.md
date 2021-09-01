The aim of the repository is to detail which analyses were conducted and to provide the code used in the Master thesis in Cognitive Sciences (ENS/PSL, Université de Paris, EHESS) of Tomás Ariel D'Amelio, supervised by Denis-Alexander Engemann:

# Predicting modelling of arousal: an analysis of a public EEG and EDA database

## Objectives

-  Disentangle the relationship between the EDA response and EEG predictors.
-  Evaluate the performance of our self-reported arousal decoding models.

## Key points
- A regression pipeline to predict electrodermal (and electromiographic) activity at the event-level with the data obtained from EEG recordings has been implemented.
Considering linear models were used, it has been possible to observe patterns of activity at the sensor level in terms of the main components of the predictions.
- We introduce a novel methodology for understanding the dynamics between central and peripheral nervous system signals, and their corresponding behavioural measures.
- This approach may be of particular interest to anyone who wants to work with the data provided by the DEAP database in Python, especially in continuous predictive modelling tasks.
## Our approach
First, EDA decoding from EEG would be performed, with the idea of boosting high-resolution signals in each subject. Thus, we would train models with continuous inputs (i.e. EDA) and outputs (i.e. EEG) in order to extract as much information as possible from each subject to learn different subject-level function approximations. As an output of this first step, we would represent arousal with a predicted EDA version. This predicted signal would indirectly portray the coupling of autonomic and cerebral arousal, as it is the result of the sum of the EEG features weighted by the different coefficients generated after the fitting process with the EDA output data. This would allow making use of the richness of these signals in a unique representation at the subject level. In this way, the predicted arousal would contain information that is not included in the original EDA data. Consequently, in a second step, we would predict self-reported arousal from the predicted EDA, constituting the second part of the proposed statistical learning approach.
## Details of the steps carried out during the organisation, processing and statistical analysis of the data:
### 1. Database
The data were downloaded directly from the DEAP dataset website (http://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html}), after completing and submitting the End User License Agreeement (EULA).
Although it was possible to work directly with DEAP pre-processed data in Python format (provided as a variant of the dataset), it was decided to use the original unprocessed recordings in BioSemi's .bdf format to allow more flexibility in epoching necessary for the continuous predictive task.
### 2. Pre-processing
The `convert_deap_to_bids.py` script was used to reorganise the original .bdf files provided by DEAP database into BIDS format.
In the `input_path` variable it is necessary to indicate the path where the DEAP data is located, while in `bids_root` it is necessary to indicate the path where the data will be saved according to the BIDS organisation. 
To check that the data was organised accordingly to the BIDS guidelines, BIDS-Validator was used (https://bids-standard.github.io/bids-validator/). This platform allows to online validate whether it is used BIDS correctly, and for example check if there are missing data. 
#### 2.1. Data segmentation
As continuous recordings were taken for each participant, they contained sections that had to be necessarily discarded from subsequent analyses if we intend to work only with data regarding participants' responses to affective stimuli.
Thus, segments were marked for rejection any time participants' recordings did not correspond to the presentation of the emotional stimuli (nor the fixation cross after the presentation of that stimulus). This selection of segments to reject was done programmatically, by implementing the script `run_annotations.py`.
#### 2.2. BIDS Pipeline
With the original data organised according to BIDS and with the annotations already appended to the raw files, we proceeded to pre-process the data following the MNE-BIDS-Pipeline (https://mne.tools/mne-bids-pipeline/). Specifically, the MNE-BIDS-Pipeline was used for data preprocessing (i.e. filtering, artifact rejection and epoching). What this pipeline provides is a systematic way to analyse BIDS raw data, by directly setting the processing parameters in a configuration file provided by the library. 
Different pipelines were made for EDA (`mne-bids-pipeline-eda`) and EMG (`mne-bids-pipeline-emg`), considering the length of the epochs was different depending on whether the target of the predictions was one or the other signal (shorter epochs for EMG). The config files with which the pipeline bids were run are DEAP_BIDS_config_emg.py and DEAP_BIDS_config_eda.py, where it is possible to find all the settings of the preprocessing.
### 3. Physiological modelling
Covariance matrices were computed from the epochs resulting from the MNE-BIDS-pipelines mentioned above. For the computation of these covariance matrices 7 frequency bands of the EEG signal were considered, from low (0.1 Hz, 1.5 Hz) to 'gamma' (35. Hz, 49. Hz) frequencies. The files for the calculation of these covariance matrices are:
`DEAP_compute_covs_eda.py`
`DEAP_compute_covs_emg.py`
Then, regressions were performed on the different predictive models presented in the thesis, both for EMG and EDA
The codes to obtain the results of the EMG and EDA mean and variance predictions are:
`DEAP_covariances_regression_eda_var.py`
`DEAP_covariances_regression_eda_mean.py`
`DEAP_covariances_regression_emg_var.py`
`DEAP_covariances_regression_emg_mean.py`
The only difference between these four scripts are two lines of code where we specify which measurement (i.e. EDA or EMG) and which descriptive statistic of the signal (i.e. mean or variace) used, parameters that can be changed at the beginning of these scripts.
From these files we obtained a) the predictions of each participant, b) the results of the optimisations of the components for SPoC and Riemann models, and c) the plots of the first components for SPoC models.
Then, to plot the performance of all participants, the script `fig_scores_eda_var.r` was run.
### 4. Predictive modelling of self-reported arousal
The predicted EDA, alongside the observed EDA, were subjected to a linear mixed-effects model with stimulus and subject as random effects. Uncertainty estimates were obtained from non-parametric bootstrapping of the entire process with 2000 iterations.
These analyses can be found in the `arousal_decoding.r` script.
