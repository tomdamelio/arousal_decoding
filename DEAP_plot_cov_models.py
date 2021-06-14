# %%
import numpy as np
import os.path as op
import os
import pandas as pd
import matplotlib.pyplot as plt
import config as cfg
import seaborn

from subject_number import subject_number as subjects

if os.name == 'nt':
    derivative_path = 'C:/Users/dadam/OneDrive/Escritorio/tomas_damelio/outputs/DEAP-bids/derivatives/mne-bids-pipeline'  
else:
    derivative_path = 'storage/store/data/DEAP/outputs/DEAP-bids/derivatives/mne-bids-pipeline'

#score_name, scoring = "mae", "neg_mean_absolute_error"
score_name, scoring = "r2", "r2"

cv_name = 'shuffle-split'

all_scores = {}
for subject in subjects:
    score = np.load(op.join(derivative_path, 'sub-' + subject , 'eeg','sub-' + subject +
                    f'all_scores_models_DEAP_{score_name}_{cv_name}.npy'),
                        allow_pickle=True)[()]
    all_scores[subject] = score


# extracting the data

scores = pd.DataFrame(all_scores)
scores = scores.transpose()
scores = scores[['upper', 'log_diag', 'spoc', 'riemann']]

#%%

means = round(scores.mean(skipna = False), 2)
fig, ax = plt.subplots()

ax = seaborn.set_style("whitegrid")
box = seaborn.boxplot(data=scores, orient='h', showfliers=False, ax=ax)
ax = seaborn.swarmplot(data=scores, orient='h', alpha=0.5, ax=ax)
for i in range(len(means)):
    box.annotate(str(means[i]), xy=(means[i], i), horizontalalignment='center')
ax.set_xlabel('R2')
# %%
