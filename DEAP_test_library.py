#%%
import numpy as np
import pandas as pd
from meegpowreg import make_filter_bank_regressor

#%%
# Define a dict 'freq_bands' with keys 'alpha' and 'beta' frequencies.
freq_bands = {'alpha': (8.0, 15.0), 'beta': (15.0, 30.0)}
n_freq_bands = len(freq_bands) #2
n_subjects = 10
n_channels = 4

#%%
# Make toy data
# Create random array with shape (n_subjects, n_freq_bands,
# n_channels, n_channels)
X_cov = np.random.randn(n_subjects, n_freq_bands, n_channels, n_channels)
# for every subject
for sub in range(n_subjects):
      # and every frequency band
    for fb in range(n_freq_bands):
        # Calculate covariance matrix (A * AT) of channels x channels  
        X_cov[sub, fb] = X_cov[sub, fb] @ X_cov[sub, fb].T
# column name -> 'alpha' and 'beta'
X_df = pd.DataFrame(
  {band: list(X_cov[:, ii]) for ii, band in enumerate(freq_bands)})
# add column 'drug' with size n_subject
X_df['drug'] = np.random.randint(2, size=n_subjects)
# Define our y in the model
y = np.random.randn(len(X_df))

#%%
# Models
## names     -> The column names of the data frame corresponding
#               to different covariances -> 'alpha' and 'beta'
## method    -> Method used for extracting features from covariances
## estimator -> Defaults -> RidgeCV(alphas=np.logspace(-3, 5, 100))
## scaling   -> Defaults -> StandardScaler()
fb_model = make_filter_bank_regressor(names=freq_bands.keys(),
                                      method='riemann')

#%%
fb_model.fit(X_df, y).predict(X_df)

# %%
