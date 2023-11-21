#!/usr/bin/env python
# coding: utf-8

import enlighten
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join, exists
from sklearn.mixture import BayesianGaussianMixture

from nilearn import plotting

from scipy.stats import spearmanr, pointbiserialr
from utils import jili_sidak_mc, series_2_nifti, assign_region_names
#from sklearn.linear_model import LinearRegression

from datetime import datetime

today = datetime.today().strftime('%Y-%m-%d')

sns.set(style='whitegrid', context='paper')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

imputed_cdk = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))
#imputed_dcg = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data.csv")), 
#                          index_col='subjectkey', 
#                          header=0)

scanners = [
    "GE MEDICAL SYSTEMS", 
    #"Philips Medical Systems", 
    "SIEMENS"
    ]

brain_meas = {
    'thick': 'smri_thick.*change_score',
    'rni': 'dmri_rsirni.*change_score',
    'rnd': 'dmri_rsirnd.*change_score',
    'var': 'rsfmri_var.*change_score',
}

scanner_ppts = {}
for scanner in scanners:
    scanner_ppts[scanner] = imputed_cdk[imputed_cdk['mri_info_manufacturer.baseline_year_1_arm_1'] == scanner].index

imaging_cols = list(imputed_cdk.filter(regex='.*mri.*change_score').columns)

atlases = {'desikankillany': imputed_cdk,
           #'destrieux+gordon': imputed_dcg
        }

### update with "best" params
estimator = BayesianGaussianMixture(
    weight_concentration_prior_type='dirichlet_distribution', 
    weight_concentration_prior=0.001, 
    covariance_type='diag',
    n_components=30,
    max_iter=1000
)

# train on Siemens, test on GE?!
results = pd.DataFrame()
ppts = scanner_ppts["SIEMENS"]
siemens_data = imputed_cdk.loc[ppts][imaging_cols].dropna()

ppts = scanner_ppts["GE MEDICAL SYSTEMS"]
ge_data = imputed_cdk.loc[ppts][imaging_cols]

siemens_subj = siemens_data.index.to_list()
ge_subj = ge_data.index.to_list()

siemens_data = siemens_data.loc[siemens_subj]
ge_data = ge_data.loc[ge_subj]
#print(data.isna().sum())

siemens = estimator.fit(siemens_data)
probs = siemens.predict_proba(siemens_data)
score = np.round(siemens.score(siemens_data), 3)
weights = pd.DataFrame(siemens.weights_, 
                        index=list(range(0,estimator.n_components)))
means = pd.DataFrame(siemens.means_.T, 
                        index=siemens.feature_names_in_,
                        columns=list(range(0,estimator.n_components)))
responsibilities = pd.DataFrame(probs, 
                                index=siemens_subj, 
                                columns=list(range(0,estimator.n_components)))
keep = list(responsibilities.columns[responsibilities.sum() > 0])
responsibilities = responsibilities[keep]

means = means[keep]
means = assign_region_names(means)

ge_predicts = siemens.predict(ge_data)
ge_responsibilities = siemens.predict_proba(ge_data)
ge_score = siemens.score(ge_data)

for atlas in atlases.keys():
    for scanner in scanners:
        results = pd.DataFrame()
        ppts = scanner_ppts[scanner]
        data = atlases[atlas]
        data = data.loc[ppts][imaging_cols]
        
        all_subj = data.index.to_list()
        
        data = data.loc[all_subj]
        #print(data.isna().sum())

        # need train test split
        clustering = estimator.fit(data)
        probs = clustering.predict_proba(data)
        score = np.round(clustering.score(data), 3)
        weights = pd.DataFrame(clustering.weights_, 
                             index=list(range(0,estimator.n_components)))
        means = pd.DataFrame(clustering.means_.T, 
                             index=clustering.feature_names_in_,
                             columns=list(range(0,estimator.n_components)))
        responsibilities = pd.DataFrame(probs, 
                                        index=all_subj, 
                                        columns=list(range(0,estimator.n_components)))
        keep = list(responsibilities.columns[responsibilities.sum() > 0])
        responsibilities = responsibilities[keep]
        
        means = means[keep]
        means = assign_region_names(means)
        #print(scanner, '\t', score)
        
        try:
            responsibilities.to_csv(join(PROJ_DIR, 'output', f'responsibilities-{scanner}-{today}.csv'))
            weights.to_csv(join(PROJ_DIR, 'output', f'weights-{scanner}-{today}.csv'))
            means.to_csv(join(PROJ_DIR, 'output', f'brain_means-{scanner}-{today}.csv'))
        except:
            responsibilities.to_csv(f'responsibilities-{scanner}-{today}.csv')
            weights.to_csv(f'means-{scanner}-{today}.csv')
            means.to_csv(f'brain_means-{scanner}-{today}.csv')
        
        components = list(responsibilities.columns)
        stats = ['p', 'r']
        columns = pd.MultiIndex.from_product([components, stats])
        corrs = pd.DataFrame(index=data.columns, columns=columns)
        for resp in components:
            #print(resp)
            for col in data.columns:
                #print(col)
                if len(responsibilities[resp].unique()) > 2:
                    r, p = spearmanr(responsibilities.loc[:][resp], data[col])
                    corrs.at[col, (resp, 'r')] = r
                    corrs.at[col, (resp, 'p')] = p
                else:
                    r, p = pointbiserialr(responsibilities.loc[:][resp], data[col])
                    corrs.at[col, (resp, 'r')] = r
                    corrs.at[col, (resp, 'p')] = p
        corrs = assign_region_names(corrs)
        try:
            corrs.to_csv(join(PROJ_DIR, 'output', f'responsibilities_by_brain-{scanner}-{today}.csv'))
        except:
            corrs.to_csv(f'responsibilities_by_brain-{scanner}-{today}.csv')
        
        #plot the significant correlations on the brainnnnnnnnn
        #first, mask the correlations by significance
        alpha, _ = jili_sidak_mc(data, 0.05)
        for comp in components:
            for meas in brain_meas.keys():
                print('plotting', comp, meas)
                dat = means.filter(regex=brain_meas[meas], axis=0)[comp]
                dat.name = f'means-{scanner}_{str(comp)}-{meas}'
                series_2_nifti(dat, 'figures')
                plotting.plot_img_on_surf(
                    f'figures/means-{scanner}_{str(comp)}-{meas}.nii',
                    cmap='bwr', 
                    threshold=0.0001,
                    symmetric_cbar=True, 
                    output_file=f'figures/means-{scanner}_{str(comp)}-{meas}.png'
                    )
                plt.close()
