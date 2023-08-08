#!/usr/bin/env python
# coding: utf-8

import enlighten
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from os.path import join, exists
#from sklearn.experimental import enable_halving_search_cv
#from sklearn.model_selection import HalvingGridSearchCV#, train_test_split
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.preprocessing import Normalizer, StandardScaler

from utils import residualize

from datetime import datetime

today = datetime.today().strftime('%Y-%m-%d')

sns.set(style='white', context='paper')
plt.rcParams["font.family"] = "Avenir Next Condensed"


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))

noisy_modalities = {
    'smri': [
        "smri_vol_cdk_total"
        ],
    'dmri': [
        'dmri_rsi_meanmotion',
        'dmri_rsi_meantrans',
        'dmri_rsi_meanrot'
        ], 
    'rsfmri': [
        'rsfmri_var_ntpoints', 
        'rsfmri_var_meanmotion',
        'rsfmri_var_meantrans',
        'rsfmri_var_maxtrans',
        'rsfmri_var_meanrot',
        'rsfmri_var_maxrot',
        ]
}


timepoints = [
    'baseline_year_1_arm_1',
    '2_year_follow_up_y_arm_1',
    'change_score'
]

imaging = df.filter(regex='.*mri.*change_score')
for cov in [item for sublist in noisy_modalities.values() for item in sublist]:
    #print(cov)
    for tp in timepoints:
        if f'{cov}.{tp}' in imaging.columns:
            imaging = imaging.drop(f'{cov}.{tp}', axis=1)
imaging_cols = list(imaging.columns)
imaging_cols.append('rel_family_id.baseline_year_1_arm_1')


scanners = [
    "SIEMENS", 
    "GE MEDICAL SYSTEMS", 
    "Philips Medical Systems"
    ]

scanner_ppts = {}
for scanner in scanners:
    scanner_ppts[scanner] = df[df['mri_info_manufacturer.baseline_year_1_arm_1'] == scanner].index


# hyper parameter tuning
iterations = 10
manager = enlighten.get_manager()
tocks = manager.counter(total=iterations * len(scanners),
                        desc='Number of Iterations', 
                        unit='iter')
max_comp = {}

for scanner in scanners:
    ppts = scanner_ppts[scanner]
    best_params = pd.DataFrame()
    ppt_labels = pd.DataFrame(index=ppts, dtype=int)

    data = df.loc[ppts][imaging_cols]
    #print("duplicated columns",
    #      np.sum(data.columns.duplicated()),
    #      '\n', data.columns[data.columns.duplicated()])
    for i in range(0,iterations):
        all_subj = data.index.to_list()
        for id_ in data['rel_family_id.baseline_year_1_arm_1']:
            siblings = data[data['rel_family_id.baseline_year_1_arm_1'] == id_].index.to_list()
            if len(siblings) > 1:
                keep = np.random.choice(siblings)
                siblings.remove(keep)
                all_subj = list(set(all_subj) - set(siblings))
            else:
                pass
        
        temp_data = data.loc[all_subj].dropna()
        temp_data = temp_data.drop('rel_family_id.baseline_year_1_arm_1', axis=1)
        
        #print(scanner, 'ppts ', len(temp_data.index), '\tcols', len(temp_data.columns))
        resid_temp = pd.DataFrame()
        for modality in noisy_modalities.keys():
            cov_df = pd.DataFrame()
            mini_dset = temp_data.filter(like=modality)
            subj = temp_data.index
            img_cols = mini_dset.columns
            covs = []
            for covariate in noisy_modalities[modality]:
                smol = df.loc[subj]
                covs.append(f'{covariate}.baseline_year_1_arm_1')
                covs.append(f'{covariate}.2_year_follow_up_y_arm_1')
                cov_df = pd.concat([cov_df, smol[f'{covariate}.baseline_year_1_arm_1']], axis=1)
                cov_df = pd.concat([cov_df, smol[f'{covariate}.2_year_follow_up_y_arm_1']], axis=1)
            #print(img_cols, covs)
            mini_dset = pd.concat([mini_dset, cov_df], axis=1)
            #print(mini_dset.describe())
            temp2 = residualize(mini_dset[img_cols], confounds=mini_dset[covs])
            resid_temp = pd.concat([resid_temp, temp2], axis=1)

        #print(len(resid_temp.columns))
        temp_data = pd.DataFrame(
            Normalizer().fit_transform(resid_temp), 
            index=temp_data.index, 
            columns=temp_data.columns)
        #print(data.isna().sum())
    
        #print(len(temp_data.columns))
        estimator = MeanShift(
                        bin_seeding=True,
                        cluster_all=True,
                        n_jobs=4,
                    ).fit(temp_data)
        parameters = pd.Series(estimator.get_params(), name=i)
        temp_labels = pd.Series(dtype=int)
        for scanner2 in scanners:
            # now we need data from other participants, scanned on scanner 2
            ppts2 = scanner_ppts[scanner2]
            # and we need to fetch their data
            data2 = df.loc[ppts2]
            # but just their imaging data
            temp2 = data2[imaging_cols]
            # oh and also the covariates
            cov2 = data2[covs]
            # then we put it together, so we get a df with covariates and imaging data
            pre_resid = pd.concat([temp2, cov2], axis=1)
            resid_temp = pd.DataFrame()
            for modality in noisy_modalities.keys():
                # narrow down to just data from this modality
                mini_brain = temp2.filter(like=modality, axis=1)
                mini_covs = cov2.filter(like=modality, axis=1)
                
                img_cols = mini_brain.columns
                cov_cols = mini_covs.columns
                #print(mini_dset.columns)
                # bring covariates and data together 
                # so we can eliminate all missing values
                mini_dset = pd.concat([mini_brain, mini_covs], axis=1).dropna()
                
                #print(mini_dset.columns)
                temp3 = residualize(mini_dset[img_cols], confounds=mini_dset[cov_cols])
                resid_temp = pd.concat([resid_temp, temp3], axis=1)
            #temp_data = data.drop(drops, axis=1).dropna()
            resid_temp = resid_temp.dropna()
            normed_data = pd.DataFrame(
                Normalizer().fit_transform(resid_temp),
                index=resid_temp.index, 
                columns=resid_temp.columns
                )
            normed_data = normed_data[estimator.feature_names_in_]
            solution = estimator.predict(normed_data)
            n_clusters = len(np.unique(solution))
            parameters.at[f'n_clusters-{scanner2}'] = n_clusters
            if n_clusters > 1:
                parameters.at[f'davies_bouldin-{scanner2}'] =  davies_bouldin_score(normed_data, solution)
                parameters.at[f'calinski_harabasz-{scanner2}'] = calinski_harabasz_score(normed_data, solution)
                parameters.at[f'silhouette-{scanner2}'] =  silhouette_score(normed_data, solution)

            #print(temp_data.values.shape)
            #print(np.sum(temp_data.columns.duplicated()), np.sum(temp_data.index.duplicated()))
            labels = pd.Series(solution, 
                                name=f'{scanner2} {i}', 
                                index=resid_temp.index)
            temp_labels = pd.concat([temp_labels, labels], axis=0)
        ppt_labels = pd.concat([ppt_labels, temp_labels], axis=0)
        best_params = pd.concat([best_params, parameters], axis=1)
        tocks.update()
        
    try:
        ppt_labels.to_csv(join(PROJ_DIR, 
                                OUTP_DIR, 
                                f'MeanShift-{scanner}_ppt-labels-{today}.csv'))
        best_params.T.to_csv(join(PROJ_DIR,
                                OUTP_DIR,
                                f'MeanShift-{scanner}_params-{today}.csv'))
    except:
        ppt_labels.to_csv(join('/Users/katherine.b/Dropbox/Projects/deltaABCD_clustering/output', 
                                f'MeanShift-{scanner}_ppt-labels-{today}.csv'))
        best_params.T.to_csv(join('/Users/katherine.b/Dropbox/Projects/deltaABCD_clustering/output',
                                f'MeanShift-{scanner}_params-{today}.csv'))

    # Reordering first the rows and then the columns.
