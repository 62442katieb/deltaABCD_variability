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
from sklearn.cluster import FeatureAgglomeration
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

# there are duplicated indices... why?
df.columns[df.columns.duplicated()]
#imputed_dcg = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data-{today}.csv")), 
#                          index_col='subjectkey', 
#                          header=0)

scanners = [
    "SIEMENS", 
    "GE MEDICAL SYSTEMS", 
    "Philips Medical Systems"
    ]

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
        'rsfmri_var_maxrot'
        ]
}

timepoints = [
    'baseline_year_1_arm_1',
    '2_year_follow_up_y_arm_1',
    'change_score'
]

scanner_ppts = {}
for scanner in scanners:
    scanner_ppts[scanner] = df[df['mri_info_manufacturer.baseline_year_1_arm_1'] == scanner].index

imaging = df.filter(regex='.*mri.*change_score')
for cov in [item for sublist in noisy_modalities.values() for item in sublist]:
    #print(cov)
    for tp in timepoints:
        if f'{cov}.{tp}' in imaging.columns:
            imaging = imaging.drop(f'{cov}.{tp}', axis=1)
imaging_cols = list(imaging.columns)
imaging_cols.append('rel_family_id.baseline_year_1_arm_1')


atlases = {'desikankillany': df,
           #'destrieux+gordon': imputed_dcg
        }

cluster_range = list(range(2,15))
#cluster_range = list(range(2,6))

#scoring = make_scorer(davies_bouldin_score, greater_is_better=False)

# hyper parameter tuning
iterations = 10
manager = enlighten.get_manager()
tocks = manager.counter(total=iterations * (len(cluster_range) ** 2) * len(scanners),
                        desc='Number of Iterations', 
                        unit='iter')
max_comp = {}
for atlas in atlases.keys():
    for scanner in scanners:
        ppts = scanner_ppts[scanner]
        best_params = pd.DataFrame()
        ppt_labels = pd.DataFrame(index=ppts)
        mri_labels = pd.DataFrame(index=imaging_cols)

        data = atlases[atlas]
        data = data.loc[ppts][imaging_cols]
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
            
            data = data.loc[all_subj]
            data.index.duplicated()
            temp_data = data.drop('rel_family_id.baseline_year_1_arm_1', axis=1).dropna()
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
            temp_data = pd.DataFrame(
                Normalizer().fit_transform(temp_data), 
                index=temp_data.index, 
                columns=temp_data.columns)
            #print(data.isna().sum())
        
            for n_clusters in cluster_range:
                for n_row_clust in cluster_range:
                    estimator = SpectralBiclustering(
                                    #n_best=n_best,
                                    #n_components=n_best + 1,
                                    method='bistochastic',
                                    n_clusters=(n_row_clust, n_clusters),
                                ).fit(temp_data)
                    parameters = pd.Series(estimator.get_params(), name=f'{n_row_clust, n_clusters} {i}')
                    parameters.at['ppt_clusters'] = n_row_clust
                    parameters.at['mri_clusters'] = n_clusters
                    
                    parameters.at['col_davies_bouldin'] =  davies_bouldin_score(temp_data.T, estimator.column_labels_)
                    parameters.at['row_davies_bouldin'] =  davies_bouldin_score(temp_data, estimator.row_labels_)

                    parameters.at['col_silhouette'] =  silhouette_score(temp_data.T, estimator.column_labels_)
                    parameters.at['row_silhouette'] =  silhouette_score(temp_data, estimator.row_labels_)

                    #print(temp_data.values.shape)
                    #print(np.sum(temp_data.columns.duplicated()), np.sum(temp_data.index.duplicated()))
                    cols = pd.Series(estimator.column_labels_, 
                                        name=f'{n_row_clust, n_clusters} {i}', 
                                        index=temp_data.columns)
                    rows = pd.Series(estimator.row_labels_, 
                                        name=f'{n_row_clust, n_clusters} {i}', 
                                        index=temp_data.index)
                    #print(np.sum(temp_data.columns.duplicated()))
                    
                    ppt_labels = pd.concat([ppt_labels, rows], axis=1)
                    mri_labels = pd.concat([mri_labels, cols], axis=1)
                    
                    
                    best_params = pd.concat([best_params, parameters], axis=1)
                    tocks.update()
                    
        try:
            ppt_labels.to_csv(join(PROJ_DIR, 
                                    OUTP_DIR, 
                                    f'sbc-dbs_{atlas}-{scanner}_ppt-labels-{today}.csv'))
            mri_labels.to_csv(join(PROJ_DIR, 
                                    OUTP_DIR, 
                                    f'sbc-dbs_{atlas}-{scanner}_mri-labels-{today}.csv'))
            best_params.T.to_csv(join(PROJ_DIR,
                                    OUTP_DIR,
                                    f'sbc-dbs_{atlas}-{scanner}_best-models-{today}.csv'))
        except:
            ppt_labels.to_csv(join('/Users/katherine.b/Dropbox/Projects/deltaABCD_clustering/output', 
                                    f'sbc-dbs_{atlas}-{scanner}_ppt-labels-{today}.csv'))
            mri_labels.to_csv(join('/Users/katherine.b/Dropbox/Projects/deltaABCD_clustering/output', 
                                    f'sbc-dbs_{atlas}-{scanner}_mri-labels-{today}.csv'))
            best_params.T.to_csv(join('/Users/katherine.b/Dropbox/Projects/deltaABCD_clustering/output',
                                    f'sbc-dbs_{atlas}-{scanner}_best-models-{today}.csv'))
