#!/usr/bin/env python
# coding: utf-8

import enlighten
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join, exists
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.mixture import BayesianGaussianMixture
#from sklearn.linear_model import LinearRegression


sns.set(style='whitegrid', context='paper')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

imputed_cdk = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_knn-cdk.pkl"))
#imputed_dcg = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data.csv")), 
#                          index_col='subjectkey', 
#                          header=0)

scanners = [
    "SIEMENS", 
    "GE MEDICAL SYSTEMS", 
    #"Philips Medical Systems"
    ]

scanner_ppts = {}
for scanner in scanners:
    scanner_ppts[scanner] = imputed_cdk[imputed_cdk['mri_info_manufacturer.baseline_year_1_arm_1'] == scanner].index

imaging_cols = list(imputed_cdk.filter(regex='.*mri.*change_score').columns)
imaging_cols.append('rel_family_id.baseline_year_1_arm_1', )



atlases = {'desikankillany': imputed_cdk,
           #'destrieux+gordon': imputed_dcg
        }

n_components = 30

parameter_grid = {
    'weight_concentration_prior_type': [
        'dirichlet_process', 
        'dirichlet_distribution'],
    'weight_concentration_prior': [
        10**-1, 
        10**-2,
        10**0,
        10**1, 
        10**2,
        10**3,
        10**-3,
        10**4,
    ],
    #'n_components': list(range(2,10)),
    'covariance_type': [
        'diag'
        ]
}

estimator = BayesianGaussianMixture(
    n_components=n_components,
    max_iter=1000
)

# hyper parameter tuning
iterations = 1000
manager = enlighten.get_manager()
tocks = manager.counter(total=iterations * len(atlases.keys()) * len(scanners),
                        desc='Number of Iterations', 
                        unit='iter')
max_comp = {}
for atlas in atlases.keys():
    for scanner in scanners:
        big_results = pd.DataFrame()
        best_params = pd.DataFrame()
        n_comps = 0
        ppts = scanner_ppts[scanner]
        data = atlases[atlas]
        data = data.loc[ppts][imaging_cols]
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
            temp_data = data.drop('rel_family_id.baseline_year_1_arm_1', axis=1)
            #print(data.isna().sum())

            # need train test split
            search = HalvingGridSearchCV(estimator, 
                                        parameter_grid, 
                                        factor=4, 
                                        min_resources=200,
                                        cv=5,
                                        verbose=1, 
                                        n_jobs=-1).fit(temp_data)
            parameters = pd.Series(search.best_estimator_.get_params(), name=i)
            parameters['test_score'] = search.best_estimator_.score(temp_data)
            labels = search.best_estimator_.predict(temp_data)
            for k in range(0, parameters['n_components']):
                parameters.loc[k] = np.sum(labels == k)
                if np.max(labels) > n_comps:
                    n_comps = np.max(labels)
            nonzero_components = np.sum(parameters.loc[list(range(0, parameters['n_components']))] != 0)
            parameters.loc['n_components_nonzero'] = nonzero_components
            
            #results = pd.DataFrame.from_dict(search.cv_results_)
            #results["params_str"] = results.params.apply(str)
            parameters.loc["converged"] = search.best_estimator_.converged_
            
            #big_results = pd.concat([big_results, results], axis=0)
            
            best_params = pd.concat([best_params, parameters], axis=1)
            tocks.update()
        try:
            #big_results.to_csv(join(PROJ_DIR, 
            #                        OUTP_DIR, 
            #                        f'bgmm_{atlas}-{scanner}_cv-results.csv'))
            best_params.T.to_csv(join(PROJ_DIR,
                                    OUTP_DIR,
                                    f'bgmm_{atlas}-{scanner}_best-models.csv'))
        except:
            #big_results.to_csv(join('..', 
            #                        OUTP_DIR, 
            #                        f'bgmm_{atlas}-{scanner}_cv-results.csv'))
            best_params.T.to_csv(join('..',
                                    OUTP_DIR,
                                    f'bgmm_{atlas}-{scanner}_best-models.csv'))
