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
from sklearn.linear_model import LinearRegression


sns.set(style='whitegrid', context='paper')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

imputed_cdk = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "data_qcd_mice-cdk.csv")), 
                          index_col='subjectkey', 
                          header=0)
#imputed_dcg = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data.csv")), 
#                          index_col='subjectkey', 
#                          header=0)

scanners = ["SIEMENS", "GE MEDICAL SYSTEMS", "Philips Medical Systems"]

scanner_ppts = {}
for scanner in scanners:
    scanner_ppts[scanner] = imputed_cdk[imputed_cdk['mri_info_manufacturer.baseline_year_1_arm_1'] == scanner].index

imaging_cols = list(imputed_cdk.filter(regex='.*mri.*change_score').columns)
imaging_cols.append('rel_family_id.baseline_year_1_arm_1', )



atlases = {'desikankillany': imputed_cdk,
           #'destrieux+gordon': imputed_dcg
        }

parameter_grid = {
    'weight_concentration_prior_type': [
        'dirichlet_process', 
        'dirichlet_distribution'],
    'weight_concentration_prior': [
        10**-1,
        10**0,
        10**1, 
        10**2, 
    ],
    'n_components': list(range(2,16)),
    'covariance_type': [
        'full',
        'tied',
        'diag', 
        'spherical']
}

estimator = BayesianGaussianMixture(
    max_iter=1000
)

# hyper parameter tuning
iterations = 100
manager = enlighten.get_manager()
tocks = manager.counter(total=iterations, * len(atlases.keys()),
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
            data.drop('rel_family_id.baseline_year_1_arm_1', axis=1, inplace=True)

            # need train test split
            search = HalvingGridSearchCV(estimator, 
                                        parameter_grid, 
                                        factor=2, 
                                        cv=10, 
                                        n_jobs=-1).fit(data)
            parameters = pd.Series(search.best_estimator_.get_params(), name=i)
            parameters['test_score'] = search.best_estimator_.score(data)
            labels = search.best_estimator_.predict(data)
            for k in range(0, parameters['n_components']):
                parameters.loc[k] = np.sum(labels == k)
                if np.max(labels) > n_comps:
                    n_comps = np.max(labels)
            nonzero_components = np.sum(parameters.loc[list(range(0, parameters['n_components']))] != 0)
            parameters.loc['n_components_nonzero'] = nonzero_components
            
            results = pd.DataFrame.from_dict(search.cv_results_)
            results["params_str"] = results.params.apply(str)
            
            big_results = pd.concat([big_results, results], axis=0)
            
            best_params = pd.concat([best_params, parameters], axis=1)
            tocks.update()
        big_results.to_csv(join(PROJ_DIR, 
                                OUTP_DIR, 
                                f'bgmm_{atlas}-{scanner}_cv-results0.csv'))
        best_params.to_csv(join(PROJ_DIR,
                                OUTP_DIR,
                                f'bgmm_{atlas}-{scanner}_best-models0.csv'))
        max_comp[(atlas, scanner)] = n_comps


#indices = range(0, 14
#                #max_comp['destrieux+gordon']
#                )
#gdn_indices = [str(i) for i in indices]

indices = range(0, 14
                #max_comp['desikankillany']
                )
cdk_indices = [str(i) for i in indices]

for atlas in atlases:
    scanner_models = {}
    fig,ax = plt.subplots(nrows=3, ncols=len(scanners.keys()), figsize=(7,10))
    plt.tight_layout(pad=4)
    i = 0
    for scanner in scanners:
        temp = pd.read_csv(join(PROJ_DIR,
                                    OUTP_DIR,
                                    f'bgmm_{atlas}-{scanner}_best-models0.csv'), index_col=0, header=0)
        
        temp = temp.T
        #gdn_models = gdn_models.T
        #change datatype of test_score
        temp['test_score'] = temp['test_score'].astype(float)
        #gdn_models['test_score'] = gdn_models['test_score'].astype(float)
        temp['n_components'] = temp['n_components'].astype(int)
        #gdn_models['n_components'] = gdn_models['n_components'].astype(int)
        temp['n_components_nonzero'] = temp['n_components'].astype(int)
        #gdn_models['n_components_nonzero'] = gdn_models['n_components'].astype(int)

        nonzero_components = np.sum(temp[cdk_indices].fillna(0).astype(int) > 1, axis=1)
        temp['n_components_nonzero'] = nonzero_components

        #nonzero_components = np.sum(gdn_models[gdn_indices].fillna(0).astype(int) > 1, axis=1)
        #gdn_models['n_components_nonzero'] = nonzero_components

        temp['weight_concentration_prior'] = temp['weight_concentration_prior'].astype(float)
        #gdn_models['weight_concentration_prior'] = gdn_models['weight_concentration_prior'].astype(float)
        scanner_models[scanner] = temp

    
        #how often was each # components the best option
        sns.histplot(x='n_components_nonzero',
                    data=temp.sort_values('n_components'), 
                    ax=ax[0][0])
        ax[0][i].set_title(scanner)
        ax[0][i].set_xlabel('Non-Empty Components')
        ax[0][i].set_ylabel('Frequency')

        #what were the scores of each # components, does it depend on covariance type?
        sns.scatterplot(x='n_components',
                    y='test_score',
                    hue='covariance_type',
                    data=temp.sort_values('n_components'),
                        palette='Pastel1', 
                    ax=ax[1][0])
        ax[1][i].set_title(f'Number of Components')
        ax[1][i].set_xlabel('Components')
        ax[1][i].set_ylabel('Log Likelihood')
        ylabels = [f'{int(y)}' for y in ax[1][i].get_yticks()]
        ax[1][i].set_yticklabels(ylabels)
        ax[1][i].get_legend().remove()

        #what weight concentration prior settings had best scores
        sns.scatterplot(x='weight_concentration_prior',
                    y='test_score',
                    hue='weight_concentration_prior_type',
                    data=temp,
                        palette='Pastel2',
                    ax=ax[2][0])
        ax[2][i].set_title(f'Weight Concentration Prior')
        ax[2][i].set_xlabel('Prior')
        ax[2][i].set_ylabel('Log Likelihood')
        ylabels = [f'{int(y)}' for y in ax[2][i].get_yticks()]
        ax[2][i].set_yticklabels(ylabels)
        ax[2][i].get_legend().remove()

        i += 1

    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'bgmm_{atlas}_best-models.png'),
                dpi=500, bbox_inches='tight')