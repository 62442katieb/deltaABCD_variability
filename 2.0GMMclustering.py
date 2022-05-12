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


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

imputed_cdk = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "desikankillany_MICEimputed_data.csv")), 
                          index_col='subjectkey', 
                          header=0)
imputed_dcg = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data.csv")), 
                          index_col='subjectkey', 
                          header=0)

atlases = {'desikankillany': imputed_cdk,
           'destrieux+gordon': imputed_dcg}

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
tocks = manager.counter(total=iterations, * len(atlases.keys(),
                        desc='Number of Iterations', 
                        unit='iter')
max_comp = {}
for atlas in atlases.keys():
    if atlas == 'desikankillany':
        pass
    else:
        big_results = pd.DataFrame()
        best_params = pd.DataFrame()
        n_comps = 0
        for i in range(0,iterations):
            data = atlases[atlas]
            all_subj = data.index.to_list()
            for id_ in data['rel_family_id']:
                siblings = data[data['rel_family_id'] == id_].index.to_list()
                if len(siblings) > 1:
                    keep = np.random.choice(siblings)
                    siblings.remove(keep)
                    all_subj = list(set(all_subj) - set(siblings))
                else:
                    pass
            data = data.loc[all_subj]
            data.drop('rel_family_id', axis=1, inplace=True)

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
            big_results.to_csv(join(PROJ_DIR, 
                                OUTP_DIR, 
                                f'bgmm_{atlas}_cv-results0.csv'))
            best_params = pd.concat([best_params, parameters], axis=1)
            best_params.to_csv(join(PROJ_DIR,
                                    OUTP_DIR,
                                    f'bgmm_{atlas}_best-models0.csv'))
            tocks.update()
        max_comp[atlas] = n_comps


indices = range(0, 14
                #max_comp['destrieux+gordon']
                )
gdn_indices = [str(i) for i in indices]

indices = range(0, 14
                #max_comp['desikankillany']
                )
cdk_indices = [str(i) for i in indices]


cdk_models = pd.read_csv(join(PROJ_DIR, 
                            OUTP_DIR, 
                            f'bgmm_desikankillany_best-models.csv'), 
                     index_col=0, 
                     header=0)
gdn_models = pd.read_csv(join(PROJ_DIR, 
                            OUTP_DIR, 
                            f'bgmm_destrieux+gordon_best-models.csv'), 
                     index_col=0, 
                     header=0)

cdk_models = cdk_models.T
gdn_models = gdn_models.T
#change datatype of test_score
cdk_models['test_score'] = cdk_models['test_score'].astype(float)
gdn_models['test_score'] = gdn_models['test_score'].astype(float)
cdk_models['n_components'] = cdk_models['n_components'].astype(int)
gdn_models['n_components'] = gdn_models['n_components'].astype(int)
cdk_models['n_components_nonzero'] = cdk_models['n_components'].astype(int)
gdn_models['n_components_nonzero'] = gdn_models['n_components'].astype(int)

nonzero_components = np.sum(cdk_models[cdk_indices].fillna(0).astype(int) > 1, axis=1)
cdk_models['n_components_nonzero'] = nonzero_components

nonzero_components = np.sum(gdn_models[gdn_indices].fillna(0).astype(int) > 1, axis=1)
gdn_models['n_components_nonzero'] = nonzero_components

cdk_models['weight_concentration_prior'] = cdk_models['weight_concentration_prior'].astype(float)
gdn_models['weight_concentration_prior'] = gdn_models['weight_concentration_prior'].astype(float)


fig,ax = plt.subplots(nrows=3, ncols=2, figsize=(7,7))
plt.tight_layout(pad=4)
#how often was each # components the best option
sns.histplot(x='n_components_nonzero',
              data=cdk_models.sort_values('n_components'), 
              ax=ax[0][0])
ax[0][0].set_title(f'Desikan & Killany')
ax[0][0].set_xlabel('Non-Empty Components')
ax[0][0].set_ylabel('Frequency')

#what were the scores of each # components, does it depend on covariance type?
sns.scatterplot(x='n_components',
              y='test_score',
              hue='covariance_type',
              data=cdk_models.sort_values('n_components'),
                palette='Pastel1', 
              ax=ax[1][0])
ax[1][0].set_title(f'Number of Components')
ax[1][0].set_xlabel('Components')
ax[1][0].set_ylabel('Log Likelihood')
ylabels = [f'{int(y)}' for y in ax[1][0].get_yticks()]
ax[1][0].set_yticklabels(ylabels)
ax[1][0].get_legend().remove()

#what weight concentration prior settings had best scores
sns.scatterplot(x='weight_concentration_prior',
              y='test_score',
              hue='weight_concentration_prior_type',
              data=cdk_models,
                palette='Pastel2',
              ax=ax[2][0])
ax[2][0].set_title(f'Weight Concentration Prior')
ax[2][0].set_xlabel('Prior')
ax[2][0].set_ylabel('Log Likelihood')
ylabels = [f'{int(y)}' for y in ax[2][0].get_yticks()]
ax[2][0].set_yticklabels(ylabels)
ax[2][0].get_legend().remove()

sns.histplot(x='n_components_nonzero',
              data=gdn_models.sort_values('n_components'), 
              ax=ax[0][1])
ax[0][1].set_title(f'Destrieux, Gordon')
ax[0][1].set_xlabel('Non-Empty Components')
ax[0][1].set_ylabel('Score (test)')
sns.scatterplot(x='n_components',
              y='test_score',
              hue='covariance_type',
                palette='Pastel1',
              data=gdn_models.sort_values('n_components'), 
              ax=ax[1][1])
ax[1][1].set_title(f'Number of Components')
ax[1][1].set_xlabel('Components')
ax[1][1].set_ylabel('Score (test)')
ylabels = [f'{int(y)}' for y in ax[1][1].get_yticks()]
ax[1][1].set_yticklabels(ylabels)
ax[1][1].legend(bbox_to_anchor=(1.04,1), loc="upper left")
sns.scatterplot(x='weight_concentration_prior',
              y='test_score',
              hue='weight_concentration_prior_type',
              data=gdn_models, 
                palette='Pastel2',
              ax=ax[2][1])
ax[2][1].set_title(f'Weight Concentration Prior')
ax[2][1].set_xlabel('Prior')
ax[2][1].set_ylabel('Score (test)')
ylabels = [f'{int(y)}' for y in ax[2][1].get_yticks()]
ax[2][1].set_yticklabels(ylabels)
ax[2][1].legend(bbox_to_anchor=(1.04,1), loc="upper left")

fig.savefig(join(PROJ_DIR, FIGS_DIR, 'bgmm_best-models.png'),
            dpi=500, bbox_inches='tight')