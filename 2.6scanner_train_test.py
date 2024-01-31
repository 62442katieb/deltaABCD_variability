#!/usr/bin/env python
# coding: utf-8

import enlighten
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join#, exists
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

#from nilearn import plotting

from scipy.stats import spearmanr, pointbiserialr
from utils import jili_sidak_mc, residualize
#from sklearn.linear_model import LinearRegression

from datetime import datetime

today = datetime.today().strftime('%Y-%m-%d')


def jili_sidak_mc(data, alpha):
    '''
    Accepts a dataframe (data, samples x features) and a type-i error rate (alpha, float), 
    then adjusts for the number of effective comparisons between variables
    in the dataframe based on the eigenvalues of their pairwise correlations.
    '''
    import math
    import numpy as np

    mc_corrmat = data.corr()
    mc_corrmat.fillna(0, inplace=True)
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print('\nFor {0} vars, number of effective comparisons: {1}\n'.format(mc_corrmat.shape[0], M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff


sns.set(style='white', context='talk', font_scale=1.1)
plt.rcParams["font.family"] = "Avenir Next Condensed"


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))
#imputed_dcg = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data.csv")), 
#                          index_col='subjectkey', 
#                          header=0)


scanners = [
    "GE MEDICAL SYSTEMS", 
    "Philips Medical Systems", 
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
    scanner_ppts[scanner] = df[df['mri_info_manufacturer.baseline_year_1_arm_1'] == scanner].index

imaging_cols = list(df.filter(regex='.*mri.*change_score').columns)



numerical = ["interview_age","rsfmri_var_meanmotion",
        "rsfmri_var_subthreshnvols",
        "rsfmri_var_subtcignvols",
        "rsfmri_var_ntpoints","nihtbx_picvocab_uncorrected",
        "nihtbx_flanker_uncorrected",
        "nihtbx_list_uncorrected",
        "nihtbx_cardsort_uncorrected",
        "nihtbx_pattern_uncorrected",
        "nihtbx_picture_uncorrected",
        "nihtbx_reading_uncorrected", "pds_p_ss_female_category_2", 
        "pds_p_ss_male_category_2","cbcl_scr_syn_anxdep_r", 
        "cbcl_scr_syn_withdep_r", 
        "cbcl_scr_syn_somatic_r", 
        "cbcl_scr_syn_social_r", 
        "cbcl_scr_syn_thought_r", 
        "cbcl_scr_syn_attention_r", 
        "cbcl_scr_syn_rulebreak_r", 
        "cbcl_scr_syn_aggressive_r", 
        "cbcl_scr_syn_internal_r", 
        "cbcl_scr_syn_external_r", 
        "cbcl_scr_syn_totprob_r", ]

non_numerical = [#"interview_date", 
        "sex", "site_id_l", "demo_prnt_ethn_v2",
        "demo_prnt_marital_v2",
        "demo_prnt_ed_v2",
        "demo_comb_income_v2"]

vars_of_interest = numerical + non_numerical

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

cols = []
for var in numerical:
    columns = df.filter(regex=f'{var}.*', axis=1).columns
    if len(columns) == 0:
        pass
    else:
        for column in columns:
            cols.append(column)


num_df = df[cols]

cols = []
for var in non_numerical:
    columns = df.filter(regex=f'{var}.*', axis=1).columns
    if len(columns) == 0:
        pass
    else:
        for column in columns:
            cols.append(column)

#print(len(cols))
#print(cols)
non_df = df[cols]

alpha,_ = jili_sidak_mc(pd.concat([num_df, non_df], axis=1), 0.05)

# train on Siemens, test on GE?!
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

ppts = scanner_ppts["SIEMENS"]
siemens_data = df.loc[ppts][imaging_cols].dropna()
siemens_subj = siemens_data.index.to_list()
siemens_data = siemens_data.loc[siemens_subj]

siemens_resid = pd.DataFrame()
for modality in noisy_modalities.keys():
    cov_df = pd.DataFrame()
    mini_dset = siemens_data.filter(like=modality)
    subj = siemens_data.index
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
    siemens_resid = pd.concat([siemens_resid, temp2], axis=1)
siemens_data = pd.DataFrame(
    Normalizer().fit_transform(siemens_resid), 
    index=siemens_resid.index, 
    columns=siemens_resid.columns)

ppts = scanner_ppts["GE MEDICAL SYSTEMS"]
ge_data = df.loc[ppts][imaging_cols].dropna()
ge_subj = ge_data.index.to_list()
ge_data = ge_data.loc[ge_subj]

ge_resid = pd.DataFrame()
for modality in noisy_modalities.keys():
    cov_df = pd.DataFrame()
    mini_dset = ge_data.filter(like=modality)
    subj = ge_data.index
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
    ge_resid = pd.concat([ge_resid, temp2], axis=1)
ge_data = pd.DataFrame(
    Normalizer().fit_transform(ge_resid), 
    index=ge_resid.index, 
    columns=ge_resid.columns)

ppts = scanner_ppts["Philips Medical Systems"]
philips_data = df.loc[ppts][imaging_cols].dropna()
philips_subj = philips_data.index.to_list()
philips_data = philips_data.loc[philips_subj]

philips_resid = pd.DataFrame()
for modality in noisy_modalities.keys():
    cov_df = pd.DataFrame()
    mini_dset = philips_data.filter(like=modality)
    subj = philips_data.index
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
    philips_resid = pd.concat([philips_resid, temp2], axis=1)
philips_data = pd.DataFrame(
    Normalizer().fit_transform(philips_resid), 
    index=philips_resid.index, 
    columns=philips_resid.columns)

metrics = ['silhouette', 'davies-bouldin', 'calinksi-harabasz', 'inertia', 'score']
cols = pd.MultiIndex.from_product([scanners, metrics])
scores = pd.DataFrame(columns=cols, index=scanner, dtype=float)
### update with "best" params
estimator = KMeans(
                n_init=100,
                max_iter=1000,
                n_clusters=2,
                random_state=1
            )

siemens = estimator.fit(siemens_data)
siemens_labels = siemens.predict(siemens_data)

scores.at["SIEMENS", ("SIEMENS", 'score')] = siemens.score(siemens_data)
scores.at["SIEMENS", ("SIEMENS", 'inertia')] = siemens.inertia_
scores.at["SIEMENS", ("SIEMENS", 'silhouette')] = silhouette_score(siemens_data, siemens_labels)
scores.at["SIEMENS", ("SIEMENS", 'davies-bouldin')] = davies_bouldin_score(siemens_data, siemens_labels)
scores.at["SIEMENS", ("SIEMENS", 'calinksi-harabasz')] = calinski_harabasz_score(siemens_data, siemens_labels)

siemens_labels = pd.DataFrame(siemens_labels, 
                        index=siemens_subj,
                        columns=list(range(0,estimator.n_clusters)))

ge_labels = siemens.predict(ge_data)
ge_labels = pd.DataFrame(ge_labels, 
                        index=ge_subj,
                        columns=list(range(0,estimator.n_clusters)))

scores.at["SIEMENS", ("GE MEDICAL SYSTEMS", 'score')] = siemens.score(ge_data)
scores.at["SIEMENS", ("GE MEDICAL SYSTEMS", 'silhouette')] = silhouette_score(ge_data, ge_labels)
scores.at["SIEMENS", ("GE MEDICAL SYSTEMS", 'davies-bouldin')] = davies_bouldin_score(ge_data, ge_labels)
scores.at["SIEMENS", ("GE MEDICAL SYSTEMS", 'calinksi-harabasz')] = calinski_harabasz_score(ge_data, ge_labels)

philips_labels = siemens.predict(philips_data)
philips_labels = pd.DataFrame(philips_labels, 
                                index=philips_subj, 
                                columns=list(range(0,estimator.n_components)))

scores.at["SIEMENS", ("Philips Medical Systems", 'score')] = siemens.score(philips_data)
scores.at["SIEMENS", ("Philips Medical Systems", 'silhouette')] = silhouette_score(philips_data, philips_labels)
scores.at["SIEMENS", ("Philips Medical Systems", 'davies-bouldin')] = davies_bouldin_score(philips_data, philips_labels)
scores.at["SIEMENS", ("Philips Medical Systems", 'calinksi-harabasz')] = calinski_harabasz_score(philips_data, philips_labels)


pd.concat(
    [siemens_labels, 
     ge_labels, 
     philips_labels],
     axis=0
     ).to_csv(join(PROJ_DIR, OUTP_DIR, f'all_labels-siemens-{today}.csv'))


siemens_num = num_df.loc[siemens_subj]
ge_num = num_df.loc[ge_subj]
philips_num = num_df.loc[philips_subj]

index = pd.MultiIndex.from_product(
    [list(range(0,estimator.n_components)), scanners, ['r', 'p', 'sig']],
    names=['Cluster', 'Scanner', 'Stat']
)

numerical_corrs = pd.DataFrame(
    index=num_df.columns, 
    columns=index,
    dtype=float
)
for cluster in list(range(0,estimator.n_clusters)):
    for column in num_df.columns:
        siemens_dumb = pd.get_dummies(siemens_labels)
        for dumb in siemens_dumb.columns:
            siemens_corr,p = pointbiserialr(pd.concat(
                [siemens_dumb[dumb], 
                siemens_num[column]
                ], 
                axis=1).dropna())
            numerical_corrs.at[column, (cluster, 'Siemens', 'r')] = siemens_corr
            numerical_corrs.at[column, (cluster, 'Siemens', 'p')] = p
            if p < alpha:
                numerical_corrs.at[column, (cluster, 'Siemens', 'sig')] = '*'
            else:
                numerical_corrs.at[column, (cluster, 'Siemens', 'sig')] = ''
        ge_dumb = pd.get_dummies(ge_labels)
        for dumb in ge_dumb:
            ge_corr,p = pointbiserialr(pd.concat(
                [ge_dumb[dumb],
                 ge_num[column]
                ], 
                axis=1).dropna())
        numerical_corrs.at[column, (cluster, 'GE', 'r')] = ge_corr
        numerical_corrs.at[column, (cluster, 'GE', 'p')] = p
        if p < alpha:
            numerical_corrs.at[column, (cluster, 'GE', 'sig')] = '*'
        else:
            numerical_corrs.at[column, (cluster, 'GE', 'sig')] = ''
            
        philips_corr,p = spearmanr(pd.concat(
            [philips_responsibilities[cluster], 
             philips_num[column]
            ], 
            axis=1).dropna())
        numerical_corrs.at[column, (cluster, 'Philips', 'r')] = philips_corr
        numerical_corrs.at[column, (cluster, 'Philips', 'p')] = p
        if p < alpha:
            numerical_corrs.at[column, (cluster, 'Philips', 'sig')] = '*'
        else:
            numerical_corrs.at[column, (cluster, 'Philips', 'sig')] = '' 


long_names = {
    'picvocab': 'Picture Vocabulary',
    'flanker': 'Inhibitory Control',
    'cardsort': 'Card Sorting',
    'list': 'List Sorting',
    'pattern': 'Pattern Comparison',
    'picture': 'Picture Sequence',
    'reading': 'Reading Recognition',
    'totprob': 'Total Problems',
    'anxdep': 'Anxious/Depressed',
    'withdep': 'Withdrawn',
    'somatic': 'Somatic Complaints',
    'thought': 'Thought Problems',
    'rulebreak': 'Rule Breaking',
    'aggressive': 'Aggressive',
    'internal': 'Internalizing',
    'external': 'Externalizing',
    'age': 'Age',
    'female': 'Female Puberty',
    'male': 'Male Puberty',
    'social': 'Social Problems',
    'attention': 'Attention Problems'
    }


for i in numerical_corrs.index:
    vars_ = i.split('.')
    timepoint = vars_[-1]
    if '2' in timepoint:
        numerical_corrs.at[i, 'timepoint'] = 'at 11-13 years'
    elif 'change' in timepoint:
        numerical_corrs.at[i, 'timepoint'] = 'change 9-13 years'
    else:
        numerical_corrs.at[i, 'timepoint'] = 'at 9-10 years'
    if 'cbcl' in i:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[3]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
    elif 'pds' in i:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[3]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
    else:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[1]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
numerical_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'siemens_model_numerical_corrs.csv'))


sig_annot = numerical_corrs[numerical_corrs.columns[numerical_corrs.columns.get_level_values(2) == 'sig']]

corrs = numerical_corrs[numerical_corrs.columns[numerical_corrs.columns.get_level_values(2) == 'r']].sort_values((keep[0],'Siemens','r'))
corrs = corrs.droplevel(2, axis=1)

nine_ten = numerical_corrs[numerical_corrs['timepoint'] == 'at 9-10 years'].sort_values(
    (keep[0],'Siemens','r'), 
    ascending=False).index
eleven_twelve = numerical_corrs[numerical_corrs['timepoint'] == 'at 11-13 years'].sort_values(
    (keep[0],'Siemens','r'), 
    ascending=False).index
change = numerical_corrs[numerical_corrs['timepoint'] == 'change 9-13 years'].sort_values(
    (keep[0],'Siemens','r'), 
    ascending=False).index

row_order = list(nine_ten) + list(eleven_twelve) + list(change)


numerical_corrs.loc[row_order]['measure'].values

fig,ax = plt.subplots(figsize=(5,20))
sns.heatmap(
    corrs.loc[row_order],
    cmap='seismic',
    center=0,
    ax=ax,
    annot=sig_annot.loc[row_order],
    fmt=''
)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'siemens_vbgmm_corrs.png'), bbox_inches='tight', dpi=400)

ge = estimator.fit(ge_data)
ge_responsibilities = ge.predict_proba(ge_data)
ge_score = np.round(ge.score(ge_data), 3)

ge_weights = pd.DataFrame(ge.weights_, 
                        index=list(range(0,estimator.n_components)))
ge_means = pd.DataFrame(ge.means_.T, 
                        index=ge.feature_names_in_,
                        columns=list(range(0,estimator.n_components)))

ge_responsibilities = pd.DataFrame(ge_responsibilities, 
                                index=ge_subj, 
                                columns=list(range(0,estimator.n_components)))
keep = list(ge_responsibilities.columns[ge_responsibilities.sum() > 0])
ge_responsibilities = ge_responsibilities[keep]

ge_responsibilities.to_csv(join(PROJ_DIR, OUTP_DIR, f'responsibilities-ge-{today}.csv'))
ge_weights.to_csv(join(PROJ_DIR, OUTP_DIR, f'weights-ge-{today}.csv'))
ge_means.to_csv(join(PROJ_DIR, OUTP_DIR, f'brain_means-ge-{today}.csv'))

siemens_score = ge.score_samples(siemens_data)
siemens_predict = ge.predict(siemens_data)

siemens_responsibilities = ge.predict_proba(siemens_data)
siemens_responsibilities = pd.DataFrame(siemens_responsibilities, 
                                index=siemens_subj, 
                                columns=list(range(0,estimator.n_components)))
siemens_responsibilities = siemens_responsibilities[keep]
siemens_score = ge.score(siemens_data)

philips_predicts = ge.predict(philips_data)
philips_responsibilities = ge.predict_proba(philips_data)
philips_responsibilities = pd.DataFrame(philips_responsibilities, 
                                index=philips_subj, 
                                columns=list(range(0,estimator.n_components)))
philips_responsibilities = philips_responsibilities[keep]
philips_score = ge.score(philips_data)

pd.concat(
    [siemens_responsibilities, 
     ge_responsibilities, 
     philips_responsibilities],
     axis=0
     ).to_csv(join(PROJ_DIR, OUTP_DIR, f'all_responsibilities-ge-{today}.csv'))

print(
    'GE (train):\t', np.mean(ge_score),
    '\nSiemens (test):\t', np.mean(siemens_score),
    '\nPhilips (test):\t', np.mean(philips_score)
)
pd.Series(
    [np.mean(philips_score), 
     np.mean(siemens_score), 
     np.mean(ge_score)], 
    index=['Philips', 'Siemens', 'GE']
).to_csv(join(PROJ_DIR, OUTP_DIR, 'ge_model_scores.csv'))


index = pd.MultiIndex.from_product(
    [keep, ['GE', 'Siemens', 'Philips'], ['r', 'p', 'sig']],
    names=['Cluster', 'Scanner', 'Stat']
)

numerical_corrs = pd.DataFrame(
    index=num_df.columns, 
    columns=index,
    dtype=float
)
for cluster in keep:
    for column in num_df.columns:
        siemens_corr,p = spearmanr(pd.concat(
            [siemens_responsibilities[cluster], 
             siemens_num[column]
            ], 
            axis=1).dropna())
        numerical_corrs.at[column, (cluster, 'Siemens', 'r')] = siemens_corr
        numerical_corrs.at[column, (cluster, 'Siemens', 'p')] = p
        if p < alpha:
            numerical_corrs.at[column, (cluster, 'Siemens', 'sig')] = '*'
        else:
            numerical_corrs.at[column, (cluster, 'Siemens', 'sig')] = ''
        
        ge_corr,p = spearmanr(pd.concat(
            [ge_responsibilities[cluster], 
             ge_num[column]
            ], 
            axis=1).dropna())
        numerical_corrs.at[column, (cluster, 'GE', 'r')] = ge_corr
        numerical_corrs.at[column, (cluster, 'GE', 'p')] = p
        if p < alpha:
            numerical_corrs.at[column, (cluster, 'GE', 'sig')] = '*'
        else:
            numerical_corrs.at[column, (cluster, 'GE', 'sig')] = ''
            
        philips_corr,p = spearmanr(pd.concat(
            [philips_responsibilities[cluster], 
             philips_num[column]
            ], 
            axis=1).dropna())
        numerical_corrs.at[column, (cluster, 'Philips', 'r')] = philips_corr
        numerical_corrs.at[column, (cluster, 'Philips', 'p')] = p
        if p < alpha:
            numerical_corrs.at[column, (cluster, 'Philips', 'sig')] = '*'
        else:
            numerical_corrs.at[column, (cluster, 'Philips', 'sig')] = '' 

for i in numerical_corrs.index:
    vars_ = i.split('.')
    timepoint = vars_[-1]
    if '2' in timepoint:
        numerical_corrs.at[i, 'timepoint'] = 'at 11-13 years'
    elif 'change' in timepoint:
        numerical_corrs.at[i, 'timepoint'] = 'change 9-13 years'
    else:
        numerical_corrs.at[i, 'timepoint'] = 'at 9-10 years'
    if 'cbcl' in i:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[3]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
    elif 'pds' in i:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[3]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
    else:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[1]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
numerical_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'ge_model_numerical_corrs.csv'))


sig_annot = numerical_corrs[numerical_corrs.columns[numerical_corrs.columns.get_level_values(2) == 'sig']]

corrs = numerical_corrs[numerical_corrs.columns[numerical_corrs.columns.get_level_values(2) == 'r']].sort_values((keep[0], 'GE', 'r'))
corrs = corrs.droplevel(2, axis=1)

w = len(keep) + 2
fig,ax = plt.subplots(figsize=(w,20))
sns.heatmap(
    corrs.loc[row_order],
    cmap='seismic',
    center=0,
    ax=ax,
    annot=sig_annot.loc[row_order],
    fmt=''
)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'ge_vbgmm_corrs.png'), bbox_inches='tight', dpi=400)


philips = estimator.fit(philips_data)
philips_responsibilities = philips.predict_proba(philips_data)
philips_score = np.round(philips.score(philips_data), 3)
philips_weights = pd.DataFrame(philips.weights_, 
                        index=list(range(0,estimator.n_components)))
philips_means = pd.DataFrame(philips.means_.T, 
                        index=philips.feature_names_in_,
                        columns=list(range(0,estimator.n_components)))

philips_responsibilities = pd.DataFrame(philips_responsibilities, 
                                index=philips_subj, 
                                columns=list(range(0,estimator.n_components)))
keep = list(philips_responsibilities.columns[philips_responsibilities.sum() > 0])
philips_responsibilities = philips_responsibilities[keep]

philips_responsibilities.to_csv(join(PROJ_DIR, OUTP_DIR, f'responsibilities-philips-{today}.csv'))
philips_weights.to_csv(join(PROJ_DIR, OUTP_DIR, f'weights-philips-{today}.csv'))
philips_means.to_csv(join(PROJ_DIR, OUTP_DIR, f'brain_means-philips-{today}.csv'))

siemens_score = philips.score_samples(siemens_data)
siemens_predict = philips.predict(siemens_data)

siemens_responsibilities = philips.predict_proba(siemens_data)
siemens_responsibilities = pd.DataFrame(siemens_responsibilities, 
                                index=siemens_subj, 
                                columns=list(range(0,estimator.n_components)))
siemens_responsibilities = siemens_responsibilities[keep]
siemens_score = philips.score(siemens_data)

ge_predicts = philips.predict(ge_data)
ge_responsibilities = philips.predict_proba(ge_data)
ge_responsibilities = pd.DataFrame(ge_responsibilities, 
                                index=ge_subj, 
                                columns=list(range(0,estimator.n_components)))
ge_responsibilities = ge_responsibilities[keep]
ge_score = philips.score(ge_data)

pd.concat(
    [siemens_responsibilities, 
     ge_responsibilities, 
     philips_responsibilities],
     axis=0
     ).to_csv(join(PROJ_DIR, OUTP_DIR, f'all_responsibilities-philips-{today}.csv'))


print(
    'Philips (train):\t', np.mean(philips_score),
    '\nSiemens (test):\t\t', np.mean(siemens_score),
    '\nGE (test):\t\t', np.mean(ge_score)
)
pd.Series(
    [np.mean(philips_score), 
     np.mean(siemens_score), 
     np.mean(ge_score)], 
    index=['Philips', 'Siemens', 'GE']
).to_csv(join(PROJ_DIR, OUTP_DIR, 'philips_model_scores.csv'))


index = pd.MultiIndex.from_product(
    [keep, ['Philips', 'GE', 'Siemens'], ['r', 'p', 'sig']],
    names=['Cluster', 'Scanner', 'Stat']
)

numerical_corrs = pd.DataFrame(
    index=num_df.columns, 
    columns=index,
    dtype=float
)
for cluster in keep:
    for column in num_df.columns:
        siemens_corr,p = spearmanr(pd.concat(
            [siemens_responsibilities[cluster], 
             siemens_num[column]
            ], 
            axis=1).dropna())
        numerical_corrs.at[column, (cluster, 'Siemens', 'r')] = siemens_corr
        numerical_corrs.at[column, (cluster, 'Siemens', 'p')] = p
        if p < alpha:
            numerical_corrs.at[column, (cluster, 'Siemens', 'sig')] = '*'
        else:
            numerical_corrs.at[column, (cluster, 'Siemens', 'sig')] = ''
        
        ge_corr,p = spearmanr(pd.concat(
            [ge_responsibilities[cluster], 
             ge_num[column]
            ], 
            axis=1).dropna())
        numerical_corrs.at[column, (cluster, 'GE', 'r')] = ge_corr
        numerical_corrs.at[column, (cluster, 'GE', 'p')] = p
        if p < alpha:
            numerical_corrs.at[column, (cluster, 'GE', 'sig')] = '*'
        else:
            numerical_corrs.at[column, (cluster, 'GE', 'sig')] = ''
            
        philips_corr,p = spearmanr(pd.concat(
            [philips_responsibilities[cluster], 
             philips_num[column]
            ], 
            axis=1).dropna())
        numerical_corrs.at[column, (cluster, 'Philips', 'r')] = philips_corr
        numerical_corrs.at[column, (cluster, 'Philips', 'p')] = p
        if p < alpha:
            numerical_corrs.at[column, (cluster, 'Philips', 'sig')] = '*'
        else:
            numerical_corrs.at[column, (cluster, 'Philips', 'sig')] = '' 


for i in numerical_corrs.index:
    vars_ = i.split('.')
    timepoint = vars_[-1]
    if '2' in timepoint:
        numerical_corrs.at[i, 'timepoint'] = 'at 11-13 years'
    elif 'change' in timepoint:
        numerical_corrs.at[i, 'timepoint'] = 'change 9-13 years'
    else:
        numerical_corrs.at[i, 'timepoint'] = 'at 9-10 years'
    if 'cbcl' in i:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[3]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
    elif 'pds' in i:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[3]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
    else:
        numerical_corrs.at[i, 'instrument'] = vars_[0].split('_')[0]
        measure = vars_[0].split('_')[1]
        numerical_corrs.at[i, 'measure'] = long_names[measure]
numerical_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'philips_model_numerical_corrs.csv'))


sig_annot = numerical_corrs[numerical_corrs.columns[numerical_corrs.columns.get_level_values(2) == 'sig']]

corrs = numerical_corrs[numerical_corrs.columns[numerical_corrs.columns.get_level_values(2) == 'r']].sort_values((keep[0], 'Philips', 'r'))
corrs = corrs.droplevel(2, axis=1)


w = len(keep) + 2
fig,ax = plt.subplots(figsize=(w,20))
sns.heatmap(
    corrs.loc[row_order],
    cmap='seismic',
    center=0,
    ax=ax,
    annot=sig_annot.loc[row_order],
    fmt=''
)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'philips_vbgmm_corrs.png'), bbox_inches='tight', dpi=400)