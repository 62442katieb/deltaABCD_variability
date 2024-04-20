#!/usr/bin/env python
# coding: utf-8
import json

import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
from scipy.stats import pointbiserialr

#from sklearn.ensemble import IsolationForest

sns.set_context('paper')

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle("/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/data/data.pkl")
all_ppts = df.index

df['interview_date.2_year_follow_up_y_arm_1'] = pd.to_datetime(df['interview_date.2_year_follow_up_y_arm_1'])
before_covid = df[df['interview_date.2_year_follow_up_y_arm_1'] < '2020-3-1'].index
after_covid = list(set(all_ppts) - set(before_covid))

demographics = ["demo_prnt_ed_v2.baseline_year_1_arm_1",
                "demo_comb_income_v2.baseline_year_1_arm_1",
                "race_ethnicity.baseline_year_1_arm_1",
                "sex.baseline_year_1_arm_1", 
                "mri_info_manufacturer.baseline_year_1_arm_1"
               ]


demo_df = df[demographics]
#df = None

small_df = df.loc[all_ppts][demographics]
small_df = small_df.replace({'F': 1, 'M': 0})
small_df = small_df.replace({999: np.nan})

qcd = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_covar_qcd.pkl'))
# pull out one variable per brain measure

# pull out one variable per brain measure
smri_base_var = qcd.filter(regex='smri_thick_cdk.*baseline_year_1_arm_1', axis=1).columns[0]
smri_2yfu_var = qcd.filter(regex='smri_thick_cdk.*2_year.*', axis=1).columns[0]

dmri_base_var = qcd.filter(regex='dmri_rsirndgm_cdk.*baseline_year_1_arm_1', axis=1).columns[0]
dmri_2yfu_var = qcd.filter(regex='dmri_rsirndgm_cdk.*2_year.*', axis=1).columns[0]

fmri_base_var = qcd.filter(regex='rsfmri_var_cdk.*baseline_year_1_arm_1', axis=1).columns[0]
fmri_2yfu_var = qcd.filter(regex='rsfmri_var_cdk.*2_year.*', axis=1).columns[0]

small_brain = qcd[[smri_base_var, smri_2yfu_var, dmri_base_var, dmri_2yfu_var, fmri_base_var, fmri_2yfu_var]]
mini_df = pd.concat([demo_df, small_brain], axis=1)

small_df = small_df.replace({999: np.nan})
mini_df = mini_df.replace({999: np.nan})


columns = [
    'Caregiver Education',
       'Household Income',
       'Race/Ethnicity', 
       'Sex',
       'MRI Manufacturer',
       'sMRI, age 9-10 y',
       'sMRI, age 11-13 y',
       'dMRI, age 9-10 y',
       'dMRI, age 11-13 y',
       'rsfMRI, age 9-10 y',
       'rsfMRI, age 11-13 y',]

mini_df.columns = columns

# plot null by variable by case
fig, ax = plt.subplots()
matrix = msno.matrix(mini_df, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missing_matrix.png'), dpi=600, bbox_inches="tight")

# nullity per variable
fig, ax = plt.subplots()
nulls = msno.bar(mini_df, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missing_bars.png'), dpi=600, bbox_inches="tight")

# nullity correlation
fig, ax = plt.subplots()
heatmap = msno.heatmap(mini_df, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missing_heatmap.png'), dpi=600, bbox_inches="tight")

# group variables by nullity correlation
fig, ax = plt.subplots()
dendro = msno.dendrogram(mini_df, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missing_dendrogram.png'), dpi=600, bbox_inches="tight")

qc_vars = list(qcd.columns[-16:])


smri_base = qcd.filter(regex='smri_thick_cdk.*baseline_year_1_arm_1', axis=1)
smri_2yfu = qcd.filter(regex='smri_thick_cdk.*2_year.*', axis=1)

dmri_base = qcd.filter(regex='dmri_rsirndgm_cdk.*baseline_year_1_arm_1', axis=1)
dmri_2yfu = qcd.filter(regex='dmri_rsirndgm_cdk.*2_year.*', axis=1)

fmri_base = qcd.filter(regex='rsfmri_var_cdk.*baseline_year_1_arm_1', axis=1)
fmri_2yfu = qcd.filter(regex='rsfmri_var_cdk.*2_year.*', axis=1)


brain_means = pd.concat(
    [
        smri_base.mean(axis=1),
        smri_2yfu.mean(axis=1),
        dmri_base.mean(axis=1),
        dmri_2yfu.mean(axis=1),
        fmri_base.mean(axis=1),
        fmri_2yfu.mean(axis=1)
    ],
    axis=1
)


brain_means.columns = ['smri_base', 'smri_2yfu', 'dmri_base', 'dmri_2yfu', 'fmri_base', 'fmri_2yfu']
brain = pd.concat([qcd[qc_vars], brain_means], axis=1)
small = pd.concat([small_df, brain], axis=1)


numerical_vars = brain.columns


categorical_vars = [
    'demo_prnt_ed_v2.baseline_year_1_arm_1',
       'demo_comb_income_v2.baseline_year_1_arm_1',
       'race_ethnicity.baseline_year_1_arm_1', 'sex.baseline_year_1_arm_1',
       'mri_info_manufacturer.baseline_year_1_arm_1',

       
]

small = pd.concat([small, df['site_id_l.baseline_year_1_arm_1']], axis=1)

categorical_vars.append('site_id_l.baseline_year_1_arm_1')

# need to combine puberty by 
# replacing nans with 0s
# adding m + f
# replacing 0s with nans
df['pds_p_ss_category_2.baseline_year_1_arm_1'] = df['pds_p_ss_female_category_2.baseline_year_1_arm_1'].fillna(0) + df['pds_p_ss_male_category_2.baseline_year_1_arm_1'].fillna(0)
df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] = df['pds_p_ss_female_category_2.2_year_follow_up_y_arm_1'].fillna(0) + df['pds_p_ss_male_category_2.2_year_follow_up_y_arm_1'].fillna(0)

small = pd.concat(
    [
        small,
        df['pds_p_ss_category_2.baseline_year_1_arm_1'].replace({0: np.nan}),
        df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'].replace({0: np.nan})
     ],
     axis=1
)

all_dumbs = {}
for variable in categorical_vars:
    dumbies = pd.get_dummies(df[variable], prefix=variable, dummy_na=True)
    all_dumbs[variable] = list(dumbies.columns)
    dumbies[f'{variable}_nan'].replace({1:np.nan}, inplace=True)
    small = pd.concat([small, dumbies], axis=1)

na_indicators = list(numerical_vars) + [item for sublist in all_dumbs.values() for item in sublist if 'nan' in item]
model_vars = list(numerical_vars)
model_vars += [item for sublist in all_dumbs.values() for item in sublist]

response_indicator = small[na_indicators].isnull()
missingness = response_indicator.sum(axis=0) / len(df.index)
missingness.to_csv('output/proportion_missing.csv')

stats = ['r', 'p']
columns = pd.MultiIndex.from_product([model_vars, stats])
missingness_corrs = pd.DataFrame(index=model_vars, columns=columns)
for variable in na_indicators:
    for variable2 in model_vars:
        if variable != variable2:
            no_nans = small[variable2].dropna().index
            #print(variable2, len(no_nans))
            r,p = pointbiserialr(response_indicator.loc[no_nans][variable], small.loc[no_nans][variable2])
            missingness_corrs.at[variable, (variable2, 'r')] = r
            missingness_corrs.at[variable, (variable2, 'p')] = p


missingness_corrs.to_csv("output/missingness_correlations.csv")


patterns = response_indicator[response_indicator.columns].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)


pattern_counts = patterns.value_counts()
pattern_labels = []
for i in pattern_counts.index:
    # turn concatenated string back into boolean list
    pattern = [eval(x) for x in i.split(',')]
    #print(pattern)
    # yoink only the column names where pattern == True
    # these are the columns with missingness in the pattern
    the_missing_ones = response_indicator.columns[pattern]
    pattern_labels.append(', '.join(the_missing_ones))
fig, ax = plt.subplots(figsize=(20,10))
pattern_counts.plot.bar(ax=ax)
ax.set_xticklabels(range(0,len(pattern_labels)))

fig.savefig('figures/missingness_patterns.png', dpi=400, bbox_inches="tight")

mapping = {}
for i in range(0, len(pattern_labels)):
    mapping[int(i)] = (str(pattern_labels[i]), int(pattern_counts.iloc[i]))
#print(mapping)
with open('output/missingness_patterns.json', 'w') as f:
    # write the dictionary to the file in JSON format
    json.dump(mapping, f)


fig = msno.matrix(small[na_indicators])
fig_copy = fig.get_figure()
fig_copy.savefig('figures/missingno_matrix-indicators.png', dpi=400, bbox_inches="tight")