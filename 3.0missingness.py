#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
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
smri_base_var = qcd.filter(regex='smri_thick_cdk.*baseline_year_1_arm_1', axis=1).columns[0]
smri_2yfu_var = qcd.filter(regex='smri_thick_cdk.*2_year.*', axis=1).columns[0]

dmri_base_var = qcd.filter(regex='dmri_rsirndgm_cdk.*baseline_year_1_arm_1', axis=1).columns[0]
dmri_2yfu_var = qcd.filter(regex='dmri_rsirndgm_cdk.*2_year.*', axis=1).columns[0]

fmri_base_var = qcd.filter(regex='rsfmri_var_cdk.*baseline_year_1_arm_1', axis=1).columns[0]
fmri_2yfu_var = qcd.filter(regex='rsfmri_var_cdk.*2_year.*', axis=1).columns[0]

small_brain = qcd[[smri_base_var, smri_2yfu_var, dmri_base_var, dmri_2yfu_var, fmri_base_var, fmri_2yfu_var]]
mini_df = pd.concat([demo_df, small_brain], axis=1)

# plot null by variable by case
fig, ax = plt.subplots()
matrix = msno.matrix(mini_df, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missing_matrix.png'), dpi=600)

# nullity per variable
fig, ax = plt.subplots()
nulls = msno.bar(mini_df, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missing_bars.png'), dpi=600)

# nullity correlation
fig, ax = plt.subplots()
heatmap = msno.heatmap(mini_df, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missing_heatmap.png'), dpi=600)

# group variables by nullity correlation
fig, ax = plt.subplots()
dendro = msno.dendrogram(mini_df, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missing_dendrogram.png'), dpi=600)
