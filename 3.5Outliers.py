import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt

import pyreadr

from os.path import join
from scipy.stats import spearmanr, fligner, variation
from sklearn.ensemble import IsolationForest


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

# set the plotting settings so our graphs are pretty
sns.set(context='talk', style='white', palette='Set2')

thk_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.pkl'))
var_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.pkl'))
rni_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.pkl'))
rnd_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.pkl'))

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'residualized_change_scores.pkl'))
# ONLY SIEMENS #
# need to load in 3.0 vars that include head motion and brain volume

smri_raw = df.filter(regex='smri_thick_cdk.*')
smri_raw *= -1

rni_raw = df.filter(regex='dmri_rsirnigm_cdk.*')

rnd_raw = df.filter(regex='dmri_rsirndgm_cdk.*')
rnd_raw *= -1

var_raw = df.filter(regex='rsfmri_var_cdk.*')

# redo SA rank by hemisphere
thk_df['hemi'] = [i.split('_')[-1][-2:] for i in thk_df.index]
rni_df['hemi'] = [i.split('_')[-1][-2:] for i in rni_df.index]
rnd_df['hemi'] = [i.split('_')[-1][-2:] for i in rnd_df.index]
var_df['hemi'] = [i.split('_')[-1][-2:] for i in var_df.index]

left_sorted = thk_df[thk_df['hemi'] == 'lh'].sort_values('SA_avg')
right_sorted = thk_df[thk_df['hemi'] == 'rh'].sort_values('SA_avg')
thk_df = pd.concat([left_sorted, right_sorted])
#print(thk_df['SA_avg'])

left_sorted = rni_df[rni_df['hemi'] == 'lh'].sort_values('SA_avg')
right_sorted = rni_df[rni_df['hemi'] == 'rh'].sort_values('SA_avg')
rni_df = pd.concat([left_sorted, right_sorted])

left_sorted = rnd_df[rnd_df['hemi'] == 'lh'].sort_values('SA_avg')
right_sorted = rnd_df[rnd_df['hemi'] == 'rh'].sort_values('SA_avg')
rnd_df = pd.concat([left_sorted, right_sorted])
#print(rnd_df['SA_avg'])


left_sorted = var_df[var_df['hemi'] == 'lh'].sort_values('SA_avg')
right_sorted = var_df[var_df['hemi'] == 'rh'].sort_values('SA_avg')
var_df = pd.concat([left_sorted, right_sorted])

#outlier brain regions
smri_outlier_trees = IsolationForest().fit(smri_raw.dropna().T)
smri_outliers = smri_outlier_trees.decision_function(smri_raw.dropna().T)
#print(smri_outliers)


smri_outlier_trees = IsolationForest().fit(smri_raw.dropna())
smri_outliers = smri_outlier_trees.decision_function(smri_raw.dropna())

rnd_outlier_trees = IsolationForest().fit(rnd_raw.dropna())
rnd_outliers = rnd_outlier_trees.decision_function(rnd_raw.dropna())

rni_outlier_trees = IsolationForest().fit(rni_raw.dropna())
rni_outliers = rni_outlier_trees.decision_function(rni_raw.dropna())

rsfmri_outlier_trees = IsolationForest().fit(var_raw.dropna())
rsfmri_outliers = rsfmri_outlier_trees.decision_function(var_raw.dropna())

outlier_smri = pd.DataFrame(smri_outliers, index=smri_raw.dropna().index)
outlier_rni = pd.DataFrame(rni_outliers, index=rni_raw.dropna().index)
outlier_rnd = pd.DataFrame(rnd_outliers, index=rnd_raw.dropna().index)
outlier_fmri = pd.DataFrame(rsfmri_outliers, index=var_raw.dropna().index)

outlier_df = pd.concat([outlier_smri, outlier_rni, outlier_rnd, outlier_fmri], axis=1)
outlier_df.columns = ['Cortical thickness', 'RNI', 'RND', 'BOLD variance']

outliers = np.product(outlier_df.fillna(0) > 0, axis=1) == 1
outliers.to_csv(join(PROJ_DIR, OUTP_DIR, 'outlier_participants.csv'))