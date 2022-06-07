#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

from os.path import join

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data.csv"), index_col=0, header=0)
df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
no_2yfu = df[df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
df = df.drop(no_2yfu, axis=0)

qc_vars = ["imgincl_dmri_include",
           "imgincl_rsfmri_include",
           "rsfmri_c_ngd_meanmotion",
           "rsfmri_c_ngd_ntpoints",
           "imgincl_t1w_include",
           "mrif_score"]

keep = df[df['mrif_score.baseline_year_1_arm_1'].between(1,2)].index
df = df.loc[keep]

# modality-specific filtering via masks
rsfmri_mask1 = df['imgincl_rsfmri_include.baseline_year_1_arm_1'] == 0
rsfmri_mask2 = df['rsfmri_var_ntpoints.baseline_year_1_arm_1'] <= 750.
rsfmri_mask3 = df['imgincl_rsfmri_include.2_year_follow_up_y_arm_1'] == 0
rsfmri_mask4 = df['rsfmri_var_ntpoints.2_year_follow_up_y_arm_1'] <= 750.
rsfmri_mask = rsfmri_mask1 * rsfmri_mask2 * rsfmri_mask3 * rsfmri_mask4

smri_mask1 = df['imgincl_t1w_include.baseline_year_1_arm_1'] == 0
smri_mask2 = df['imgincl_t1w_include.2_year_follow_up_y_arm_1'] == 0
smri_mask = smri_mask1 * smri_mask2

dmri_mask1 = df['imgincl_dmri_include.baseline_year_1_arm_1'] == 0
dmri_mask2 = df['imgincl_dmri_include.2_year_follow_up_y_arm_1'] == 0
dmri_mask = dmri_mask1 * dmri_mask2

smri_cols = list(df.filter(regex='smri.').columns) + list(df.filter(regex='mrisdp.').columns)
rsfmri_cols = df.filter(regex='rsfmri.').columns
dmri_cols = df.filter(regex='dmri').columns
other_cols = set(df.columns) - set(smri_cols) - set(rsfmri_cols) - set(dmri_cols)

rsfmri_quality = df[rsfmri_cols].mask(rsfmri_mask)
smri_quality = df[smri_cols].mask(smri_mask)
dmri_quality = df[dmri_cols].mask(dmri_mask)
other = df[other_cols]

# after filtering out radiological abnormalities with mrif_score
# apply modality-specific filters

# up first: rsfmri
quality_df = pd.concat([other, rsfmri_quality, smri_quality, dmri_quality], axis=1)

quality_df.to_csv(join(PROJ_DIR, DATA_DIR, "data_qcd.csv"))


