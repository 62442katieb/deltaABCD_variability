#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pingouin as pg

from os.path import join
import warnings

warnings.filterwarnings("ignore")

def unvectorize_r(df, networks):
    corrmat = np.zeros((len(networks), len(networks)))
    for ntwk1 in networks:
        i = networks.index(ntwk1)
        for ntwk2 in networks:
            j = networks.index(ntwk2)
            var = f'rsfmri_c_ngd_{ntwk1}_ngd_{ntwk2}'
            try:
                corrmat[i,j] = np.tanh(df[var])
            except Exception as e:
                print(e)
    return corrmat

def residualize(X, y=None, confounds=None):
    '''
    all inputs need to be arrays, not dataframes
    '''
    # residualize the outcome
    if confounds is not None:
        if y is not None:
            temp_y = np.reshape(y, (y.shape[0],))
            y = pg.linear_regression(confounds, temp_y)
            resid_y = y.residuals_

            # residualize features
            resid_X = np.zeros_like(X)
            # print(X.shape, resid_X.shape)
            for i in range(0, X.shape[1]):
                X_temp = X[:, i]
                # print(X_temp.shape)
                X_ = pg.linear_regression(confounds, X_temp)
                # print(X_.residuals_.shape)
                resid_X[:, i] = X_.residuals_.flatten()
            return resid_y, resid_X
        else:
            # residualize features
            resid_X = np.zeros_like(X)
            # print(X.shape, resid_X.shape)
            for i in range(0, X.shape[1]):
                X_temp = X[:, i]
                # print(X_temp.shape)
                X_ = pg.linear_regression(confounds, X_temp)
                # print(X_.residuals_.shape)
                resid_X[:, i] = X_.residuals_.flatten()
            return resid_X

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))

ppts = df.filter(regex="rsfmri_c_.*change_score").dropna().index

rsfc = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.0/core/imaging/mri_y_rsfmr_cor_gp_gp.csv",
    header=0,
    index_col=[0,1]
).dropna()

rs_motion = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.0/core/imaging/mri_y_qc_motion.csv",
    header=0,
    index_col=[0,1]
).dropna()
base_mot = rs_motion.swaplevel(axis=0).loc['baseline_year_1_arm_1'].filter(like='rsfmri').drop(['rsfmri_numtrs', 'rsfmri_nvols', 'rsfmri_tr'], axis=1)
y2fu_mot = rs_motion.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1'].filter(like='rsfmri').drop(['rsfmri_numtrs', 'rsfmri_nvols', 'rsfmri_tr'], axis=1)
y4fu_mot = rs_motion.swaplevel(axis=0).loc['4_year_follow_up_y_arm_1'].filter(like='rsfmri').drop(['rsfmri_numtrs', 'rsfmri_nvols', 'rsfmri_tr'], axis=1)

nones = list(rsfc.filter(regex='rsfmri_c_ngd_.*_ngd_n').columns) + list(rsfc.filter(regex='rsfmri_c_ngd_n_.*').columns)
rsfc = rsfc.drop(nones, axis=1)

tpts = [
    'baseline_year_1_arm_1',
    '2_year_follow_up_y_arm_1'
]

within_network = [i for i in rsfc.columns if i.split('_')[3] == i.split('_')[5]]

btwn = rsfc.columns

network_wise = {}

for i in [j.split('_')[3] for j in within_network]:
    network_wise[i] = [k for k in btwn if i == k.split('_')[3]]

between_network = [i for i in rsfc.columns if i.split('_')[3] != i.split('_')[5]]

conns = rsfc.columns
motion_cols = base_mot.filter(like='mean').columns

base_df = rsfc.swaplevel(axis=0).loc['baseline_year_1_arm_1']
y2fu_df = rsfc.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1']

base_df = pd.concat([base_df,base_mot[motion_cols]], axis=1).dropna()
y2fu_df = pd.concat([y2fu_df,y2fu_mot[motion_cols]], axis=1).dropna()
#y4fu_df = pd.concat([y4fu_df,y4fu_mot[motion_cols]], axis=1).dropna()

base_resid = residualize(base_df.drop(motion_cols, axis=1).values, confounds=base_df[motion_cols].values)
y2fu_resid = residualize(y2fu_df.drop(motion_cols, axis=1).values, confounds=y2fu_df[motion_cols].values)
#y4fu_resid = residualize(y4fu_df.drop(motion_cols, axis=1).values, confounds=y4fu_df[motion_cols].values)

base_resid = pd.DataFrame(base_resid, index=base_df.index, columns=base_df.drop(motion_cols, axis=1).columns)
y2fu_resid = pd.DataFrame(y2fu_resid, index=y2fu_df.index, columns=y2fu_df.drop(motion_cols, axis=1).columns)
#y4fu_resid = pd.DataFrame(y4fu_resid, index=y4fu_df.index, columns=y4fu_df.drop(motion_cols, axis=1).columns)

base_resid.to_pickle(join(PROJ_DIR, DATA_DIR, 'rsfc_sans_motion-baseline.pkl'))
y2fu_resid.to_pickle(join(PROJ_DIR, DATA_DIR, 'rsfc_sans_motion-2yearfup.pkl'))

print(base_resid.describe(), '\n', y2fu_resid.describe())