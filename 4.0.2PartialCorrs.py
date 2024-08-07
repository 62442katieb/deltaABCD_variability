#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pingouin as pg
import math

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
#y4fu_mot = rs_motion.swaplevel(axis=0).loc['4_year_follow_up_y_arm_1'].filter(like='rsfmri').drop(['rsfmri_numtrs', 'rsfmri_nvols', 'rsfmri_tr'], axis=1)

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
base_temp = base_df.copy()
base_temp.columns = [f'{i}-base' for i in base_df.columns]
y2fu_temp = y2fu_df.copy()
y2fu_temp.columns = [f'{i}-y2fu' for i in y2fu_df.columns]

all_the_things = pd.concat([base_temp, y2fu_temp], axis=1).dropna()
for col in motion_cols:
    all_the_things[f'{col}-avg'] = (all_the_things[f'{col}-base'] + all_the_things[f'{col}-y2fu']) / 2

corrs = ['correlation', 'semipartial_base', 'semipartial_y2fu', 'partial']
stats = ['r', 'ci95', 'p(r)']
cols = pd.MultiIndex.from_product([corrs, stats])
corr_semi = pd.DataFrame(
    index=conns,
    columns=cols
)

for conn in conns:
    pmr = pg.corr(
        all_the_things[f'{conn}-base'], 
        all_the_things[f'{conn}-y2fu'], 
        method='spearman'
    )
    corr_semi.at[conn, ('correlation', 'r')] = pmr['r'].values[0]
    corr_semi.at[conn, ('correlation', 'ci95')] = pmr['CI95%'].values[0]
    corr_semi.at[conn, ('correlation', 'p(r)')] = pmr['p-val'].values[0]
    sp0 = pg.partial_corr(
        all_the_things, 
        f'{conn}-base', 
        f'{conn}-y2fu', 
        x_covar=[f'{i}-base' for i in motion_cols],
        method='spearman'
    )
    corr_semi.at[conn, ('semipartial_base', 'r')] = sp0['r'].values[0]
    corr_semi.at[conn, ('semipartial_base', 'ci95')] = sp0['CI95%'].values[0]
    corr_semi.at[conn, ('semipartial_base', 'p(r)')] = sp0['p-val'].values[0]
    sp2 = pg.partial_corr(
        all_the_things.dropna(), 
        f'{conn}-base', 
        f'{conn}-y2fu', 
        y_covar=[f'{i}-y2fu' for i in motion_cols],
        method='spearman'
    )
    corr_semi.at[conn, ('semipartial_y2fu', 'r')] = sp2['r'].values[0]
    corr_semi.at[conn, ('semipartial_y2fu', 'ci95')] = sp2['CI95%'].values[0]
    corr_semi.at[conn, ('semipartial_y2fu', 'p(r)')] = sp2['p-val'].values[0]
    pcr = pg.partial_corr(
        all_the_things, 
        f'{conn}-base', 
        f'{conn}-y2fu', 
        covar=[f'{i}-avg' for i in motion_cols],
        method='spearman'
    )
    corr_semi.at[conn, ('partial', 'r')] = pcr['r'].values[0]
    corr_semi.at[conn, ('partial', 'ci95')] = pcr['CI95%'].values[0]
    corr_semi.at[conn, ('partial', 'p(r)')] = pcr['p-val'].values[0]

corr_semi.to_pickle(join(PROJ_DIR, OUTP_DIR, 'deltafc-timepoint_correlations.pkl'))
#print(corr_semi.describe())