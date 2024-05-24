#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
from scipy.stats import spearmanr
from utils import jili_sidak_mc

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

thk_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.pkl'))
var_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.pkl'))
rni_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.pkl'))
rnd_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.pkl'))

change_scores = {
    'apd': pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_change_scores.pkl')),
    'rci': pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_rci.pkl')),
} 
for score in change_scores.keys():
    temp_df = change_scores[score]
    # this cell does the correlations
    residualized_thk = temp_df.filter(like="smri_thick_cdk")
    residualized_rnd = temp_df.filter(like="dmri_rsirndgm_cdk_")
    residualized_rni = temp_df.filter(like="dmri_rsirnigm_cdk_")
    residualized_var = temp_df.filter(like="rsfmri_var_cdk")
    # first, make empty dataframes that we'll fill in the for loop
    # for s-a axis loading corrs/alignment
    sa_rnd_corrs = pd.DataFrame()


    # now for each person (i),
    for i in residualized_rnd.index:
        # we'll grab all their residual RND change scores
        temp1 = residualized_rnd.loc[i]
        # and rename the mini-dataframe, so that we know these are per-participant values
        temp1.name = 'ppt'
        # fix the index so that it matches the SA-axis rank
        temp1.index = [var.split('.')[0] for var in temp1.index]
        # just grab the S-A axis rank column from rnd_df
        temp2 = rnd_df['SA_avg']
        # rename it so that we know these are per-region s-a axis values
        temp2.name = 'sa_axis'
        # put those two mini-dfs together to make life easier
        # this aligns them based on the index, which they share
        temp = pd.concat([temp1, temp2], axis=1).dropna()
        # correlate! using the new names we gave the two columns
        r,p = spearmanr(temp['ppt'], temp['sa_axis'])
        # save the r values to the 'r' column and the p-values to the 'p' column
        sa_rnd_corrs.at[i,'p'] = p
        sa_rnd_corrs.at[i,'r'] = r


    sa_rnd_corrs.to_pickle(join(PROJ_DIR, OUTP_DIR, f'sa_rnd_corrs-{score}.pkl'))

    # set the plotting settings so our graphs are pretty
    sns.set(context='paper', style='white', palette='Greys_r')
    sig = sa_rnd_corrs[sa_rnd_corrs['p'] < 0.01]['r']
    thresh_pos = sig[sig > 0].min()
    thresh_neg = sig[sig < 0].max()
    # we're going to plot all the correlations
    # and in a different color, the significant correlations @ p < 0.01
    fig,ax = plt.subplots(figsize=(4,2))
    sns.kdeplot(sa_rnd_corrs['r'], fill=True, ax=ax)
    ax.axvline(x=thresh_pos, lw=2, ls='--', color='#333333', alpha=0.4)
    ax.axvline(x=thresh_neg, lw=2, ls='--', color='#333333', alpha=0.4) 
    #sns.kdeplot(sa_rnd_corrs[sa_rnd_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'rnd_x_sa-axis-{score}.png'), dpi=400, bbox_inches='tight')


    # this cell does the correlations

    # first, make empty dataframes that we'll fill in the for loop
    # for s-a axis loading corrs/alignment
    sa_rni_corrs = pd.DataFrame()

    # now for each person (i),
    for i in residualized_rni.index:
        # we'll grab all their residual RNI change scores
        temp1 = residualized_rni.loc[i]
        # and rename the mini-dataframe, so that we know these are per-participant values
        temp1.name = 'ppt'
        # fix the index so that it matches the SA-axis rank
        temp1.index = [var.split('.')[0] for var in temp1.index]
        # just grab the S-A axis rank column from rni_df
        temp2 = rni_df['SA_avg']
        # rename it so that we know these are per-region s-a axis values
        temp2.name = 'sa_axis'
        # put those two mini-dfs together to make life easier
        # this aligns them based on the index, which they share
        temp = pd.concat([temp1, temp2], axis=1).dropna()
        # correlate! using the new names we gave the two columns
        r,p = spearmanr(temp['ppt'], temp['sa_axis'])
        # save the r values to the 'r' column and the p-values to the 'p' column
        sa_rni_corrs.at[i,'p'] = p
        sa_rni_corrs.at[i,'r'] = r


    sa_rni_corrs.to_pickle(join(PROJ_DIR, OUTP_DIR, f'sa_rni_corrs-{score}.pkl'))


    # set the plotting settings so our graphs are pretty
    sns.set(context='paper', style='white', palette='Greys_r')
    sig = sa_rni_corrs[sa_rni_corrs['p'] < 0.01]['r']
    thresh_pos = sig[sig > 0].min()
    thresh_neg = sig[sig < 0].max()
    # we're going to plot all the correlations
    # and in a different color, the significant correlations @ p < 0.01
    fig,ax = plt.subplots(figsize=(4,2))
    sns.kdeplot(sa_rni_corrs['r'], fill=True, ax=ax, warn_singular=False)
    ax.axvline(x=thresh_pos, lw=2, ls='--', color='#333333', alpha=0.4)
    ax.axvline(x=thresh_neg, lw=2, ls='--', color='#333333', alpha=0.4) 
    #sns.kdeplot(sa_rni_corrs[sa_rni_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'rni_x_sa-axis-{score}.png'), dpi=400, bbox_inches='tight')


    # this cell does the correlations

    # first, make empty dataframes that we'll fill in the for loop
    # for s-a axis loading corrs/alignment
    sa_var_corrs = pd.DataFrame()


    # now for each person (i),
    for i in residualized_var.index:
        # we'll grab all their residual RSFMRI change scores
        temp1 = residualized_var.loc[i]
        # and rename the mini-dataframe, so that we know these are per-participant values
        temp1.name = 'ppt'
        # fix the index so that it matches the SA-axis rank
        temp1.index = [var.split('.')[0] for var in temp1.index]
        # just grab the S-A axis rank column from var_df
        temp2 = var_df['SA_avg']
        # rename it so that we know these are per-region s-a axis values
        temp2.name = 'sa_axis'
        # put those two mini-dfs together to make life easier
        # this aligns them based on the index, which they share
        temp = pd.concat([temp1, temp2], axis=1).dropna()
        # correlate! using the new names we gave the two columns
        r,p = spearmanr(temp['ppt'], temp['sa_axis'])
        # save the r values to the 'r' column and the p-values to the 'p' column
        sa_var_corrs.at[i,'p'] = p
        sa_var_corrs.at[i,'r'] = r

        

    sa_var_corrs.to_pickle(join(PROJ_DIR, OUTP_DIR, f'sa_var_corrs-{score}.pkl'))


    # set the plotting settings so our graphs are pretty
    sns.set(context='paper', style='white', palette='Greys_r')
    sig = sa_var_corrs[sa_var_corrs['p'] < 0.01]['r']
    thresh_pos = sig[sig > 0].min()
    thresh_neg = sig[sig < 0].max()
    # we're going to plot all the correlations
    # and in a different color, the significant correlations @ p < 0.01
    fig,ax = plt.subplots(figsize=(4,2))
    sns.kdeplot(sa_var_corrs['r'], fill=True, ax=ax, warn_singular=False)
    ax.axvline(x=thresh_pos, lw=2, ls='--', color='#333333', alpha=0.4)
    ax.axvline(x=thresh_neg, lw=2, ls='--', color='#333333', alpha=0.4) 
    #sns.kdeplot(sa_var_corrs[sa_var_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'rsfmri_x_sa-axis-{score}.png'), dpi=400, bbox_inches='tight')



    # this cell does the correlations

    # first, make empty dataframes that we'll fill in the for loop
    # for s-a axis loading corrs/alignment
    sa_thk_corrs = pd.DataFrame()


    # now for each person (i),
    for i in residualized_thk.index:
        # we'll grab all their residual thickness change scores
        temp1 = residualized_thk.loc[i]
        # and rename the mini-dataframe, so that we know these are per-participant values
        temp1.name = 'ppt'
        # fix the index so that it matches the SA-axis rank
        temp1.index = [var.split('.')[0] for var in temp1.index]
        # just grab the S-A axis rank column from thk_df
        temp2 = thk_df['SA_avg']
        # rename it so that we know these are per-region s-a axis values
        temp2.name = 'sa_axis'
        # put those two mini-dfs together to make life easier
        # this aligns them based on the index, which they share
        temp = pd.concat([temp1, temp2], axis=1).dropna()
        # correlate! using the new names we gave the two columns
        r,p = spearmanr(temp['ppt'], temp['sa_axis'])
        # save the r values to the 'r' column and the p-values to the 'p' column
        sa_thk_corrs.at[i,'p'] = p
        sa_thk_corrs.at[i,'r'] = r


    sa_thk_corrs.to_pickle(join(PROJ_DIR, OUTP_DIR, f'sa_thk_corrs-{score}.pkl'))


    # set the plotting settings so our graphs are pretty
    sns.set(context='paper', style='white', palette='Greys_r')
    sig = sa_thk_corrs[sa_thk_corrs['p'] < 0.01]['r']
    thresh_pos = sig[sig > 0].min()
    thresh_neg = sig[sig < 0].max()
    # we're going to plot all the correlations
    # and in a different color, the significant correlations @ p < 0.01
    fig,ax = plt.subplots(figsize=(4,2))
    sns.kdeplot(sa_thk_corrs['r'], fill=True, ax=ax, warn_singular=False)
    ax.axvline(x=thresh_pos, lw=2, ls='--', color='#333333', alpha=0.4)
    ax.axvline(x=thresh_neg, lw=2, ls='--', color='#333333', alpha=0.4) 
    #sns.kdeplot(sa_thk_corrs[sa_thk_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'thk_x_sa-axis-{score}.png'), dpi=400, bbox_inches='tight')



    r_df = pd.concat(
        [
            sa_thk_corrs['r'].rename('Cortical thickness'),
            sa_var_corrs['r'].rename('Functional variance'),
            sa_rni_corrs['r'].rename('Isotropic diffusion'),
            sa_rnd_corrs['r'].rename('Directional diffusion'),

        ],
        axis=1
    )
    descriptives = r_df.describe()
    for col in descriptives.columns:
        descriptives.at['range', col] = descriptives.loc['max'][col] - descriptives.loc['min'][col]
    descriptives.at['align', 'Cortical thickness'] = np.sum(sa_thk_corrs[sa_thk_corrs['p'] < 0.01]['r'] > 0)
    descriptives.at['contrast', 'Cortical thickness'] = np.sum(sa_thk_corrs[sa_thk_corrs['p'] < 0.01]['r'] < 0)

    descriptives.at['align', 'Functional variance'] = np.sum(sa_var_corrs[sa_var_corrs['p'] < 0.01]['r'] > 0)
    descriptives.at['contrast', 'Functional variance'] = np.sum(sa_var_corrs[sa_var_corrs['p'] < 0.01]['r'] < 0)

    descriptives.at['align', 'Isotropic diffusion'] = np.sum(sa_rni_corrs[sa_rni_corrs['p'] < 0.01]['r'] > 0)
    descriptives.at['contrast', 'Isotropic diffusion'] = np.sum(sa_rni_corrs[sa_rni_corrs['p'] < 0.01]['r'] < 0)

    descriptives.at['align', 'Directional diffusion'] = np.sum(sa_rnd_corrs[sa_rnd_corrs['p'] < 0.01]['r'] > 0)
    descriptives.at['contrast', 'Directional diffusion'] = np.sum(sa_rnd_corrs[sa_rnd_corrs['p'] < 0.01]['r'] < 0)
    descriptives.to_csv(join(PROJ_DIR, OUTP_DIR, f'{score}-alignment_descriptives.csv'))
    a_corr, M_eff = jili_sidak_mc(r_df, 0.05)
    np.savetxt(join(PROJ_DIR, OUTP_DIR, f'{score}-alpha_corr.txt'), np.asanyarray((a_corr, M_eff)))
    fig,ax = plt.subplots(figsize=(5,2))
    sns.heatmap(r_df.dropna().sort_values('Cortical thickness').T, ax=ax, cmap='RdBu_r', center=0, vmax=1, vmin=-1)
    ax.set_xticklabels('')
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'SA_corrs_across_measures-{score}.png'), dpi=400, bbox_inches='tight')

    fig,ax = plt.subplots(figsize=(2,2))
    sns.kdeplot(r_df, palette='Set2', fill=True, ax=ax)
    ax.set_xlabel('')
    ax.set_xlim(-1, 1)
    legend = ax.get_legend()
    #legend.set_ncols(2)
    legend.set_bbox_to_anchor((1,-0.15))
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'distribution-SA_corrs_across_measures-{score}.png'), dpi=400, bbox_inches='tight')