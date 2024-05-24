
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
sns.set(context='paper', palette="Greys_r", style="white")
quartile_palette = sns.color_palette(['#A2B017', '#D87554', '#9518AA', '#0D0993'])

thk_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.pkl'))
var_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.pkl'))
rni_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.pkl'))
rnd_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.pkl'))

# grab variable names for the most sensorimotor and the most associative regions
thk_s = thk_df[thk_df['SA_rank'] < 17].index
thk_a = thk_df[thk_df['SA_rank'] > 51].index

var_s = var_df[var_df['SA_rank'] < 17].index
var_a = var_df[var_df['SA_rank'] > 51].index

rni_s = rni_df[rni_df['SA_rank'] < 17].index
rni_a = rni_df[rni_df['SA_rank'] > 51].index

# for some reason I messed up the SA ranks in rnd
rnd_df['SA_rank'] += 34.5
rnd_s = rnd_df[rnd_df['SA_rank'] < 17].index
rnd_a = rnd_df[rnd_df['SA_rank'] > 51].index

big_df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd.pkl'))

change_scores = [
    'apd',
    'rci'
] 
for score in change_scores:
    # grab raw estimates for each measure at each time point
    # and turn them into a mmultiindexed dataframe
    rni_raw = big_df.filter(regex='dmri_rsirnigm.*year')
    base_dmri = rni_raw.filter(like="baseline")
    y2fu_dmri = rni_raw.filter(like="2_year")
    base_dmri.columns = pd.MultiIndex.from_product(
        [
            ['base'], 
            [i.split('.')[0] for i in base_dmri.columns]
        ]
    )

    y2fu_dmri.columns = pd.MultiIndex.from_product(
        [
            ['2yfu'], 
            [i.split('.')[0] for i in y2fu_dmri.columns]
        ]
    )
    rni_raw = pd.concat([base_dmri, y2fu_dmri], axis=1)

    smri_raw = big_df.filter(regex='smri_thick_cdk.*year')

    base_smri = smri_raw.filter(like="baseline")
    y2fu_smri = smri_raw.filter(like="2_year")
    base_smri.columns = pd.MultiIndex.from_product(
        [
            ['base'], 
            [i.split('.')[0] for i in base_smri.columns]
        ]
    )
    y2fu_smri.columns = pd.MultiIndex.from_product(
        [
            ['2yfu'], 
            [i.split('.')[0] for i in y2fu_smri.columns]
        ]
    )
    smri_raw = pd.concat([base_smri, y2fu_smri], axis=1)

    rnd_raw = big_df.filter(regex='dmri_rsirndgm.*year')
    base_dmri = rnd_raw.filter(like="baseline")
    y2fu_dmri = rnd_raw.filter(like="2_year")
    base_dmri.columns = pd.MultiIndex.from_product(
        [
            ['base'], 
            [i.split('.')[0] for i in base_dmri.columns]
        ]
    )

    y2fu_dmri.columns = pd.MultiIndex.from_product(
        [
            ['2yfu'], 
            [i.split('.')[0] for i in y2fu_dmri.columns]
        ]
    )
    rnd_raw = pd.concat([base_dmri, y2fu_dmri], axis=1)

    fmri_raw = big_df.filter(regex='rsfmri_var_cdk.*year')
    base_fmri = fmri_raw.filter(like="baseline")
    y2fu_fmri = fmri_raw.filter(like="2_year")
    base_fmri.columns = pd.MultiIndex.from_product(
        [
            ['base'], 
            [i.split('.')[0] for i in base_fmri.columns]
        ]
    )

    y2fu_fmri.columns = pd.MultiIndex.from_product(
        [
            ['2yfu'], 
            [i.split('.')[0] for i in y2fu_fmri.columns]
        ]
    )
    fmri_raw = pd.concat([base_fmri, y2fu_fmri], axis=1)

    # get rid of big_df for memory reasons
    #big_df = None

    # read in individual-level alignment
    thk_corrs = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, f'sa_thk_corrs-{score}.pkl'))
    var_corrs = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, f'sa_var_corrs-{score}.pkl'))
    rni_corrs = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, f'sa_rni_corrs-{score}.pkl'))
    rnd_corrs = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, f'sa_rnd_corrs-{score}.pkl'))

    # get ppt IDs for the significant aligners/contrasters
    sig_thk = thk_corrs[thk_corrs['p'] < 0.001]
    sig_var = var_corrs[var_corrs['p'] < 0.001]
    sig_rni = rni_corrs[rni_corrs['p'] < 0.001]
    sig_rnd = rnd_corrs[rnd_corrs['p'] < 0.001]

    align_thk = sig_thk[sig_thk['r'] > 0].index
    contr_thk = sig_thk[sig_thk['r'] < 0].index

    fig,ax = plt.subplots(figsize=(2,1))
    sns.kdeplot(thk_corrs['r'], fill=True, ax=ax)
    ax.set_xlabel(r'$r_S$')
    ax.set_ylabel('')
    ax.axvline(x=thk_corrs.loc[align_thk]['r'].min(), lw=sns.plotting_context()['lines.linewidth'], ls='--', color='#333333', alpha=0.4)
    ax.axvline(x=thk_corrs.loc[contr_thk]['r'].max(), lw=sns.plotting_context()['lines.linewidth'], ls='--', color='#333333', alpha=0.4) 
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'thk_corr_dist-{score}.png'), dpi=300, bbox_inches='tight')

    align_var = sig_var[sig_var['r'] > 0].index
    contr_var = sig_var[sig_var['r'] < 0].index

    fig,ax = plt.subplots(figsize=(2,1))
    sns.kdeplot(var_corrs['r'], fill=True, ax=ax)
    ax.set_xlabel(r'$r_S$')
    ax.set_ylabel('')
    ax.axvline(x=var_corrs.loc[align_var]['r'].min(), lw=sns.plotting_context()['lines.linewidth'], ls='--', color='#333333', alpha=0.4)
    ax.axvline(x=var_corrs.loc[contr_var]['r'].max(), lw=sns.plotting_context()['lines.linewidth'], ls='--', color='#333333', alpha=0.4) 
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'var_corr_dist-{score}.png'), dpi=300, bbox_inches='tight')

    align_rni = sig_rni[sig_rni['r'] > 0].index
    contr_rni = sig_rni[sig_rni['r'] < 0].index

    fig,ax = plt.subplots(figsize=(2,1))
    sns.kdeplot(rni_corrs['r'], fill=True, ax=ax)
    ax.set_xlabel(r'$r_S$')
    ax.set_ylabel('')
    ax.axvline(x=rni_corrs.loc[align_rni]['r'].min(), lw=sns.plotting_context()['lines.linewidth'], ls='--', color='#333333', alpha=0.4)
    ax.axvline(x=rni_corrs.loc[contr_rni]['r'].max(), lw=sns.plotting_context()['lines.linewidth'], ls='--', color='#333333', alpha=0.4) 
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'rni_corr_dist-{score}.png'), dpi=300, bbox_inches='tight')

    align_rnd = sig_rnd[sig_rnd['r'] > 0].index
    contr_rnd = sig_rnd[sig_rnd['r'] < 0].index

    fig,ax = plt.subplots(figsize=(2,1))
    sns.kdeplot(rnd_corrs['r'], fill=True, ax=ax)
    ax.set_xlabel(r'$r_S$')
    ax.set_ylabel('')
    ax.axvline(x=rnd_corrs.loc[align_rnd]['r'].min(), lw=sns.plotting_context()['lines.linewidth'], ls='--', color='#333333', alpha=0.4)
    ax.axvline(x=rnd_corrs.loc[contr_rnd]['r'].max(), lw=sns.plotting_context()['lines.linewidth'], ls='--', color='#333333', alpha=0.4) 
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'rnd_corr_dist-{score}.png'), dpi=300, bbox_inches='tight')

    # make dataframes with the most sensorimotor regions for aligners/contrasters
    # and the most associative regions for aligners/contrasters

    align_thk_s = smri_raw.loc[align_thk].swaplevel(axis=1)[thk_s].melt()
    align_thk_a = smri_raw.loc[align_thk].swaplevel(axis=1)[thk_a].melt()

    contr_thk_s = smri_raw.loc[contr_thk].swaplevel(axis=1)[thk_s].melt()
    contr_thk_a = smri_raw.loc[contr_thk].swaplevel(axis=1)[thk_a].melt()

    # plot the most sensorimotor and the most associative regions' values
    # separately for aligners and contrasters
    fig,ax = plt.subplots(
        ncols=4, 
        figsize=(4,1.2), 
        layout='constrained', 
        #sharey=True
    )
    sns.pointplot(contr_thk_s, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[0], palette='YlOrBr_r')
    sns.pointplot(contr_thk_a, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[1], palette='RdPu')
    sns.pointplot(align_thk_s, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[2], palette='YlOrBr_r')
    sns.pointplot(align_thk_a, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[3], palette='RdPu')
    for i in range(4):
        ax[i].get_legend().remove()
        ax[i].set_xlabel('')
        if i > 0:
            ax[i].set_ylabel('')
        else:
            ax[i].set_ylabel('Cortical thickness')
    fig.set_constrained_layout_pads(w_pad=-10 / 72, h_pad=-10 / 72, hspace=0, wspace=0)
    sns.despine()
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'tails_s_vs_a-thk-{score}.png'), 
        dpi=300, 
        bbox_inches='tight'
    )

    # now for fmri
    align_var_s = fmri_raw.loc[align_var].swaplevel(axis=1)[var_s].melt()
    align_var_a = fmri_raw.loc[align_var].swaplevel(axis=1)[var_a].melt()

    contr_var_s = fmri_raw.loc[contr_var].swaplevel(axis=1)[var_s].melt()
    contr_var_a = fmri_raw.loc[contr_var].swaplevel(axis=1)[var_a].melt()

    # plot the most sensorimotor and the most associative regions' values
    # separately for aligners and contrasters
    fig,ax = plt.subplots(
        ncols=4, 
        figsize=(4,1.2), 
        layout='constrained', 
        #sharey=True
    )

    sns.pointplot(contr_var_s, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[0], palette='YlOrBr_r')
    sns.pointplot(contr_var_a, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[1], palette='RdPu')
    sns.pointplot(align_var_s, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[2], palette='YlOrBr_r')
    sns.pointplot(align_var_a, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[3], palette='RdPu')
    for i in range(4):
        ax[i].get_legend().remove()
        ax[i].set_xlabel('')
        if i > 0:
            ax[i].set_ylabel('')
        else:
            ax[i].set_ylabel('Functional fluctuations')
    fig.set_constrained_layout_pads(w_pad=-10 / 72, h_pad=-10 / 72, hspace=0, wspace=0)
    sns.despine()
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'tails_s_vs_a-var-{score}.png'), 
        dpi=300, 
        bbox_inches='tight'
    )

    # now for rni
    align_rni_s = rni_raw.loc[align_rni].swaplevel(axis=1)[rni_s].melt()
    align_rni_a = rni_raw.loc[align_rni].swaplevel(axis=1)[rni_a].melt()

    contr_rni_s = rni_raw.loc[contr_rni].swaplevel(axis=1)[rni_s].melt()
    contr_rni_a = rni_raw.loc[contr_rni].swaplevel(axis=1)[rni_a].melt()

    # plot the most sensorimotor and the most associative regions' values
    # separately for aligners and contrasters
    fig,ax = plt.subplots(
        ncols=4, 
        figsize=(4,1.2), 
        layout='constrained', 
        #sharey=True
    )
    sns.pointplot(contr_rni_s, x='variable_1', y='value', hue='variable_0',scale=0.75, 
                ax=ax[0], palette='YlOrBr_r')
    sns.pointplot(contr_rni_a, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[1], palette='RdPu')
    sns.pointplot(align_rni_s, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[2], palette='YlOrBr_r')
    sns.pointplot(align_rni_a, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[3], palette='RdPu')
    for i in range(4):
        ax[i].get_legend().remove()
        ax[i].set_xlabel('')
        if i > 0:
            ax[i].set_ylabel('')
        else:
            ax[i].set_ylabel('Isotropic diffusion')
    sns.despine()
    fig.set_constrained_layout_pads(w_pad=-10 / 72, h_pad=-10 / 72, hspace=0, wspace=0)
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'tails_s_vs_a-rni-{score}.png'), 
        dpi=300, 
        bbox_inches='tight'
    )

    # now for rnd
    align_rnd_s = rnd_raw.loc[align_rnd].swaplevel(axis=1)[rnd_s].melt()
    align_rnd_a = rnd_raw.loc[align_rnd].swaplevel(axis=1)[rnd_a].melt()

    contr_rnd_s = rnd_raw.loc[contr_rnd].swaplevel(axis=1)[rnd_s].melt()
    contr_rnd_a = rnd_raw.loc[contr_rnd].swaplevel(axis=1)[rnd_a].melt()

    # plot the most sensorimotor and the most associative regions' values
    # separately for aligners and contrasters
    fig,ax = plt.subplots(
        ncols=4, 
        figsize=(4,1.2), 
        layout='constrained', 
        #sharey=True
    )
    sns.pointplot(contr_rnd_s, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[0], palette='YlOrBr_r')
    sns.pointplot(contr_rnd_a, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[1], palette='RdPu')
    sns.pointplot(align_rnd_s, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[2], palette='YlOrBr_r')
    sns.pointplot(align_rnd_a, x='variable_1', y='value', hue='variable_0',scale=0.75,
                ax=ax[3], palette='RdPu')
    for i in range(4):
        ax[i].get_legend().remove()
        ax[i].set_xlabel('')
        if i > 0:
            ax[i].set_ylabel('')
        else:
            ax[i].set_ylabel('Directional diffusion')
    sns.despine()
    fig.set_constrained_layout_pads(w_pad=-10 / 72, h_pad=-10 / 72, hspace=0, wspace=0)
    fig.savefig(
        join(PROJ_DIR, FIGS_DIR, f'tails_s_vs_a-rnd-{score}.png'), 
        dpi=300, 
        bbox_inches='tight'
    )

