#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import bct
import pingouin as pg

from os.path import join
import warnings

from nilearn import plotting
from scipy.stats import ttest_rel
from utils import series_2_nifti

import matplotlib.pyplot as plt

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

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))

ppts = df.filter(regex="rsfmri_c_.*change_score").dropna().index

base_rsfc = pd.read_pickle(
    join(PROJ_DIR, DATA_DIR, 'rsfc_sans_motion-baseline.pkl')
).dropna()

y2fu_rsfc = pd.read_pickle(
    join(PROJ_DIR, DATA_DIR, 'rsfc_sans_motion-2yearfup.pkl')
).dropna()

nones = list(base_rsfc.filter(regex='rsfmri_c_ngd_.*_ngd_n').columns) + list(base_rsfc.filter(regex='rsfmri_c_ngd_n_.*').columns)
base_rsfc = base_rsfc.drop(nones, axis=1)
y2fu_rsfc = y2fu_rsfc.drop(nones, axis=1)

tpts = [
    'baseline_year_1_arm_1',
    '2_year_follow_up_y_arm_1'
]

within_network = [i for i in base_rsfc.columns if i.split('_')[3] == i.split('_')[5]]

btwn = base_rsfc.columns

network_wise = {}

for i in [j.split('_')[3] for j in within_network]:
    network_wise[i] = [k for k in btwn if i == k.split('_')[3]]

between_network = [i for i in base_rsfc.columns if i.split('_')[3] != i.split('_')[5]]

conns = base_rsfc.columns

sign_change = pd.DataFrame(index=ppts, columns=conns, dtype=float)
change = pd.DataFrame(index=ppts, columns=conns, dtype=float)
change_abs = pd.DataFrame(index=ppts, columns=conns, dtype=float)
change_plus1 = pd.DataFrame(index=ppts, columns=conns, dtype=float)
rci = pd.DataFrame(index=ppts, columns=conns, dtype=float)
rci_abs = pd.DataFrame(index=ppts, columns=conns, dtype=float)

measures = {
    'rci': rci,
    '|rci|': rci_abs,
    'apd': change,
    '|apd|': change_abs,
    'delta+1': change_plus1
}

for i in ppts:
    if i not in base_rsfc.index or i not in y2fu_rsfc.index:
        pass
    else:
        age0 = df.loc[i, 'interview_age.baseline_year_1_arm_1'] / 12.
        age2 = df.loc[i, 'interview_age.2_year_follow_up_y_arm_1'] / 12.
        for conn in conns:
            base = base_rsfc.loc[i, conn]
            y2fu = y2fu_rsfc.loc[i, conn]
            
            temp = pd.concat([base_rsfc[conn], y2fu_rsfc[conn]])
            
            sem = np.std(temp.values, ddof=1) / np.sqrt(np.size(temp.values))
            abs_sem = np.std(np.abs(temp.values), ddof=1) / np.sqrt(np.size(temp.values))
            
            rci.at[i,conn] = (y2fu - base) / sem
            rci_abs.at[i,conn] = (np.abs(y2fu) - np.abs(base)) / abs_sem
            #print(base * y2fu)
            if base * y2fu > 0:
                if y2fu > 0:
                    sign_change.at[i, conn] = '+ to +'
                else:
                    sign_change.at[i, conn] = '- to -'
            else:
                if y2fu > 0:
                    sign_change.at[i, conn] = '- to +'
                else:
                    sign_change.at[i, conn] = '+ to -'
            # calc change in raw corr
            change.at[i, conn] = (((y2fu - base) / np.mean([y2fu, base])) * 100) / (age2 - age0)

            # calc change in absolute z-scored correlation
            change_abs.at[i, conn] = (((np.abs(y2fu) - np.abs(base)) / np.mean([np.abs(y2fu), np.abs(base)])) * 100) / (age2 - age0)

            # calc change in corr + 1
            base_plus1 = np.tanh(base) + 1
            y2fu_plus1 = np.tanh(y2fu) + 1
            change_plus1.at[i, conn] = (((y2fu_plus1 - base_plus1) / np.mean([y2fu_plus1, base_plus1])) * 100) / (age2 - age0)


sign_change.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-sign_changes.pkl'))
rci.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-rci.pkl'))
rci_abs.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-rci_abs.pkl'))
change.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-change.pkl'))
change_abs.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-change_abs.pkl'))
change_plus1.to_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-change_plus1.pkl'))

long_corr = pd.concat(
    [
        change.melt(), 
        sign_change.melt(value_name='sign change').drop('variable',axis=1)
    ], 
    axis=1
)
long_corr['score'] = ['z(corr)'] * 593136

long_abs = pd.concat(
    [
        change_abs.melt(), 
        sign_change.melt(value_name='sign change').drop('variable',axis=1)
    ], 
    axis=1
)
long_abs['score'] = ['abs(z(corr))'] * 593136
long_rci = pd.concat(
    [
        rci.melt(), 
        sign_change.melt(value_name='sign change').drop('variable',axis=1)
    ], 
    axis=1
)
long_rci['score'] = ['rci'] * 593136
long_rci_abs = pd.concat(
    [
        rci_abs.melt(), 
        sign_change.melt(value_name='sign change').drop('variable',axis=1)
    ], 
    axis=1
)
long_rci_abs['score'] = ['abs(rci)'] * 593136
long_plus1 = pd.concat(
    [
        change_plus1.melt(), 
        sign_change.melt(value_name='sign change').drop('variable',axis=1)
    ], 
    axis=1
)
long_plus1['score'] = ['corr + 1'] * 593136

mega_df = pd.concat([
    long_corr,
    long_rci,
    long_abs, 
    long_rci_abs, 
    long_plus1
                    ])

# also plot the raw correlation differences
# and another plot of sign changes|
# maybe also correlation changes by sign change

sns.set(style='white')
fig,ax = plt.subplots(figsize=(7,6))
sns.boxenplot(
    mega_df.reset_index(),
    y='value',
    x='score',
    hue='sign change',
    #hue_order=['z(corr)', 'abs(z(corr))', 'rci', 'abs(rci)', 'corr + 1'],
    #color='#d95f02', #orange
    #fill=True,
    ax=ax,
    palette='Set2'
)
ax.set_yscale("symlog")
ax.set_ylim(bottom=-(10**7), top=10**7)
ax.set_ylabel('Change Estimate')
ax.set_xlabel('Change Algorithm')
ax.legend(bbox_to_anchor=(1,1.05), title='Sign Change')
ax.axhline(0, color='#333333', linestyle='dashed', alpha=0.6)
sns.despine()
fig.savefig(
    join(PROJ_DIR, FIGS_DIR, 'change_score_distributions.png'), 
    dpi=400, 
    bbox_inches='tight'
)

nifti_mapping = pd.read_csv('/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/data/variable_to_nifti_mapping.csv', 
                                header=0, 
                                index_col=0)

gordon = nib.load('/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/resources/gordon_networks_222.nii')

network_arr = gordon.get_fdata()

network_vals = nifti_mapping[nifti_mapping['atlas'] == 'Gordon Networks']['atlas_value']

network_vals = [(i.split('_')[3], network_vals[i]) for i in network_vals.index]

ntwk_df = pd.Series()

for i in range(0, len(network_vals)):
    ntwk_df.at[network_vals[i][0]] = network_vals[i][1]

for delta in measures.keys():
    ntwks = {}
    ntwk_conns = {}
    ntwk_niftis = {}
    temp = measures[delta]
    for ntwk in network_wise.keys():
        ntwks[ntwk] = temp.describe()[network_wise[ntwk]].T
        ntwks[ntwk].index = [i.split('_')[-1] for i in ntwks[ntwk].index]
        ntwk_conns[ntwk] = np.zeros_like(network_arr)
        for i in ntwks[ntwk].index:
            val = ntwk_df[i]
            ntwk_conns[ntwk][np.where(network_arr == val)] = ntwks[ntwk].loc[i]['mean']
        ntwk_niftis[ntwk] = nib.Nifti2Image(ntwk_conns[ntwk], gordon.affine)

    for ntwk in ntwk_niftis.keys():
        temp = ntwk_niftis[ntwk]
        plotting.plot_img_on_surf(
            temp, 
            threshold=0.01, 
            cmap='seismic', 
            title=ntwk,
            vmax=5,
            kwargs=dict(alpha=0.6),
            output_file=join(PROJ_DIR, FIGS_DIR, f'{ntwk}_{delta}.png')
        )