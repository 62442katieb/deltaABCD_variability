#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns

import bct

from os.path import join
import warnings

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

rsfc = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.0/core/imaging/mri_y_rsfmr_cor_gp_gp.csv",
    header=0,
    index_col=[0,1]
).dropna()

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

base_df = rsfc.swaplevel(axis=0).loc['baseline_year_1_arm_1']
y2fu_df = rsfc.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1']


graph_measures = [
    'global_efficiency', 
    'modularity'
]

cols = pd.MultiIndex.from_product((tpts,graph_measures))
graph_df = pd.DataFrame(
    index=base_df.index, 
    columns=cols,
    dtype=float
)

no_2yfu = []

for ppt in graph_df.index:
    corrmat = unvectorize_r(base_df.loc[ppt], list(network_wise.keys()))
    graph_df.at[ppt, (tpts[0], 'modularity')] = bct.modularity_louvain_und_sign(corrmat)[1]
    A = bct.threshold_proportional(corrmat, 0.5)
    graph_df.at[ppt, (tpts[0], 'global_efficiency')] = bct.efficiency_bin(A)
    try:
        corrmat = unvectorize_r(y2fu_df.loc[ppt], list(network_wise.keys()))
        graph_df.at[ppt, (tpts[1], 'modularity')] = bct.modularity_louvain_und_sign(corrmat)[1]
        A = bct.threshold_proportional(corrmat, 0.5)
        graph_df.at[ppt, (tpts[1], 'global_efficiency')] = bct.efficiency_bin(A)
    except:
        no_2yfu.append(ppt)

graph_df.to_pickle(join(PROJ_DIR, OUTP_DIR, 'delta_rsFC-graph_measures.pkl'))

long_graph = graph_df.melt(var_name=['visit', 'metric'])

long_mod = long_graph[long_graph['metric'] == 'modularity']
long_eff = long_graph[long_graph['metric'] == 'global_efficiency']

fig,ax = plt.subplots(ncols=2, layout='constrained')
sns.pointplot(
    long_mod, 
    y='value',
    x='visit',
    #hue='visit', 
    #multiple='stack', 
    ax=ax[0]
)
sns.pointplot(
    long_eff, 
    y='value',
    x='visit',
    #hue='visit', 
    #multiple='stack', 
    ax=ax[1]
)
ax[0].set_title('Modularity')
ax[1].set_title('Global Efficiency')

ax[0].set_xlabel('Ages')
ax[1].set_xlabel('Ages')

ax[0].set_xticklabels(['9-10 years', '11-13 years'])
ax[1].set_xticklabels(['9-10 years', '11-13 years'])
#ax.axvline(long_mod[long_mod['visit'] == 'baseline_year_1_arm_1'])
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'delta_rsFC-graph_theory_base-2yfu.png'), dpi=400, bbox_inches='tight')
