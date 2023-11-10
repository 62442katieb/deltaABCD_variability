#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import topcorr as tpc
import networkx as nx

import bct
import enlighten

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

graph_measures = [
    'global_efficiency', 
    'modularity'
]

cols = pd.MultiIndex.from_product((tpts,graph_measures))
graph_df = pd.DataFrame(
    index=base_rsfc.index, 
    columns=cols,
    dtype=float
)

cols = pd.MultiIndex.from_product((tpts,list(network_wise.keys())))
local_df = pd.DataFrame(
    index=base_rsfc.index, 
    columns=cols,
    dtype=float
)

no_2yfu = []

manager = enlighten.get_manager()
tocks = manager.counter(total=len(graph_df.index), desc='Progress', unit='ppts')

for ppt in graph_df.index:
    corrmat = unvectorize_r(base_rsfc.loc[ppt], list(network_wise.keys()))
    A = tpc.tmfg(corrmat, absolute=True, threshold_mean=True)
    A = nx.to_numpy_array(A)
    ntwk_df = pd.DataFrame(
        np.fill_diagonal(A, 0),
        columns=list(network_wise.keys()), 
        index=list(network_wise.keys())
        )
    for ntwk in list(network_wise.keys()):
        local_df.at[ppt, (tpts[0], ntwk)] = ntwk_df[ntwk].sum()
    graph_df.at[ppt, (tpts[0], 'modularity')] = bct.modularity_louvain_und_sign(A)[1]
    #A = bct.threshold_proportional(corrmat, 0.5)
    graph_df.at[ppt, (tpts[0], 'global_efficiency')] = bct.efficiency_bin(A)
    try:
        corrmat = unvectorize_r(y2fu_rsfc.loc[ppt], list(network_wise.keys()))
        A = tpc.tmfg(corrmat, absolute=True, threshold_mean=True)
        A = nx.to_numpy_array(A)
        ntwk_df = pd.DataFrame(
            np.fill_diagonal(A, 0),
            columns=list(network_wise.keys()), 
            index=list(network_wise.keys())
            )
        for ntwk in list(network_wise.keys()):
            local_df.at[ppt, (tpts[1], ntwk)] = ntwk_df[ntwk].sum() / 2
        graph_df.at[ppt, (tpts[1], 'modularity')] = bct.modularity_louvain_und_sign(A)[1]
        #A = bct.threshold_proportional(corrmat, 0.5)
        graph_df.at[ppt, (tpts[1], 'global_efficiency')] = bct.efficiency_bin(A)
    except:
        no_2yfu.append(ppt)
    tocks.update()

graph_df.to_pickle(join(PROJ_DIR, OUTP_DIR, 'delta_rsFC-graph_measures-global.pkl'))
local_df.to_pickle(join(PROJ_DIR, OUTP_DIR, 'delta_rsFC-graph_measures-local.pkl'))

long_graph = graph_df.melt(var_name=['visit', 'metric'])
long_local = local_df.melt(var_name=['visit', 'network'])

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

fig,ax = plt.subplots()
sns.pointplot(
    long_local, 
    y='value',
    x='visit',
    hue='network', 
    dodge=True, 
    ax=ax
)
ax.set_title('Network Strength')

ax.set_xlabel('Ages')

ax.set_xticklabels(['9-10 years', '11-13 years'])
#ax.axvline(long_mod[long_mod['visit'] == 'baseline_year_1_arm_1'])
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'delta_rsFC-network_strength_base-2yfu.png'), dpi=400, bbox_inches='tight')
