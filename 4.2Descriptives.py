#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns

from os.path import join
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")



PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))

ppts = df.filter(regex="rsfmri_c_ngd_.*change_score").dropna().index

rsfc = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.0/core/imaging/mri_y_rsfmr_cor_gp_gp.csv",
    header=0,
    index_col=[0,1]
).dropna()

tpts = [
    'baseline_year_1_arm_1',
    '2_year_follow_up_y_arm_1'
]

# make a list of all of the column names (resting-state FC estimates) that represent
# within-network connections
within_network = [i for i in rsfc.columns if i.split('_')[3] == i.split('_')[5]]

#base_df = rsfc.swaplevel(axis=0).loc['baseline_year_1_arm_1']
#y2fu_df = rsfc.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1']

# we need ids for the vectorized upper triangle!
network_names = ['dt', 'ca', 'smh', 'dla', 'ad', 'smm', 'sa', 'fo', 'vs', 'cgc', 'vta', 'rspltp']

variable_df = pd.DataFrame(dtype=str)
for ntwk1 in network_names:
    i = network_names.index(ntwk1)
    for ntwk2 in network_names:
        j = network_names.index(ntwk2)
        variable_df.at[i,j] = f'rsfmri_c_ngd_{ntwk1}_ngd_{ntwk2}'

upper_tri = np.triu_indices(12)
upper_tri_vars = list(variable_df.values[upper_tri])

# reading in the dataframes for each type of change score
sign_change = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-sign_changes.pkl'))
rci = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-rci.pkl'))
rci_abs = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-rci_abs.pkl'))
change = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-change.pkl'))
change_abs = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-change_abs.pkl'))
change_plus1 = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-change_plus1.pkl'))

measures = {
    'rci': rci,
    '|rci|': rci_abs,
    'apd': change,
    '|apd|': change_abs,
    'delta+1': change_plus1
}

for measure in measures.values():
    measure = measure[upper_tri_vars]


btwn = upper_tri_vars

# make a list per network of all of that network's connections
network_wise = {}
for i in [j.split('_')[3] for j in within_network]:
    network_wise[i] = [k for k in btwn if i == k.split('_')[3]]

# make a list of all between-network connections
between_network = {}
for network in network_wise.keys():
    between_network[network] = [i for i in network_wise[network] if i.split('_')[3] != i.split('_')[5]]


# compute the descriptive statistics for 
# 1. all connections of each network (see `network_wise`)
# 2. each within-network connection (see `within_network`)
# 3. all of the between-network connections, per network

# here's an example of 1

# calculating descriptives of all auditory network connections
# then, grab all the variable names for ad connections
ad_conns = network_wise['ad']
# compute descriptives for each way that we calculated deltaFC
for measure in measures.keys():
    # this is the vectorized upper tri dataframe, we're grabbing just the auditory network connections
    ad_df = measures[measure][ad_conns]
    # by "melting" that dataframe, we can collapse across _all_ auditory network connections
    all_ad = ad_df.melt()
    # then you can "describe" this melted df to get descriptive statistics about _all_ changes in auditory network connectivity
    # or you can use all_ad.mean() to get the mean & .std() to get the standard deviation
    # make sure to save out the mean and standard deviation for each combination of network & deltaFC measure
    

# now repeat this with all the other networks 
# don't re-initialize the `network_descriptives` dataframe, just keep adding to it


# now a scaffold for 2. within-network connectivity

# compute descriptives for each way that we calculated deltaFC
for measure in measures.keys():
    # we'll just loop over all the within-network connections
    for conn in within_network:
        # let's pull out the network name
        network_name = conn.split('_')[3]
        #here's a small dataframe with just the within-network connectivity
        temp = measures[measure][conn]

        # now translate the way you did descriptives above, but just for `temp`
        # no need to melt, since we've only got one variable.

        # concatenate them onto `network_descriptives`

# and now a scaffold for 3!
# between_network is a dictionary of between-network connections for each network
# like network_wise, but without the within-network connections

ad_conns = between_network['ad']

# compute descriptives for each way that we calculated deltaFC
for measure in measures.keys():
    # this is the vectorized upper tri dataframe, we're grabbing just the auditory network connections
    ad_df = measures[measure][ad_conns]
    # by "melting" that dataframe, we can collapse across _all_ auditory network connections
    all_ad = ad_df.melt()
    # then we can "describe" this melted df to get descriptive statistics about _all_ changes in auditory network connectivity
    # or you can use all_ad.mean() to get the mean & .std() to get the standard deviation
    # make sure to save out the mean and standard deviation for each combination of network & deltaFC measure
   