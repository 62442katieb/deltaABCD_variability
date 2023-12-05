#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
from scipy.stats import ttest_rel
import warnings

warnings.filterwarnings("ignore")

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))
# grap ppt ids for people who pass QC
ppts = df.filter(regex="rsfmri_c_.*change_score").dropna().index
df = None
# we need ids for the vectorized upper triangle!
network_names = ['dt', 'ca', 'smh', 'dla', 'ad', 'smm', 'sa', 'fo', 'vs', 'cgc', 'vta', 'rspltp']

# reading in the dataframes for each type of change score
sign_change = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'delta_rsFC-sign_changes.pkl'))
graph_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'delta_rsFC-graph_measures-global.pkl'))
local_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'delta_rsFC-graph_measures-local.pkl'))

# I'm just interested in ppts who pass the quality control process
quality_ppts = list(set(ppts) & set(graph_df.index))

graph_df = graph_df.loc[quality_ppts]
local_df = local_df.loc[quality_ppts]

############### global measures ###################
# one per person per timepoint
# in graph_df
# measures are 'modularity' and 'global_efficiency'

# calculate descriptives for baseline and 2-year follow-up using .describe()

# now let's do a paired t-test to see if they increase or decrease
efficiency = graph_df.swaplevel(axis=1)['global_efficiency']
efficiency = efficiency.dropna()
print(efficiency.describe())

change_eff = ttest_rel(efficiency['2_year_follow_up_y_arm_1'], efficiency['baseline_year_1_arm_1'])
# save out the ttest results (t, p) and the number of participants included in the test
# can find number of participants via len(efficiency.index)
# now do the same for modularity

# and save out the results (however you like)

############# local measures ###################
# one per brain network per person per timepoint
# in local_df
# measures are 'clust_coeff' and 'btwn_cent'
local_df = local_df.dropna(how='all')

base_local = local_df.swaplevel(axis=0).loc['baseline_year_1_arm_1']
y2fu_local = local_df.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1']

# use .describe() to get descriptives for each local measure and each network
# separately for baseline and 2-year follow-up

# now we'll assess change, again
# here's how you'd do it for one network:

# it's probably easiest to just make a temporary dataframe per network
# per measure to make sure the participants are aligned

temp_cc_ad = pd.concat(
    [base_local['clust_coeff']['ad'].rename('base'),
      y2fu_local['clust_coeff']['ad'].rename('y2fu')], 
      axis=1).dropna()
print(ttest_rel(temp_cc_ad['y2fu'], temp_cc_ad['base']))

