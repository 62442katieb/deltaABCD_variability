#!/usr/bin/env python
# coding: utf-8

import enlighten
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from os.path import join, exists
from datetime import datetime
from matplotlib import colors
#from pyampute.exploration.md_patterns import mdPatterns
#from pyampute.exploration.mcar_statistical_tests import MCARTest
#from sklearn.model_selection import KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

# these two are from https://github.com/RianneSchouten/pyampute/blob/master/pyampute/exploration/md_patterns.py
# but they were throwing weird errors so I decided to fix them myself
def _calculate_patterns(
        X: pd.DataFrame, count_or_proportion: str = "count"
    ) -> pd.DataFrame:
    """Extracts all unique missing data patterns in an incomplete dataset and transforms into a pandas DataFrame"""

    # mask
    mask = X.isna()

    # count number of missing values per column
    colsums = mask.sum()
    sorted_col = colsums.sort_values().index.tolist()
    colsums["n_missing_values"] = colsums.sum()
    colsums["row_count"] = ""

    # finding missing values per group
    group_values = (~mask).groupby(sorted_col).size().reset_index(name="row_count")
    group_values["n_missing_values"] = group_values.isin([0]).sum(axis=1)
    group_values.sort_values(
        by=["n_missing_values", "row_count"], ascending=[True, False], inplace=True
    )
    group_values = group_values.append(colsums, ignore_index=True)

    # add extra row to patterns when there are no incomplete rows in dataset
    if group_values.iloc[0, 0:-2].values.tolist() != list(np.ones(len(sorted_col))):
        group_values.loc[-1] = np.concatenate(
            (np.ones(len(sorted_col)), np.zeros(2))
        ).astype(int)
        group_values.index = group_values.index + 1  # shifting index
        group_values.sort_index(inplace=True)

        # put row_count in the begining
    cols = list(group_values)
    cols.insert(0, cols.pop(cols.index("row_count")))
    group_values = group_values.loc[:, cols]

    if count_or_proportion == "proportion":
        group_values.rename(columns={"row_count": "row_prop"}, inplace=True)
        percents = ((group_values.iloc[0:-1, 0]).astype(int) / X.shape[0]).round(2)
        group_values.iloc[0:-1, 0] = percents.astype(str)
        group_values.iloc[-1, 1:-1] = group_values.iloc[-1, 1:-1] / X.shape[0]
        group_values.iloc[-1, -1] = (
            group_values.iloc[-1, -1] / (X.shape[0] * X.shape[1])
        ).round(2)

    md_patterns = group_values
    md_patterns.index = (
        ["rows_no_missing"]
        + list(md_patterns.index[1:-1])
        + ["n_missing_values_per_col"]
    )
    return md_patterns

def _make_plot(md_patterns):
    """"Creates visualization of missing data patterns"""

    group_values = md_patterns

    heat_values = group_values.iloc[
        0 : (group_values.shape[0] - 1), 1 : group_values.shape[1] - 1
    ]

    myred = "#B61A51B3"
    myblue = "#006CC2B3"
    cmap = colors.ListedColormap([myred, myblue])

    fig, ax = plt.subplots(1)
    ax.imshow(heat_values.astype(bool), aspect="auto", cmap=cmap)

    by = ax.twinx()  # right ax
    bx = ax.twiny()  # top ax

    ax.set_yticks(np.arange(0, len(heat_values.index), 1))
    ax.set_yticklabels(
        group_values.iloc[0 : (group_values.shape[0] - 1), 0]
    )  # first column
    ax.set_yticks(np.arange(-0.5, len(heat_values.index), 1), minor=True)

    ax.set_xticks(np.arange(0, len(heat_values.columns), 1))
    ax.set_xticklabels(
        group_values.iloc[
            group_values.shape[0] - 1, 1 : (group_values.shape[1] - 1)
        ]
    )  # last row
    ax.set_xticks(np.arange(-0.5, len(heat_values.columns), 1), minor=True)

    by.set_yticks(np.arange(0, (len(heat_values.index) * 2) + 1, 1))
    right_ticklabels = list(
        group_values.iloc[
            0 : (group_values.shape[0] - 1), group_values.shape[1] - 1
        ]
    )  # last column
    by_ticklabels = [""] * (len(right_ticklabels) * 2 + 1)
    by_ticklabels[1::2] = right_ticklabels
    by.set_yticklabels(by_ticklabels, fontsize=10)

    bx.set_xticks(np.arange(0, (len(heat_values.columns) * 2) + 1, 1))
    top_ticklabels = list(heat_values.columns)
    bx_ticklabels = [""] * (len(top_ticklabels) * 2 + 1)
    bx_ticklabels[1::2] = top_ticklabels
    bx.set_xticklabels(bx_ticklabels, fontsize=10)

    by.invert_yaxis()
    by.autoscale(False)

    ax.grid(which="minor", color="w", linewidth=1)

    plt.show()
    return fig

sns.set(style='whitegrid', context='talk')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data_qcd.csv"), index_col=0, header=0)
df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
no_2yfu = df[df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
df = df.drop(no_2yfu, axis=0)

deltasmri_complete = pd.concat([df.filter(regex='smri.*change_score'), 
                                #df.filter(regex='mrisdp.*change_score')
                                ], axis=1)
deltarsfmri_complete = df.filter(regex='rsfmri.*change_score')
deltarsi_complete = df.filter(regex='dmri_rsi.*change_score')
deltadti_complete = df.filter(regex='dmri_dti.*change_score')

qc_vars = pd.concat([df.filter(regex='imgincl_.*mri.*'), df.filter(regex='mri_info_manufacturer.*')], axis=1)
#print(qc_vars.columns)

df = df.drop(df.filter(regex='.*mri_.*arm_1').columns, axis=1)
#print(df.columns)
df = pd.concat([df, qc_vars], axis=1)

# subset by atlas for smri and rsfmri (variance) data
smri_atlas = {'cdk': [], 
              'mrisdp': [],
              #'cf12': []
             }
rsfmri_atlas = {'cdk': [],
                'cortgordon': []
               }

for atlas in smri_atlas.keys():
    smri_atlas[atlas] = list(deltasmri_complete.filter(regex=f'{atlas}.*').columns)

for atlas in rsfmri_atlas.keys():
    rsfmri_atlas[atlas] = list(deltarsfmri_complete.filter(regex=f'{atlas}.*').columns)

smri_scs = list(deltasmri_complete.filter(regex='.*vol_scs_.*').columns)
rsfmri_scs = list(deltarsfmri_complete.filter(regex='.*_scs_.*').columns)

# build data subsets for clustering
subcort = smri_scs + rsfmri_scs
cdk_columns = list(deltasmri_complete.columns) + list(deltarsfmri_complete.columns) + list(deltarsi_complete.columns) + list(deltadti_complete.columns) + subcort
cdk_data = df.filter(cdk_columns)

# calculate missingness per variable
missing = pd.Series(data=df.isna().sum() / len(df.index), index=df.columns)
missing.to_csv(join(PROJ_DIR, 'data', 'missingness.csv'))

less_than_tenth_missing = []
for variable in missing.index:
    if missing[variable] < 0.1:
        less_than_tenth_missing.append(variable)

# evaluate missingness patterns with an abbreviated dataset
minimal_miss = ["imgincl_dmri_include",
        "imgincl_rsfmri_include",
        "imgincl_t1w_include","interview_age","nihtbx_picvocab_uncorrected",
        "nihtbx_flanker_uncorrected",
        "nihtbx_list_uncorrected",
        "nihtbx_cardsort_uncorrected",
        "nihtbx_pattern_uncorrected",
        "nihtbx_picture_uncorrected",
        "nihtbx_reading_uncorrected","sex",
        "mri_info_manufacturer",
        "interview_date","demo_prnt_ethn_v2",
        "demo_prnt_marital_v2",
        "demo_prnt_ed_v2",
        "demo_comb_income_v2","race_ethnicity","pds_p_ss_female_category_2", 
        "pds_p_ss_male_category_2", "cbcl_scr_syn_totprob_r"]

minimal = [f'{x}.baseline_year_1_arm_1' for x in minimal_miss ] + [f'{x}.2_year_follow_up_y_arm_1' for x in minimal_miss ]
miss_df = df[minimal]

#patterning = mdPatterns()
patterns = _calculate_patterns(miss_df, 'proportion')
patterns.to_csv(join(PROJ_DIR, 'data', 'missing_patterns.csv'))

#fig = _make_plot(patterns)
#fig.savefig(join(PROJ_DIR, 'figures', 'missing_patterns.png'), dpi=400)

# perform Little's test 
#little = MCARTest(method='little')
#little_test = little.little_mcar_test(df)
#print(f'Little\'s test for MCAR pvalues = {little_test}')

#little_pairs = MCARTest(method='ttest')
#little_ttests = little_pairs.mcar_t_tests(df)
#little_ttests.to_csv(join(PROJ_DIR, 'data', f'littles_ttests.csv'))

#if little_test >= 0.05:
#    mcar = True
#not_mcar_vars = []
#for variable in less_than_tenth_missing:
#   if variable in cdk_columns and little_ttests[variable].max() > 0.05:
#        not_mcar_vars.append(variable)

#print(f'{len(not_mcar_vars) / len(cdk_columns) * 100}% of brain changes are not mcar')
#dcg_columns = smri_atlas['mrisdp'] + rsfmri_atlas['cortgordon'] + list(deltarsi_complete.columns) + list(deltadti_complete.columns) + subcort
#dcg_data = df.filter(dcg_columns)

#get_ipython().run_line_magic('timeit', '')
# let's impute some missing values, heyyyyy
# I should prob nest within sites and/or families??
# not sure that's possible, pal

# there's a preprint that says imputing before CV is fine.
# https://doi.org/10.48550/arXiv.2010.00718
print('time for mice!\t', datetime.now())
imp = KNNImputer()
cdk_data_complete = imp.fit_transform(cdk_data)
print('finished mice!\t', datetime.now())

#cdk_data.iloc[:200]
#dcg_data_complete = mice(dcg_data.values)

imputed_cdk = pd.DataFrame(data=cdk_data_complete, 
                           columns=cdk_data.columns, 
                           index=cdk_data.index)

#imputed_dcg = pd.DataFrame(data=dcg_data_complete, 
#                           columns=dcg_data.columns, 
#                           index=dcg_data.index)

imputed_cdk.describe()
#imputed_dcg.describe()

other_vars = list(set(df.columns) - set(imputed_cdk.columns))
big_df = pd.concat([df[other_vars], imputed_cdk], axis=1)
big_nomice = pd.concat([df[other_vars], cdk_data], axis=1)
big_df.to_csv(join(PROJ_DIR, 'data', 'data_qcd_knn-cdk.csv'))
big_nomice.to_csv(join(PROJ_DIR, 'data', 'data_qcd_NOmice-cdk.csv'))

imputed_cdk.to_csv(join(join(PROJ_DIR, 'data', "data_cdk-knn.csv")))
#imputed_dcg.to_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data.csv")))
