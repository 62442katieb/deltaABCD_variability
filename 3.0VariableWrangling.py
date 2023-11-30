#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pingouin as pg
import math

from os.path import join
import warnings

warnings.filterwarnings("ignore")

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


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))
#ppts = df.filter(regex="rsfmri_c_.*change_score").dropna().index
df = df.drop(df.filter(regex='.*change_score', axis=1).columns, axis=1)


covariates = {
    'thk': {
        'vlike': "smri_thick_cdk_",
        'covar': []
    },
    'rnd': {
        'vlike': "dmri_rsirndgm_cdk_",
        'covar': ['dmri_rsi_meanmotion', 
                  'dmri_rsi_meanrot', 
                  'dmri_rsi_meantrans',
                  ]
    },
    'rni': {
        'vlike': "dmri_rsirnigm_cdk_",
        'covar': ['dmri_rsi_meanmotion', 
                  'dmri_rsi_meanrot', 
                  'dmri_rsi_meantrans',
                  ]
    },
    'var': {
        'vlike': "rsfmri_var_cdk_",
        'covar': ['rsfmri_var_meanmotion', 
                  'rsfmri_var_meanrot', 
                  'rsfmri_var_meantrans'
                  ]
    }
}
keep = []
for measure in covariates.keys():
    keep.append(df.filter(like=covariates[measure]['vlike'], axis=1))

df = pd.concat(keep, axis=1)

mri_df = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/csv/abcd_mri01.csv", 
    skiprows=[1],
    index_col=['subjectkey', 
             'eventname'],
    usecols=['subjectkey', 
    'eventname', 
    'mri_info_deviceserialnumber',
    'mri_info_manufacturer',
    'interview_age',
    'sex']

)
handedness = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/csv/abcd_ehis01.csv", 
    skiprows=[1],
    index_col=['subjectkey', 
             'eventname'],
    usecols=['subjectkey',
             'eventname',
             'ehi_y_ss_scoreb']
)
handedness = handedness.swaplevel(axis=0).loc['baseline_year_1_arm_1']
age2 = mri_df.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1']['interview_age']
age2.name = 'interview_age_2'
mri_df = mri_df.swaplevel(axis=0).loc['baseline_year_1_arm_1']
ppts = mri_df[mri_df['mri_info_manufacturer'] == 'SIEMENS'].index
mri_df['sex'] = pd.get_dummies(mri_df['sex'])['F']
scanners = pd.get_dummies(mri_df.loc[ppts]['mri_info_deviceserialnumber'])

mri_df = pd.concat(
    [
        mri_df.drop('mri_info_deviceserialnumber', axis=1),
        scanners,
        age2,
        handedness
    ],
    axis=1
)
all_covars = list(scanners.columns) + ['sex', 'ehi_y_ss_scoreb']

# need to load in 4.0 vars that include head motion and brain volume

smri_vol = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/csv/abcd_smrip10201.csv",
    index_col=['subjectkey', 'eventname'],
    skiprows=[1],
    usecols=['subjectkey', 
             'eventname', 
             'smri_vol_scs_wholeb',]
)

base = smri_vol.swaplevel(axis=0).loc['baseline_year_1_arm_1']
base.columns = [f'{col}.baseline_year_1_arm_1' for col in base.columns]
y2fu = smri_vol.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1']
y2fu.columns = [f'{col}.2_year_follow_up_y_arm_1' for col in y2fu.columns]

df = pd.concat([df, mri_df, base, y2fu], axis=1)

rsi_motion = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/csv/abcd_drsip101.csv",
    index_col=['subjectkey', 'eventname'],
    skiprows=[1],
    usecols=['subjectkey', 
             'eventname', 
             'dmri_rsi_meanmotion',
             'dmri_rsi_meanrot', 
             'dmri_rsi_meantrans',]
)

base = rsi_motion.swaplevel(axis=0).loc['baseline_year_1_arm_1']
base.columns = [f'{col}.baseline_year_1_arm_1' for col in base.columns]
y2fu = rsi_motion.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1']
y2fu.columns = [f'{col}.2_year_follow_up_y_arm_1' for col in y2fu.columns]

df = pd.concat([df, base, y2fu], axis=1)

rsfmri_motion = pd.read_csv(
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/csv/abcd_mrirstv02.csv",
    index_col=['subjectkey', 'eventname'],
    skiprows=[1],
    usecols=['subjectkey', 
             'eventname', 
             'rsfmri_var_meanmotion',
             'rsfmri_var_meanrot', 
             'rsfmri_var_meantrans',
             'rsfmri_var_subthreshnvols',]
)


base = rsfmri_motion.swaplevel(axis=0).loc['baseline_year_1_arm_1']
base.columns = [f'{col}.baseline_year_1_arm_1' for col in base.columns]
y2fu = rsfmri_motion.swaplevel(axis=0).loc['2_year_follow_up_y_arm_1']
y2fu.columns = [f'{col}.2_year_follow_up_y_arm_1' for col in y2fu.columns]

df = pd.concat([df, base, y2fu], axis=1)
# only SIEMENS data from here on out
df = df.loc[ppts]
df.to_pickle(join(PROJ_DIR, DATA_DIR, 'data_covar_qcd.pkl'))


#halp
tpts = [
    'baseline_year_1_arm_1',
    '2_year_follow_up_y_arm_1'
]

change_scores = pd.DataFrame(dtype=float)
rci = pd.DataFrame(dtype=float)
for measure in list(covariates.keys()):
    print(measure)
    #motion_df = covariates[measure]['cv_df']
    like = covariates[measure]['vlike']
    covar = covariates[measure]['covar']
    
    temp1 = df.filter(regex=f'{like}.*baseline_year_1_arm_1')
    temp1.columns = [i.split('.')[0] for i in temp1.columns]
    #print(temp1.head())
    cov_base = [f'{i}.baseline_year_1_arm_1' for i in covar]
    temp2 = df[cov_base + all_covars]
    temp2.columns = [i.split('.')[0] for i in temp2.columns]
    #temp2.columns = [f'{col}.baseline_year_1_arm_1' for col in temp2.columns]
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    confounds = covar + all_covars
    base_resid = residualize(temp.drop(confounds, axis=1).values, confounds=temp[confounds].values)
    base_resid = pd.DataFrame(data=base_resid, index=temp.index, columns=temp.drop(confounds, axis=1).columns)
    
    # now repeat for year-2 follow-up
    temp1 = df.filter(regex=f'{like}.*2_year_follow_up_y_arm_1')
    temp1.columns = [i.split('.')[0] for i in temp1.columns]
    cov_y2fu = [f'{i}.2_year_follow_up_y_arm_1' for i in covar]
    temp2 = df[cov_y2fu + all_covars]
    temp2.columns = [i.split('.')[0] for i in temp2.columns]
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    y2fu_resid = residualize(temp.drop(confounds, axis=1).values, confounds=temp[confounds].values)
    y2fu_resid = pd.DataFrame(data=y2fu_resid, index=temp.index, columns=temp.drop(confounds, axis=1).columns)
    
    #print(base_resid.head(), y2fu_resid.head())
    # calculate change scores
    for col in base_resid.columns:
        age1 = df['interview_age']
        age2 = df['interview_age_2']
        temp = pd.concat(
            [base_resid[col], y2fu_resid[col], age1, age2], 
            axis=1
        ).dropna()
        temp2 = pd.concat(
            [base_resid[col], y2fu_resid[col]], 
            axis=1
        ).dropna()
        #print(temp.head())
        for i in temp.index:
            # annual percent change
            base = base_resid.loc[i][col]
            y2fu = y2fu_resid.loc[i][col]
            age0y = age1.loc[i]
            age2y = age2.loc[i]
            change_scores.at[i, col] = (((y2fu - base) / np.mean([y2fu, base])) * 100) / (age2y - age0y)
            # and rci
            sem = np.std(temp2.values, ddof=1) / np.sqrt(np.size(temp2.values))
            #abs_sem = np.std(np.abs(temp.values), ddof=1) / np.sqrt(np.size(temp.values))
            
            rci.at[i,col] = (y2fu - base) / sem
change_scores = pd.concat(
    [
        change_scores, 
        df['interview_age'],
        df['interview_age_2']
    ],
    axis=1
)
change_scores.to_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_change_scores.pkl'))

rci = pd.concat(
    [
        rci, 
        df['interview_age'],
        df['interview_age_2']
    ],
    axis=1
)
rci.to_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_rci.pkl'))