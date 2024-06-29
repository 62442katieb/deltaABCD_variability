#!/usr/bin/env python
# coding: utf-8
import pyreadr

import pandas as pd
import numpy as np
#import seaborn as sns

from os.path import join
from sklearn.ensemble import IsolationForest

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

ABCD_DIR = '/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.0/'
demo_df = pyreadr.read_r(join(ABCD_DIR, '01_Demographics/ABCD_5.0_demographics_concise_final.RDS'))
demo_df = demo_df[None]
demo_df.index = pd.MultiIndex.from_arrays([demo_df['src_subject_id'], demo_df['eventname']])
demo_df = demo_df.drop(['src_subject_id', 'eventname'], axis=1)

baseline = demo_df.xs('baseline_year_1_arm_1', level=1)[['interview_age', 'site_id_l', 
                                                        'race_ethnicity_c_bl', 'demo_sex_v2_bl',
       'household_income_4bins_bl', 'highest_parent_educ_bl', 'rel_family_id_bl']]
year_two = demo_df.xs('2_year_follow_up_y_arm_1', level=1)['interview_age']
year_two.name = 'interview_age.2_year_follow_up_y_arm_1'

demo_df = pd.concat(
    [
        baseline.rename({'interview_age': 'interview_age.baseline_year_1_arm_1'}, axis=1),
        year_two
    ],
    axis=1
)

demo_df = demo_df.replace(
    {
        "[<50K]": "<50k",
        "[≥50K and <100K]": "50k_100k",
        "[≥100K]": ">100k"
    }
)

demo_df = demo_df.replace(
    {
        'Missing/Refused': np.nan
    }
)

pbty_df = pyreadr.read_r(join(ABCD_DIR, '04_Physical_Health/ABCD_5.0_Physical_Health.RDS'))
pbty_df = pbty_df[None]
pbty_df.index = pbty_df['src_subject_id']
pbty_df = pbty_df.drop(['src_subject_id'], axis=1)
pbty_df1 = pbty_df.replace(
    {
        'Pre Puberty': 1,
        'Early Puberty': 2,
        'Mid Puberty': 3,
        'Late Puberty': 4,
        'Post Puberty': 5
    }
)

pbty_df1 = pd.concat(
    [
        pbty_df1[pbty_df1['eventname'] == 'baseline_year_1_arm_1']['Puberty_Stage'],
        pbty_df1[pbty_df1['eventname'] == '1_year_follow_up_y_arm_1']['Puberty_Stage'],
        pbty_df1[pbty_df1['eventname'] == '2_year_follow_up_y_arm_1']['Puberty_Stage']
    ],
    axis=1
)
pbty_df1.columns = ['PDS.baseline_year_1_arm_1', 'PDS.1_year_follow_up_y_arm_1', 'PDS.2_year_follow_up_y_arm_1']

for i in pbty_df1.index:
    bln = float(pbty_df1.loc[i]['PDS.baseline_year_1_arm_1'])
    one = float(pbty_df1.loc[i]['PDS.1_year_follow_up_y_arm_1'])
    two = float(pbty_df1.loc[i]['PDS.2_year_follow_up_y_arm_1'])
    time_diff = (demo_df.loc[i]['interview_age.2_year_follow_up_y_arm_1'] - demo_df.loc[i]['interview_age.baseline_year_1_arm_1']) / 12.
    if one is not np.nan:
        change_1 = one - bln
        change_2 = two - one
        pbty_df1.at[i, 'delta_Puberty'] = (change_1 + change_2) / time_diff
    else:
        change = two - bln
        pbty_df1.at[i, 'delta_Puberty'] = change / time_diff

pbty_df = pd.concat(
    [
        pbty_df[pbty_df['eventname'] == 'baseline_year_1_arm_1']['Puberty_Stage'],
        pbty_df[pbty_df['eventname'] == '1_year_follow_up_y_arm_1']['Puberty_Stage'],
        pbty_df[pbty_df['eventname'] == '2_year_follow_up_y_arm_1']['Puberty_Stage']
    ],
    axis=1
)
pbty_df.columns = ['PDS.baseline_year_1_arm_1', 'PDS.1_year_follow_up_y_arm_1', 'PDS.2_year_follow_up_y_arm_1']

pbty_df = pbty_df.replace(
    {
        'Post Puberty': 'Late/Post Puberty',
        'Late Puberty': 'Late/Post Puberty',
    }
)

ppt_lists = pd.read_csv(join(PROJ_DIR, OUTP_DIR, 'ppts_qc.csv'), index_col=0)
rci = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_rci.pkl'))
apd = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_change_scores.pkl'))
qcd = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd.pkl'))

demographics = [
    "highest_parent_educ_bl",
    "household_income_4bins_bl",
    "race_ethnicity_c_bl",
    "demo_sex_v2_bl",
    "PDS.baseline_year_1_arm_1",
    #"PDS.2_year_follow_up_y_arm_1"
]

col_to_df = {
    'whole_sample': ppt_lists.index, 
    'pre_covid': ppt_lists[ppt_lists['Pre-COVID'] == 1].index,
    'post_covid': list(set(ppt_lists.index) - set(ppt_lists[ppt_lists['Pre-COVID'] == 1].index)),
    'siemens': ppt_lists[ppt_lists['SIEMENS'] == 1].index,
    'left_out_siemens': list(set(ppt_lists.index) - set(ppt_lists[ppt_lists['SIEMENS'] == 1].index)),
    'smri_qc': ppt_lists[ppt_lists['sMRI QC'] == 1].index,
    'dmri_qc': ppt_lists[ppt_lists['dMRI QC'] == 1].index,
    'fmri_qc': ppt_lists[ppt_lists['fMRI QC'] == 1].index,
    'left_out_smri': list(set(ppt_lists.index) - set(ppt_lists[ppt_lists['sMRI QC'] == 1].index)),
    'left_out_dmri': list(set(ppt_lists.index) - set(ppt_lists[ppt_lists['dMRI QC'] == 1].index)),
    'left_out_fmri': list(set(ppt_lists.index) - set(ppt_lists[ppt_lists['fMRI QC'] == 1].index)),
}

table = pd.DataFrame(index=['N', 
                            'Age_mean_base',
                            'Age_sdev_base',
                            'Age_Missing',
                            'Elapsed_Time_mean',
                            'Elapsed_Time_sdev',
                            'Male', 
                            'Female', 
                            'Sex_Missing',
                            'Puberty_Change_mean',
                            'Puberty_Change_sdev',
                            'Puberty_Change_Missing',
                            'Puberty0_Missing',
                            'Puberty1_Missing',
                            'Puberty2_Missing',
                            'White',
                            'Hispanic',
                            'Black',
                            'Asian/Other',
                            'Race_Missing',
                            '>100k', 
                            '50k_100k', 
                            '<50k',
                            'Don\'t know/Refuse to answer',
                            'Income_Missing',
                            '< HS Diploma',
                            'HS Diploma/GED',
                            'Some College',
                            'Bachelor Degree',
                            'Post Graduate Degree',
                            'Education_Missing',
                            ], 
                     columns=list(col_to_df.keys()))

for subset in col_to_df.keys():
     ppts = col_to_df[subset]
     temp_df = demo_df.loc[ppts]
     temp_2 = pbty_df.loc[ppts]
     temp_3 = pbty_df1.loc[ppts]['delta_Puberty']
     temp_df = pd.concat([temp_df, temp_2, temp_3], axis=1)
     table.at['N', subset] = len(ppts)
     table.at['Age_mean_base', subset] = temp_df['interview_age.baseline_year_1_arm_1'].mean()
     table.at['Age_sdev_base', subset] = temp_df['interview_age.baseline_year_1_arm_1'].std()
     elapsed = temp_df['interview_age.2_year_follow_up_y_arm_1'] - temp_df['interview_age.baseline_year_1_arm_1']
     table.at['Elapsed_Time_mean', subset] = elapsed.mean()
     table.at['Elapsed_Time_sdev', subset] = elapsed.std()
     table.at['Sex_Missing', subset] = temp_df['demo_sex_v2_bl'].isna().sum()
     table.at['Puberty_Change_mean', subset] = temp_df['delta_Puberty'].mean()
     table.at['Puberty_Change_sdev', subset] = temp_df['delta_Puberty'].std()
     table.at['Puberty_Change_Missing', subset] = temp_df['delta_Puberty'].isna().sum()
     table.at['Puberty0_Missing', subset] = temp_df['PDS.baseline_year_1_arm_1'].isna().sum()
     table.at['Puberty1_Missing', subset] = temp_df['PDS.1_year_follow_up_y_arm_1'].isna().sum()
     table.at['Puberty2_Missing', subset] = temp_df['PDS.2_year_follow_up_y_arm_1'].isna().sum()
     table.at['Race_Missing', subset] = temp_df['race_ethnicity_c_bl'].isna().sum()
     table.at['Income_Missing', subset] = temp_df['household_income_4bins_bl'].isna().sum()
     table.at['Education_Missing', subset] = temp_df['highest_parent_educ_bl'].isna().sum()
     for col in demographics:
        counts = temp_df[col].value_counts()
        for level in counts.index:
            table.at[level,subset] = counts[level]

table.to_csv(join(PROJ_DIR, OUTP_DIR, 'deltaSA-demographics.csv'))

pd.concat(
    [
        demo_df,
        pbty_df,
        pbty_df1['delta_Puberty']
    ],
    axis=1
).to_csv(join(PROJ_DIR, DATA_DIR, 'data_covar.csv'))


pd.concat(
    [
        demo_df,
        pbty_df,
        pbty_df1['delta_Puberty']
    ],
    axis=1
).to_pickle(join(PROJ_DIR, DATA_DIR, 'data_covar.pkl'))
