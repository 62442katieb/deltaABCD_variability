#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

from os.path import join

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data.csv"), index_col=0, header=0)
df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
no_2yfu = df[df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
df = df.drop(no_2yfu, axis=0)

qc_vars = ["imgincl_dmri_include",
           "imgincl_rsfmri_include",
           "rsfmri_c_ngd_meanmotion",
           "rsfmri_c_ngd_ntpoints",
           "imgincl_t1w_include",
           "mrif_score"]

keep = df[df['mrif_score.baseline_year_1_arm_1'].between(1,2)].index
df = df.loc[keep]

# modality-specific filtering via masks
rsfmri_mask1 = df['imgincl_rsfmri_include.baseline_year_1_arm_1'] == 0
rsfmri_mask2 = df['rsfmri_var_ntpoints.baseline_year_1_arm_1'] <= 750.
rsfmri_mask3 = df['imgincl_rsfmri_include.2_year_follow_up_y_arm_1'] == 0
rsfmri_mask4 = df['rsfmri_var_ntpoints.2_year_follow_up_y_arm_1'] <= 750.
rsfmri_mask = rsfmri_mask1 * rsfmri_mask2 * rsfmri_mask3 * rsfmri_mask4

smri_mask1 = df['imgincl_t1w_include.baseline_year_1_arm_1'] == 0
smri_mask2 = df['imgincl_t1w_include.2_year_follow_up_y_arm_1'] == 0
smri_mask = smri_mask1 * smri_mask2

dmri_mask1 = df['imgincl_dmri_include.baseline_year_1_arm_1'] == 0
dmri_mask2 = df['imgincl_dmri_include.2_year_follow_up_y_arm_1'] == 0
dmri_mask = dmri_mask1 * dmri_mask2

smri_cols = list(df.filter(regex='smri.').columns) + list(df.filter(regex='mrisdp.').columns)
rsfmri_cols = df.filter(regex='rsfmri.').columns
dmri_cols = df.filter(regex='dmri').columns
other_cols = set(df.columns) - set(smri_cols) - set(rsfmri_cols) - set(dmri_cols)

rsfmri_quality = df[rsfmri_cols].mask(rsfmri_mask)
smri_quality = df[smri_cols].mask(smri_mask)
dmri_quality = df[dmri_cols].mask(dmri_mask)
other = df[other_cols]

# after filtering out radiological abnormalities with mrif_score
# apply modality-specific filters

# up first: rsfmri
quality_df = pd.concat([other, rsfmri_quality, smri_quality, dmri_quality], axis=1)

quality_df.to_csv(join(PROJ_DIR, DATA_DIR, "data.csv"))

demographics = ["demo_prnt_marital_v2",
                "demo_prnt_ed_v2",
                "demo_comb_income_v2",
                "race_ethnicity",
                "site_id_l",
                "sex", 
                "mri_info_manufacturer"
               ]

mri_qc = [
    "imgincl_dmri_include",
    "imgincl_rsfmri_include",
    "imgincl_t1w_include",
    #"imgincl_t2w_include",
    "mrif_score",
    "interview_age",
    "interview_date"
]


demo_and_qc = []
for var in demographics + mri_qc:
    demo_and_qc.append(f'{var}.baseline_year_1_arm_1')
    if var in mri_qc:
        demo_and_qc.append(f'{var}.2_year_follow_up_y_arm_1')
    else:
        pass

demo_df = df[demo_and_qc]
df = None

qc_df = pd.read_csv(join(PROJ_DIR, DATA_DIR, 'data_qcd.csv'), 
                 header=0, 
                 index_col='subjectkey')
qc_ppts = qc_df.dropna(how='all').index
qc_df = None


no_2yfu = demo_df[demo_df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
lost_N = len(no_2yfu)
total_N = len(demo_df.index)

print(f"Of the total {total_N} participants at baseline, {lost_N} (or {np.round((lost_N / total_N) *100, 2)}%) did not have a 2-year follow-up imaging appointment and were, thus, excluded from further analyses.")

rsfmri_quality = demo_df.loc[rsfmri_quality.dropna().index]
smri_quality = demo_df.loc[smri_quality.dropna().index]
dmri_quality = demo_df.loc[dmri_quality.dropna().index]

table = pd.DataFrame(index=['N', 
                            'Age_mean_base',
                            'Age_sdev_base',
                            'Age_mean_2yfu',
                            'Age_sdev_2yfu',
                            'Sex_M', 
                            'Sex_F', 
                            'RE_Black',
                            'RE_White',
                            'RE_Hispanic',
                            'RE_AsianOther',
                            'Income_gt100k', 
                            'Income_50to100k', 
                            'Income_lt50k',
                            'Income_dkrefuse',
                            'Marital_Married',
                            'Marital_Widowed',
                            'Marital_Divorced',
                            'Marital_Separated',
                            'Marital_Never',
                            'Marital_Refused',
                            'Education_uptoHSGED',
                            'Education_SomeColAA',
                            'Education_Bachelors',
                            'Education_Graduate',
                            'MRI_Siemens', 
                            'MRI_GE', 
                            'MRI_Philips'], 
                     columns=['whole_sample', 'with_t1', 'with_dmri', 'with_fmri'])

table.at['N', 'whole_sample'] = len(demo_df.index)
table.at['N', 'with_t1'] = len(smri_quality.index)
table.at['N', 'with_dmri'] = len(dmri_quality.index)
table.at['N', 'with_fmri'] = len(rsfmri_quality.index)

table.at['Age_mean_base', 'whole_sample'] = np.mean(demo_df['interview_age.baseline_year_1_arm_1'])
table.at['Age_mean_base', 'with_t1'] = np.mean(smri_quality['interview_age.baseline_year_1_arm_1'])
table.at['Age_mean_base', 'with_dmri'] = np.mean(dmri_quality['interview_age.baseline_year_1_arm_1'])
table.at['Age_mean_base', 'with_rsfmri'] = np.mean(rsfmri_quality['interview_age.baseline_year_1_arm_1'])

table.at['Age_mean_2yfu', 'whole_sample'] = np.mean(demo_df['interview_age.2_year_follow_up_y_arm_1'])
table.at['Age_mean_2yfu', 'with_t1'] = np.mean(smri_quality['interview_age.2_year_follow_up_y_arm_1'])
table.at['Age_mean_2yfu', 'with_dmri'] = np.mean(dmri_quality['interview_age.2_year_follow_up_y_arm_1'])
table.at['Age_mean_2yfu', 'with_rsfmri'] = np.mean(rsfmri_quality['interview_age.2_year_follow_up_y_arm_1'])

table.at['Age_sdev_base', 'whole_sample'] = np.std(demo_df['interview_age.baseline_year_1_arm_1'])
table.at['Age_sdev_base', 'with_t1'] = np.std(smri_quality['interview_age.baseline_year_1_arm_1'])
table.at['Age_sdev_base', 'with_dmri'] = np.std(dmri_quality['interview_age.baseline_year_1_arm_1'])
table.at['Age_sdev_base', 'with_rsfmri'] = np.std(rsfmri_quality['interview_age.baseline_year_1_arm_1'])

table.at['Age_sdev_2yfu', 'whole_sample'] = np.std(demo_df['interview_age.2_year_follow_up_y_arm_1'])
table.at['Age_sdev_2yfu', 'with_t1'] = np.std(smri_quality['interview_age.2_year_follow_up_y_arm_1'])
table.at['Age_sdev_2yfu', 'with_dmri'] = np.std(dmri_quality['interview_age.2_year_follow_up_y_arm_1'])
table.at['Age_sdev_2yfu', 'with_rsfmri'] = np.std(rsfmri_quality['interview_age.2_year_follow_up_y_arm_1'])

table.at['Sex_M', 'whole_sample'] = len(demo_df[demo_df['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Sex_M', 'with_t1'] = len(smri_quality[smri_quality['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Sex_M', 'with_dmri'] = len(dmri_quality[dmri_quality['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Sex_M', 'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['sex.baseline_year_1_arm_1'] == 'M'].index)

table.at['Sex_F', 'whole_sample'] = len(demo_df[demo_df['sex.baseline_year_1_arm_1'] == 'F'].index)
table.at['Sex_F', 'with_t1'] = len(smri_quality[smri_quality['sex.baseline_year_1_arm_1'] == 'F'].index)
table.at['Sex_F', 'with_dmri'] = len(dmri_quality[dmri_quality['sex.baseline_year_1_arm_1'] == 'F'].index)
table.at['Sex_F', 'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['sex.baseline_year_1_arm_1'] == 'F'].index)


table.at['RE_White', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
table.at['RE_White', 
         'with_t1'] = len(smri_quality[smri_quality['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
table.at['RE_White', 
         'with_dmri'] = len(dmri_quality[dmri_quality['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
table.at['RE_White', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)

table.at['RE_Black', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)
table.at['RE_Black', 
         'with_t1'] = len(smri_quality[smri_quality['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)
table.at['RE_Black', 
         'with_dmri'] = len(dmri_quality[dmri_quality['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)
table.at['RE_Black', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)

table.at['RE_Hispanic', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)
table.at['RE_Hispanic', 
         'with_t1'] = len(smri_quality[smri_quality['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)
table.at['RE_Hispanic', 
         'with_dmri'] = len(dmri_quality[dmri_quality['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)
table.at['RE_Hispanic', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)

table.at['RE_AsianOther', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)
table.at['RE_AsianOther', 
         'with_t1'] = len(smri_quality[smri_quality['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)
table.at['RE_AsianOther', 
         'with_dmri'] = len(dmri_quality[dmri_quality['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)
table.at['RE_AsianOther', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)


table.at['Income_gt100k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)
table.at['Income_gt100k', 
         'with_t1'] = len(smri_quality[smri_quality['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)
table.at['Income_gt100k', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)
table.at['Income_gt100k', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)

table.at['Income_50to100k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)
table.at['Income_50to100k', 
         'with_t1'] = len(smri_quality[smri_quality['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)

table.at['Income_lt50k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)
table.at['Income_lt50k', 
         'with_t1'] = len(smri_quality[smri_quality['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)

table.at['Income_dkrefuse', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)
table.at['Income_dkrefuse', 
         'with_t1'] = len(smri_quality[smri_quality['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)

table.at['MRI_Siemens', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "SIEMENS"].index)
table.at['MRI_Siemens', 
         'with_t1'] = len(smri_quality[smri_quality['mri_info_manufacturer.baseline_year_1_arm_1'] == "SIEMENS"].index)
table.at['MRI_GE', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_GE', 
         'with_t1'] = len(smri_quality[smri_quality['mri_info_manufacturer.baseline_year_1_arm_1']  == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_Philips', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "Philips Medical Systems"].index)
table.at['MRI_Philips', 
         'with_t1'] = len(smri_quality[smri_quality['mri_info_manufacturer.baseline_year_1_arm_1'] == "Philips Medical Systems"].index)

table.at['Marital_Married', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])
table.at['Marital_Married', 
         'with_t1'] = len(smri_quality[smri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])
table.at['Marital_Widowed', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])
table.at['Marital_Widowed', 
         'with_t1'] = len(smri_quality[smri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])
table.at['Marital_Divorced', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])
table.at['Marital_Divorced', 
         'with_t1'] = len(smri_quality[smri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])
table.at['Marital_Separated', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])
table.at['Marital_Separated', 
         'with_t1'] = len(smri_quality[smri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])
table.at['Marital_Never', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])
table.at['Marital_Never', 
         'with_t1'] = len(smri_quality[smri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])
table.at['Marital_Refused', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])
table.at['Marital_Refused', 
         'with_t1'] = len(smri_quality[smri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])

table.at['Education_uptoHSGED', 
         'whole_sample'] = len(demo_df[demo_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'whole_sample'] = len(demo_df[demo_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'whole_sample'] = len(demo_df[demo_df['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
table.at['Education_Graduate', 
         'whole_sample'] = len(demo_df[demo_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, 
                                                                                                inclusive='both')])
table.at['Education_uptoHSGED', 
         'with_t1'] = len(smri_quality[smri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'with_t1'] = len(smri_quality[smri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'with_t1'] = len(smri_quality[smri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
table.at['Education_Graduate', 
         'with_t1'] = len(smri_quality[smri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, 
                                                                                                inclusive='both')])

table.at['Income_50to100k', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)
table.at['Income_50to100k', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)

table.at['Income_lt50k', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)
table.at['Income_lt50k', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)

table.at['Income_dkrefuse', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)
table.at['Income_dkrefuse', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)

table.at['MRI_Siemens', 
         'with_dmri'] = len(dmri_quality[dmri_quality['mri_info_manufacturer.baseline_year_1_arm_1'] == "SIEMENS"].index)
table.at['MRI_Siemens', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['mri_info_manufacturer.baseline_year_1_arm_1'] == "SIEMENS"].index)
table.at['MRI_GE', 
         'with_dmri'] = len(dmri_quality[dmri_quality['mri_info_manufacturer.baseline_year_1_arm_1'] == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_GE', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['mri_info_manufacturer.baseline_year_1_arm_1']  == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_Philips', 
         'with_dmri'] = len(dmri_quality[dmri_quality['mri_info_manufacturer.baseline_year_1_arm_1'] == "Philips Medical Systems"].index)
table.at['MRI_Philips', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['mri_info_manufacturer.baseline_year_1_arm_1'] == "Philips Medical Systems"].index)

table.at['Marital_Married', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])
table.at['Marital_Married', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])
table.at['Marital_Widowed', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])
table.at['Marital_Widowed', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])
table.at['Marital_Divorced', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])
table.at['Marital_Divorced', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])
table.at['Marital_Separated', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])
table.at['Marital_Separated', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])
table.at['Marital_Never', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])
table.at['Marital_Never', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])
table.at['Marital_Refused', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])
table.at['Marital_Refused', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])

table.at['Education_uptoHSGED', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
table.at['Education_Graduate', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, 
                                                                                                inclusive='both')])
table.at['Education_uptoHSGED', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
table.at['Education_Graduate', 
         'with_rsfmri'] = len(rsfmri_quality[rsfmri_quality['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, 
                                                                                                inclusive='both')])
table.to_csv(f'{PROJ_DIR}/output/demographics_by_modality.csv')