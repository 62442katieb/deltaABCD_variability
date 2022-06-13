#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import scipy.stats as sstats 

#from statsmodels.stats import contingency_tables
from os.path import exists, join

PROJ_DIR = '/Volumes/Projects_Herting/LABDOCS/Personnel/Katie/deltaABCD_clustering'
DATA_DIR = 'data'
OUT_DIR = 'output'
FIG_DIR = 'figures'

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, 'data.csv'), 
                 header=0, 
                 index_col='subjectkey')

demographics = [#"demo_prnt_ethn_v2",
                "demo_prnt_marital_v2",
                "demo_prnt_ed_v2",
                "demo_comb_income_v2",
                #"demo_race_a_p___10",
                #"demo_race_a_p___11",
                #"demo_race_a_p___12",
                #"demo_race_a_p___13",
                #"demo_race_a_p___14",
                #"demo_race_a_p___15",
                #"demo_race_a_p___16",
                #"demo_race_a_p___17",
                #"demo_race_a_p___18",
                #"demo_race_a_p___19",
                #"demo_race_a_p___20",
                #"demo_race_a_p___21",
                #"demo_race_a_p___22",
                #"demo_race_a_p___23",
                #"demo_race_a_p___24",
                #"demo_race_a_p___25",
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

y2fu_df = demo_df.loc[qc_ppts]

print(f"Of the total {total_N} participants at baseline, {lost_N} (or {np.round((lost_N / total_N) *100, 2)}%) did not have a 2-year follow-up imaging appointment and were, thus, excluded from further analyses.")

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
                     columns=['whole_sample', 'with_2yfu'])

table.at['N', 'whole_sample'] = len(demo_df.index)
table.at['N', 'with_2yfu'] = len(y2fu_df.index)

table.at['Age_mean_base', 'whole_sample'] = np.mean(demo_df['interview_age.baseline_year_1_arm_1'])
table.at['Age_mean_base', 'with_2yfu'] = np.mean(y2fu_df['interview_age.baseline_year_1_arm_1'])

table.at['Age_mean_2yfu', 'whole_sample'] = np.mean(demo_df['interview_age.2_year_follow_up_y_arm_1'])
table.at['Age_mean_2yfu', 'with_2yfu'] = np.mean(y2fu_df['interview_age.2_year_follow_up_y_arm_1'])

table.at['Age_sdev_base', 'whole_sample'] = np.std(demo_df['interview_age.baseline_year_1_arm_1'])
table.at['Age_sdev_base', 'with_2yfu'] = np.std(y2fu_df['interview_age.baseline_year_1_arm_1'])

table.at['Age_sdev_2yfu', 'whole_sample'] = np.std(demo_df['interview_age.2_year_follow_up_y_arm_1'])
table.at['Age_sdev_2yfu', 'with_2yfu'] = np.std(y2fu_df['interview_age.2_year_follow_up_y_arm_1'])

table.at['Sex_M', 'whole_sample'] = len(demo_df[demo_df['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Sex_M', 'with_2yfu'] = len(y2fu_df[y2fu_df['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Sex_F', 'whole_sample'] = len(demo_df[demo_df['sex.baseline_year_1_arm_1'] == 'F'].index)
table.at['Sex_F', 'with_2yfu'] = len(y2fu_df[y2fu_df['sex.baseline_year_1_arm_1'] == 'F'].index)


table.at['RE_White', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
table.at['RE_White', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
table.at['RE_Black', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)
table.at['RE_Black', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)
table.at['RE_Hispanic', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)
table.at['RE_Hispanic', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)
table.at['RE_AsianOther', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)
table.at['RE_AsianOther', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)


table.at['Income_gt100k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)
table.at['Income_gt100k', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)

table.at['Income_50to100k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)
table.at['Income_50to100k', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)

table.at['Income_lt50k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)
table.at['Income_lt50k', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)

table.at['Income_dkrefuse', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)
table.at['Income_dkrefuse', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)

table.at['MRI_Siemens', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "SIEMENS"].index)
table.at['MRI_Siemens', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "SIEMENS"].index)
table.at['MRI_GE', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_GE', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['mri_info_manufacturer.baseline_year_1_arm_1']  == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_Philips', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "Philips Medical Systems"].index)
table.at['MRI_Philips', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "Philips Medical Systems"].index)

table.at['Marital_Married', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])
table.at['Marital_Married', 
         'with_2yfu'] = len(y2fu_df[y2fu_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])
table.at['Marital_Widowed', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])
table.at['Marital_Widowed', 
         'with_2yfu'] = len(y2fu_df[y2fu_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])
table.at['Marital_Divorced', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])
table.at['Marital_Divorced', 
         'with_2yfu'] = len(y2fu_df[y2fu_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])
table.at['Marital_Separated', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])
table.at['Marital_Separated', 
         'with_2yfu'] = len(y2fu_df[y2fu_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])
table.at['Marital_Never', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])
table.at['Marital_Never', 
         'with_2yfu'] = len(y2fu_df[y2fu_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])
table.at['Marital_Refused', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])
table.at['Marital_Refused', 
         'with_2yfu'] = len(y2fu_df[y2fu_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])

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
         'with_2yfu'] = len(y2fu_df[y2fu_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
table.at['Education_Graduate', 
         'with_2yfu'] = len(y2fu_df[y2fu_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, 
                                                                                                inclusive='both')])

# test for differences in means with wilcoxon signed rank test
stat, pval = sstats.mannwhitneyu(demo_df['interview_age.baseline_year_1_arm_1'].dropna(), 
                                 y2fu_df['interview_age.baseline_year_1_arm_1'].dropna())
table.at['Age_mean_base', 'Stat'] = stat
table.at['Age_mean_base', 'p(Stat)'] = pval

stat, pval = sstats.mannwhitneyu(demo_df['interview_age.2_year_follow_up_y_arm_1'].dropna(), 
                                 y2fu_df['interview_age.2_year_follow_up_y_arm_1'].dropna())
table.at['Age_mean_2yfu', 'Stat'] = stat
table.at['Age_mean_2yfu', 'p(Stat)'] = pval

contingency = np.zeros((2,2))
contingency[0,0] = table.loc['Sex_M', 'whole_sample']
contingency[0,1] = table.loc['Sex_F', 'whole_sample']
contingency[1,0] = table.loc['Sex_M', 'with_2yfu']
contingency[1,1] = table.loc['Sex_F', 'with_2yfu']
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['Sex_M', 'Stat'] = x2
table.at['Sex_M', 'p(Stat)'] = p


contingency = np.zeros((2,4))
contingency[0,0] = table.loc['RE_White', 'whole_sample']
contingency[0,1] = table.loc['RE_Black', 'whole_sample']
contingency[0,2] = table.loc['RE_Hispanic', 'whole_sample']
contingency[0,3] = table.loc['RE_AsianOther', 'whole_sample']
contingency[1,0] = table.loc['RE_White', 'with_2yfu']
contingency[1,1] = table.loc['RE_Black', 'with_2yfu']
contingency[1,2] = table.loc['RE_Hispanic', 'with_2yfu']
contingency[1,3] = table.loc['RE_AsianOther', 'with_2yfu']
print('race', contingency)
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['RE_White', 'Stat'] = x2
table.at['RE_White', 'p(Stat)'] = p
print('p =', p)

contingency = np.zeros((2,4))
contingency[0,0] = table.loc['Income_gt100k', 'whole_sample']
contingency[0,1] = table.loc['Income_50to100k', 'whole_sample']
contingency[0,2] = table.loc['Income_lt50k', 'whole_sample']
contingency[0,3] = table.loc['Income_dkrefuse', 'whole_sample']
contingency[1,0] = table.loc['Income_gt100k', 'with_2yfu']
contingency[1,1] = table.loc['Income_50to100k', 'with_2yfu']
contingency[1,2] = table.loc['Income_lt50k', 'with_2yfu']
contingency[1,3] = table.loc['Income_dkrefuse', 'with_2yfu']
print('income', contingency)
x2, p, dof, exp = sstats.chi2_contingency(contingency)  
table.at['Income_gt100k', 'Stat'] = x2
table.at['Income_gt100k', 'p(Stat)'] = p
print('p =', p)

contingency = np.zeros((2,3))
contingency[0,0] = table.loc['MRI_Siemens', 'whole_sample']
contingency[0,1] = table.loc['MRI_GE', 'whole_sample']
contingency[0,2] = table.loc['MRI_Philips', 'whole_sample']
contingency[1,0] = table.loc['MRI_Siemens', 'with_2yfu']
contingency[1,1] = table.loc['MRI_GE', 'with_2yfu']
contingency[1,2] = table.loc['MRI_Philips', 'with_2yfu']
print('scanner', contingency)
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['MRI_Siemens', 'Stat'] = x2
table.at['MRI_Siemens', 'p(Stat)'] = p
print('p =', p)

contingency = np.zeros((2,5))
contingency[0,0] = table.loc['Marital_Married', 'whole_sample']
contingency[0,1] = table.loc['Marital_Widowed', 'whole_sample']
contingency[0,2] = table.loc['Marital_Divorced', 'whole_sample']
contingency[0,3] = table.loc['Marital_Separated', 'whole_sample']
contingency[0,4] = table.loc['Marital_Never', 'whole_sample']
contingency[1,0] = table.loc['Marital_Married', 'with_2yfu']
contingency[1,1] = table.loc['Marital_Widowed', 'with_2yfu']
contingency[1,2] = table.loc['Marital_Divorced', 'with_2yfu']
contingency[1,3] = table.loc['Marital_Separated', 'with_2yfu']
contingency[1,4] = table.loc['Marital_Never', 'with_2yfu']
print('income', contingency)
x2, p, dof, exp = sstats.chi2_contingency(contingency)  
table.at['Marital_Married', 'Stat'] = x2
table.at['Marital_Married', 'p(Stat)'] = p

contingency = np.zeros((2,4))
contingency[0,0] = table.loc['Education_uptoHSGED', 'whole_sample']
contingency[0,1] = table.loc['Education_SomeColAA', 'whole_sample']
contingency[0,2] = table.loc['Education_Bachelors', 'whole_sample']
contingency[0,3] = table.loc['Education_Graduate', 'whole_sample']
contingency[1,0] = table.loc['Education_uptoHSGED', 'with_2yfu']
contingency[1,1] = table.loc['Education_SomeColAA', 'with_2yfu']
contingency[1,2] = table.loc['Education_Bachelors', 'with_2yfu']
contingency[1,3] = table.loc['Education_Graduate', 'with_2yfu']
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['Education_uptoHSGED', 'Stat'] = x2
table.at['Education_uptoHSGED', 'p(Stat)'] = p


table.to_csv(join(PROJ_DIR, OUT_DIR, 'sample_demographics.csv'))

# now do the same thing for Siemens vs. GE vs. Philips
table = pd.DataFrame(index=['N', 
                            'Age_mean',
                            'Age_sdev',
                            'Age_mean_2yfu',
                            'Age_sdev_2yfu',
                            'Sex_M', 
                            'Sex_F',
                            'Puberty_1',
                            'Puberty_2',
                            'Puberty_3',
                            'Puberty_4',
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
                            'Education_AA',
                            'Education_Bachelors',
                            'Education_Graduate'], 
                     columns=['siemens', 'philips', 'ge'])

philips_df = y2fu_df[y2fu_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "Philips Medical Systems"]
siemens_df = y2fu_df[y2fu_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "SIEMENS"]
ge_df = y2fu_df[y2fu_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "GE MEDICAL SYSTEMS"]


table.at['N', 'philips'] = len(philips_df.index)
table.at['N', 'siemens'] = len(siemens_df.index)
table.at['N', 'ge'] = len(ge_df.index)

table.at['Age_mean', 'philips'] = np.mean(philips_df['interview_age.baseline_year_1_arm_1'])
table.at['Age_mean', 'siemens'] = np.mean(siemens_df['interview_age.baseline_year_1_arm_1'])
table.at['Age_mean', 'ge'] = np.mean(ge_df['interview_age.baseline_year_1_arm_1'])

table.at['Age_sdev', 'philips'] = np.std(philips_df['interview_age.baseline_year_1_arm_1'])
table.at['Age_sdev', 'siemens'] = np.std(siemens_df['interview_age.baseline_year_1_arm_1'])
table.at['Age_sdev', 'ge'] = np.std(ge_df['interview_age.baseline_year_1_arm_1'])


table.at['Sex_M', 'philips'] = len(philips_df[philips_df['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Sex_M', 'siemens'] = len(siemens_df[siemens_df['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Sex_M', 'ge'] = len(ge_df[ge_df['sex.baseline_year_1_arm_1'] == 'M'].index)

table.at['Sex_F', 'philips'] = len(philips_df[philips_df['sex.baseline_year_1_arm_1'] == 'F'].index)
table.at['Sex_F', 'siemens'] = len(siemens_df[siemens_df['sex.baseline_year_1_arm_1'] == 'F'].index)
table.at['Sex_F', 'ge'] = len(ge_df[ge_df['sex.baseline_year_1_arm_1'] == 'F'].index)

table.at['Puberty_1', 'philips'] = len(philips_df[philips_df['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Puberty_1', 'siemens'] = len(siemens_df[siemens_df['sex.baseline_year_1_arm_1'] == 'M'].index)
table.at['Puberty_1', 'ge'] = len(ge_df[ge_df['sex.baseline_year_1_arm_1'] == 'M'].index)

table.at['Sex_F', 'philips'] = len(philips_df[philips_df['sex.baseline_year_1_arm_1'] == 'F'].index)
table.at['Sex_F', 'siemens'] = len(siemens_df[siemens_df['sex.baseline_year_1_arm_1'] == 'F'].index)
table.at['Sex_F', 'ge'] = len(ge_df[ge_df['sex.baseline_year_1_arm_1'] == 'F'].index)


table.at['RE_White', 
         'philips'] = len(philips_df[philips_df['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
table.at['RE_White', 
         'siemens'] = len(siemens_df[siemens_df['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
table.at['RE_White', 
         'ge'] = len(ge_df[ge_df['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
table.at['RE_Black', 
         'philips'] = len(philips_df[philips_df['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)
table.at['RE_Black', 
         'siemens'] = len(siemens_df[siemens_df['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)
table.at['RE_Black', 
         'ge'] = len(ge_df[ge_df['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)

table.at['RE_Hispanic', 
         'philips'] = len(philips_df[philips_df['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)
table.at['RE_Hispanic', 
         'siemens'] = len(siemens_df[siemens_df['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)
table.at['RE_Hispanic', 
         'ge'] = len(ge_df[ge_df['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)

table.at['RE_AsianOther', 
         'philips'] = len(philips_df[philips_df['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)
table.at['RE_AsianOther', 
         'siemens'] = len(siemens_df[siemens_df['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)
table.at['RE_AsianOther', 
         'ge'] = len(ge_df[ge_df['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)


table.at['Income_gt100k', 
         'philips'] = len(philips_df[philips_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)
table.at['Income_gt100k', 
         'siemens'] = len(siemens_df[siemens_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)
table.at['Income_gt100k', 
         'ge'] = len(ge_df[ge_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)

table.at['Income_50to100k', 
         'philips'] = len(philips_df[philips_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)
table.at['Income_50to100k', 
         'siemens'] = len(siemens_df[siemens_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)
table.at['Income_50to100k', 
         'ge'] = len(ge_df[ge_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)

table.at['Income_lt50k', 
         'philips'] = len(philips_df[philips_df['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)
table.at['Income_lt50k', 
         'siemens'] = len(siemens_df[siemens_df['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)
table.at['Income_lt50k', 
         'ge'] = len(ge_df[ge_df['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)

table.at['Income_dkrefuse', 
         'philips'] = len(philips_df[philips_df['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)
table.at['Income_dkrefuse', 
         'siemens'] = len(siemens_df[siemens_df['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)
table.at['Income_dkrefuse', 
         'ge'] = len(ge_df[ge_df['demo_comb_income_v2.baseline_year_1_arm_1'] >= 777.].index)

table.at['Marital_Married', 
         'philips'] = len(philips_df[philips_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])
table.at['Marital_Married', 
         'siemens'] = len(siemens_df[siemens_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])
table.at['Marital_Married', 
         'ge'] = len(ge_df[ge_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 1.])

table.at['Marital_Widowed', 
         'philips'] = len(philips_df[philips_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])
table.at['Marital_Widowed', 
         'siemens'] = len(siemens_df[siemens_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])
table.at['Marital_Widowed', 
         'ge'] = len(ge_df[ge_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 2.])

table.at['Marital_Divorced', 
         'philips'] = len(philips_df[philips_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])
table.at['Marital_Divorced', 
         'siemens'] = len(siemens_df[siemens_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])
table.at['Marital_Divorced', 
         'ge'] = len(ge_df[ge_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 3.])

table.at['Marital_Separated', 
         'philips'] = len(philips_df[philips_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])
table.at['Marital_Separated', 
         'siemens'] = len(siemens_df[siemens_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])
table.at['Marital_Separated', 
         'ge'] = len(ge_df[ge_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 4.])

table.at['Marital_Never', 
         'philips'] = len(philips_df[philips_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])
table.at['Marital_Never', 
         'siemens'] = len(siemens_df[siemens_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])
table.at['Marital_Never', 
         'ge'] = len(ge_df[ge_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 5.])

table.at['Marital_Refused', 
         'philips'] = len(philips_df[philips_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])
table.at['Marital_Refused', 
         'siemens'] = len(siemens_df[siemens_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])
table.at['Marital_Refused', 
         'ge'] = len(ge_df[ge_df["demo_prnt_marital_v2.baseline_year_1_arm_1"] == 777.])

table.at['Education_uptoHSGED', 
         'philips'] = len(philips_df[philips_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'philips'] = len(philips_df[philips_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'philips'] = len(philips_df[philips_df['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
table.at['Education_Graduate', 
         'philips'] = len(philips_df[philips_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, 
                                                                                                inclusive='both')])
table.at['Education_uptoHSGED', 
         'siemens'] = len(siemens_df[siemens_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'siemens'] = len(siemens_df[siemens_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'siemens'] = len(siemens_df[siemens_df['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
table.at['Education_Graduate', 
         'siemens'] = len(siemens_df[siemens_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, 
                                                                                                inclusive='both')])

table.at['Education_uptoHSGED', 
         'ge'] = len(ge_df[ge_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'ge'] = len(ge_df[ge_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'ge'] = len(ge_df[ge_df['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
table.at['Education_Graduate', 
         'ge'] = len(ge_df[ge_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, 
                                                                                                inclusive='both')])

# test for differences in means with wilcoxon signed rank test
stat, pval = sstats.mannwhitneyu(demo_df['interview_age.baseline_year_1_arm_1'].dropna(), 
                                 y2fu_df['interview_age.baseline_year_1_arm_1'].dropna())
table.at['Age_mean', 'Stat'] = stat
table.at['Age_mean', 'p(Stat)'] = pval

contingency = np.zeros((3,2))
contingency[0,0] = table.loc['Sex_M', 'philips']
contingency[0,1] = table.loc['Sex_F', 'philips']
contingency[1,0] = table.loc['Sex_M', 'siemens']
contingency[1,1] = table.loc['Sex_F', 'siemens']
contingency[2,0] = table.loc['Sex_M', 'ge']
contingency[2,1] = table.loc['Sex_F', 'ge']
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['Sex_M', 'Stat'] = x2
table.at['Sex_M', 'p(Stat)'] = p


contingency = np.zeros((3,4))
contingency[0,0] = table.loc['RE_White', 'philips']
contingency[0,1] = table.loc['RE_Black', 'philips']
contingency[0,2] = table.loc['RE_Hispanic', 'philips']
contingency[0,3] = table.loc['RE_AsianOther', 'philips']
contingency[1,0] = table.loc['RE_White', 'siemens']
contingency[1,1] = table.loc['RE_Black', 'siemens']
contingency[1,2] = table.loc['RE_Hispanic', 'siemens']
contingency[1,3] = table.loc['RE_AsianOther', 'siemens']
contingency[2,0] = table.loc['RE_White', 'ge']
contingency[2,1] = table.loc['RE_Black', 'ge']
contingency[2,2] = table.loc['RE_Hispanic', 'ge']
contingency[2,3] = table.loc['RE_AsianOther', 'ge']
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['RE_White', 'Stat'] = x2
table.at['RE_White', 'p(Stat)'] = p

contingency = np.zeros((3,4))
contingency[0,0] = table.loc['Income_gt100k', 'philips']
contingency[0,1] = table.loc['Income_50to100k', 'philips']
contingency[0,2] = table.loc['Income_lt50k', 'philips']
contingency[0,3] = table.loc['Income_dkrefuse', 'philips']
contingency[1,0] = table.loc['Income_gt100k', 'siemens']
contingency[1,1] = table.loc['Income_50to100k', 'siemens']
contingency[1,2] = table.loc['Income_lt50k', 'siemens']
contingency[1,3] = table.loc['Income_dkrefuse', 'siemens']
contingency[2,0] = table.loc['Income_gt100k', 'ge']
contingency[2,1] = table.loc['Income_50to100k', 'ge']
contingency[2,2] = table.loc['Income_lt50k', 'ge']
contingency[2,3] = table.loc['Income_dkrefuse', 'ge']
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['Income_gt100k', 'Stat'] = x2
table.at['Income_gt100k', 'p(Stat)'] = p


contingency = np.zeros((3,5))
contingency[0,0] = table.loc['Marital_Married', 'philips']
contingency[0,1] = table.loc['Marital_Widowed', 'philips']
contingency[0,2] = table.loc['Marital_Divorced', 'philips']
contingency[0,3] = table.loc['Marital_Separated', 'philips']
contingency[0,4] = table.loc['Marital_Never', 'philips']
contingency[1,0] = table.loc['Marital_Married', 'siemens']
contingency[1,1] = table.loc['Marital_Widowed', 'siemens']
contingency[1,2] = table.loc['Marital_Divorced', 'siemens']
contingency[1,3] = table.loc['Marital_Separated', 'siemens']
contingency[1,4] = table.loc['Marital_Never', 'siemens']
contingency[2,0] = table.loc['Marital_Married', 'ge']
contingency[2,1] = table.loc['Marital_Widowed', 'ge']
contingency[2,2] = table.loc['Marital_Divorced', 'ge']
contingency[2,3] = table.loc['Marital_Separated', 'ge']
contingency[2,4] = table.loc['Marital_Never', 'ge']
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['Marital_Married', 'Stat'] = x2
table.at['Marital_Married', 'p(Stat)'] = p

contingency = np.zeros((3,4))
contingency[0,0] = table.loc['Education_uptoHSGED', 'philips']
contingency[0,1] = table.loc['Education_SomeColAA', 'philips']
contingency[0,2] = table.loc['Education_Bachelors', 'philips']
contingency[0,3] = table.loc['Education_Graduate', 'philips']
contingency[1,0] = table.loc['Education_uptoHSGED', 'siemens']
contingency[1,1] = table.loc['Education_SomeColAA', 'siemens']
contingency[1,2] = table.loc['Education_Bachelors', 'siemens']
contingency[1,3] = table.loc['Education_Graduate', 'siemens']
contingency[2,0] = table.loc['Education_uptoHSGED', 'ge']
contingency[2,1] = table.loc['Education_SomeColAA', 'ge']
contingency[2,2] = table.loc['Education_Bachelors', 'ge']
contingency[2,3] = table.loc['Education_Graduate', 'ge']
x2, p, dof, exp = sstats.chi2_contingency(contingency) 
table.at['Education_uptoHSGED', 'Stat'] = x2
table.at['Education_uptoHSGED', 'p(Stat)'] = p

table.dropna(how='all')

table.to_csv(join(PROJ_DIR, OUT_DIR, 'demographic_differences_between_scanners.csv'))