#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
#import seaborn as sns

from os.path import join
from sklearn.ensemble import IsolationForest

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))

ppts = df.filter(regex="rsfmri_c_.*change_score").dropna().index

demographics = ["demo_prnt_ed_v2.baseline_year_1_arm_1",
                "demo_comb_income_v2.baseline_year_1_arm_1",
                "race_ethnicity.baseline_year_1_arm_1",
                "site_id_l.baseline_year_1_arm_1",
                "sex.baseline_year_1_arm_1", 
                "mri_info_manufacturer.baseline_year_1_arm_1"
               ]

rsfc_vars = list(df.filter(regex="rsfmri_c_ngd_.*change_score").columns)

demo_df = df[demographics]
#df = None

vars_of_interest = rsfc_vars + demographics
small_df = df.loc[ppts][vars_of_interest]
small_df = small_df.drop(["site_id_l.baseline_year_1_arm_1","mri_info_manufacturer.baseline_year_1_arm_1"], axis=1)
small_df = small_df.replace({'F': 1, 'M': 0})

outliers = IsolationForest().fit_predict(small_df)
outlier_ser = pd.Series(outliers, small_df.index)
outlier_ser.to_pickle(join(PROJ_DIR, OUTP_DIR, 'delta_rsFC-outliers.csv'))

rsfmri_quality = demo_df.loc[ppts]

col_to_df = {
    'whole_sample': df, 
    'with_rsfmri': rsfmri_quality,
}

table = pd.DataFrame(index=['N', 
                            'Age_mean_base',
                            'Age_sdev_base',
                            'Age_Missing',
                            'Sex_M', 
                            'Sex_F', 
                            'Sex_Missing',
                            'RE_Black',
                            'RE_White',
                            'RE_Hispanic',
                            'RE_AsianOther',
                            'RE_Missing',
                            'RE_Refuse',
                            'Income_gt100k', 
                            'Income_50to100k', 
                            'Income_lt50k',
                            'Income_dkrefuse',
                            'Income_Missing',
                            'Education_uptoHSGED',
                            'Education_SomeColAA',
                            'Education_Bachelors',
                            'Education_Graduate',
                            'Education_Refused',
                            'Education_Missing',
                            'MRI_Siemens', 
                            'MRI_GE', 
                            'MRI_Philips',
                            'MRI_Missing',
                            ], 
                     columns=list(col_to_df.keys()))

for subset in col_to_df.keys():
    #print(subset, type(col_to_df[subset]))
    temp_df = col_to_df[subset]
    table.at['N', subset] = len(temp_df.index)
    table.at['Age_mean_base', subset] = np.mean(temp_df['interview_age.baseline_year_1_arm_1'])
    table.at['Age_sdev_base', subset] = np.std(temp_df['interview_age.baseline_year_1_arm_1'])
    

    # demographics
    table.at['Age_Missing', subset] = temp_df['interview_age.baseline_year_1_arm_1'].isna().sum()
    table.at['Sex_M', subset] = len(temp_df[temp_df['sex.baseline_year_1_arm_1'] == 'M'].index)
    table.at['Sex_F', subset] = len(temp_df[temp_df['sex.baseline_year_1_arm_1'] == 'F'].index)
    table.at['Sex_Missing', subset] = temp_df['sex'].isna().sum()
    table.at['RE_White',
             subset] = len(temp_df[temp_df['race_ethnicity.baseline_year_1_arm_1'] == 1.].index)
    table.at['RE_Black',
             subset] = len(temp_df[temp_df['race_ethnicity.baseline_year_1_arm_1'] == 2.].index)
    table.at['RE_Hispanic',
             subset] = len(temp_df[temp_df['race_ethnicity.baseline_year_1_arm_1'] == 3.].index)
    table.at['RE_AsianOther',
             subset] = len(temp_df[temp_df['race_ethnicity.baseline_year_1_arm_1'].between(4.,5.,inclusive='both')].index)
    table.at['RE_Refuse',
             subset] = len(temp_df[temp_df['race_ethnicity.baseline_year_1_arm_1'] == 777.].index)
    table.at['RE_Missing',
             subset] = temp_df['race_ethnicity.baseline_year_1_arm_1'].isna().sum() + len(temp_df[temp_df['race_ethnicity.baseline_year_1_arm_1'] == 999.].index)
    table.at['Income_gt100k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(9.,10., inclusive='both')].index)
    table.at['Income_50to100k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2.baseline_year_1_arm_1'].between(7., 8., inclusive='both')].index)
    table.at['Income_lt50k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2.baseline_year_1_arm_1'] <= 6.].index)
    table.at['Income_dkrefuse', 
            subset] = len(temp_df[temp_df['demo_comb_income_v2.baseline_year_1_arm_1'] == 777.].index)
    table.at['Income_Missing', 
            subset] = len(temp_df[temp_df['demo_comb_income_v2.baseline_year_1_arm_1.baseline_year_1_arm_1'] == 999.].index) + temp_df['demo_comb_income_v2.baseline_year_1_arm_1'].isna().sum()
    table.at['MRI_Siemens', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "SIEMENS"].index)
    table.at['MRI_GE', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "GE MEDICAL SYSTEMS"].index)
    table.at['MRI_Philips', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer.baseline_year_1_arm_1'] == "Philips Medical Systems"].index)
    table.at['MRI_Missing', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer.baseline_year_1_arm_1'].isna()].index)
    table.at['Education_uptoHSGED', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(0,14,inclusive='both')])
    table.at['Education_SomeColAA', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(15,17, inclusive='both')])
    table.at['Education_Bachelors', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 18])
    table.at['Education_Graduate', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].between(19,22, inclusive='both')])
    table.at['Education_Refused', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 777.])
    table.at['Education_Missing', 
            subset] = temp_df['demo_prnt_ed_v2.baseline_year_1_arm_1'].isna().sum() + len(temp_df[temp_df['demo_prnt_ed_v2.baseline_year_1_arm_1'] == 999.])

table.to_csv(join(PROJ_DIR, OUTP_DIR, 'delta_rsFC-demographics.csv'))