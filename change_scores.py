import pandas as pd
import numpy as np
import enlighten
from os.path import exists, join
import sys

data_dir = '/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0'
data_dict_path = join(data_dir, 'data_element_names.csv')

imgincl_df = pd.read_csv(join(data_dir, 'csv', 'abcd_imgincl01.csv'),
                         index_col=['subjectkey', 'eventname'], header=0, skiprows=[1])

common_vars = ["subjectkey", "interview_date", 
               "interview_age", "eventname", "sex"]
img_qc_vars = ["imgincl_t1w_include","imgincl_t2w_include",
               "imgincl_dmri_include","imgincl_rsfmri_include"]
events_of_interest = ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1']

structures = ['abcd_smrip30201', 'abcd_mrirsfd01', 'abcd_mrisdp20201', 'abcd_smrip20201', 'abcd_smrip10201', 'abcd_mrisdp10201', 
              'abcd_dti_p101', 'abcd_drsip101', 'abcd_drsip201', 
              'abcd_mrirstv02', 'abcd_tbss01']
structures = ['abcd_betnet02', 'mrirscor02']

manager = enlighten.get_manager()
tocks = manager.counter(total=len(structures), desc='Data Structures', unit='data structures')

#missing_idx = {}

for structure in structures:
    #missing_idx[structure] = {}
    #print(structure)
    
    temp_df = pd.read_csv(join(data_dir, 'csv', f'{structure}.csv'),
                            index_col='subjectkey',
                            header=0, skiprows=[1])
    temp_df['interview_age_yrs'] = temp_df['interview_age'] / 12
    float_vars = list(temp_df.dtypes[temp_df.dtypes == float].keys())
    int_vars = list(temp_df.dtypes[temp_df.dtypes == int].keys())
    num_vars = list(set(float_vars + int_vars) - set(img_qc_vars))
    base_df = temp_df[temp_df['eventname'] == 'baseline_year_1_arm_1']
    y2fu_df = temp_df[temp_df['eventname'] == '2_year_follow_up_y_arm_1']
    change_df = pd.DataFrame(index=base_df.index)
    for var2 in common_vars:
        for i in base_df.index:
            try:
                change_df.at[i, f'{var2}.baseline_year_1_arm_1'] = base_df.loc[i, var2]
                change_df.at[i, f'{var2}.2_year_follow_up_y_arm_1'] = y2fu_df.loc[i, var2]
            except:
                pass
    if structure != 'abcd_tbss01':
        for var2 in img_qc_vars:
            for i in base_df.index:
                for j in events_of_interest:
                    try:
                        change_df.at[i, f'{var2}.{j}'] = imgincl_df.loc[(i, j), var2]
                    except:
                        pass
    else:
        pass
    ticks = manager.counter(total=len(num_vars), desc=structure, unit='variables')
    for var in num_vars:
        for i in base_df.index:
            try:
                # annualized percent change for resting state connectivity had absurd values
                # and standard deviations in the millions!!
                # due to very very small denominators in the "percent" part of the formula
                # calculating the changes in the absolute value of connectivity fixes this
                # and any sign changes are retained in the "{var}.sign_change" variables
                # sign change abs(value) = 1, there was no change
                # sign change abs(value) = 2, sign changed between timepoints
                # sign change +, r-value increased from baseline to 2yfu
                # sign change -, r-value decreased from baseline to 2yfu
                # I hope that makes sense...
                if 'rsfmri_c' in var:
                    base = abs(base_df.loc[i, var])
                    y2fu = abs(y2fu_df.loc[i, var])
                    if base * y2fu > 0:
                        if y2fu > 0:
                            change_df.at[i, f'{var}.sign_change'] = 1
                        else:
                            change_df.at[i, f'{var}.sign_change'] = -1
                    else:
                        if y2fu > 0:
                            change_df.at[i, f'{var}.sign_change'] = 2
                        else:
                            change_df.at[i, f'{var}.sign_change'] = -2
                else:
                    base = base_df.loc[i, var]
                    y2fu = y2fu_df.loc[i, var]
                age0 = base_df.loc[i, 'interview_age_yrs']
                age2 = y2fu_df.loc[i, 'interview_age_yrs']
                change_df.at[i, f'{var}.baseline_year_1_arm_1'] = base
                change_df.at[i, f'{var}.2_year_follow_up_y_arm_1'] = y2fu
                change_df.at[i, f'interview_age_yrs.baseline_year_1_arm_1'] = age0
                change_df.at[i, f'interview_age_yrs.2_year_follow_up_y_arm_1'] = age2
                change_score = (((y2fu - base) / np.mean([y2fu, base])) * 100) / (age2 - age0)
                change_df.at[i, f'{var}.change_score'] = change_score
            except:
                pass
        ticks.update()
    print(f'{structure} with change scores is {sys.getsizeof(change_df) / 1000000}MB')
    change_df.to_csv(join(data_dir, 'change_scores', f'{structure}_changescores_bl_tp2.csv'))
    change_df = None
    temp_df = None
    base_df = None
    y2fu_df = None
    tocks.update()

