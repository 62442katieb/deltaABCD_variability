Author: Katie Bottenhorn
Date: Nov. 17, 2021

##

Code can be found in change_scores.py.

Calculated annualized percent change scores between baseline and 2-year follow-up visits for the following ABCD 4.0 curated release data structures (see line 19):
['abcd_smrip30201', 'abcd_mrirsfd01', 'abcd_mrisdp20201', 'abcd_smrip20201', 'abcd_smrip10201', 'abcd_mrisdp10201', 'abcd_dti_p101', 'abcd_drsip101', 'abcd_drsip201', 'abcd_mrirstv02', 'abcd_betnet02', 'mrirscor02', 'abcd_tbss01'] 

Variables common to every data structure (i.e., "subjectkey", "interview_date", "interview_age", "eventname", "sex”) were retained in each resulting data structure (see line 13) and imaging quality control variables were included in each imaging change score data structure (see like 15). 

For each data structure of interest:
1. converting interview_age from months to years (line 35)
2. data were separated by visit (‘event_name’) into separate, temporary data frames for baseline and 2-year follow-up data (lines 37, 38).
3. an empty data frame was initialized and filled with values of each common variable (lines 39-46) and imaging QC variables where appropriate (lines 47-54).

For each floating point (i.e., numerical) variable within each data structure (lines 36, 58-72), annualized change scores were calculated by:
1. grabbing baseline and 2-year follow-up values (lines 61, 62)
2. grabbing baseline and 2-year follow-up values for age (in years; lines 63,64)
3. dividing the difference in values (2yfu - baseline) by the mean of the two values, multiplying by 100 for percentages, and dividing by the difference in ages (2yfu - baseline) per Mills et al., 2021 (line 69)

Change score variable names follow the convention of the original data, appended with “.changescores_bl_tp2” (line 70). Baseline values’ variable names are appended with “.baseline_year_1_arm_1” (line 65);  2-year follow-up, “.2_year_follow_up_y_arm_1”.

Baseline and 2-year follow-up values were retained in each dataset, as well (lines 65-68).

Progress was monitored using the `enlighten` library’s progress bar (see vars “ticks” and “tocks”).

Change score data structures were saved using the naming conventions of the original data, appended with “_changescores_bl_tp2.csv”.