#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt

from os.path import join

from scipy.stats import fligner
from nilearn import plotting, datasets, surface

sns.set(style='whitegrid', context='talk')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data.csv"), index_col=0, header=0)


df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
df.drop(list(df.filter(regex='.*_cf12_.*').columns), axis=1, inplace=True)
no_2yfu = df[df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
df = df.drop(no_2yfu, axis=0)

deltasmri_complete = pd.concat([df.filter(regex='smri.*change_score'), 
                                df.filter(regex='mrisdp.*change_score')], axis=1).dropna()
deltarsfmri_complete = df.filter(regex='rsfmri.*change_score').dropna(how='any')
deltarsi_complete = df.filter(regex='dmri_rsi.*change_score').dropna()
deltadti_complete = df.filter(regex='dmri_dti.*change_score').dropna()


tests = ['variance', 
         'fligner_sex',
         'fligner_puberty_m',
         'fligner_puberty_f',
         'fligner_puberty_1',
         'fligner_puberty_2',
         'fligner_puberty_3',
         #'fligner_puberty_4',
         #'fligner_puberty_5',
         'fligner_raceth', 
         'fligner_income', 
         'fligner_edu', 
         'fligner_marital', 
         'fligner_age', 
         'fligner_scanner']


alpha = 0.05
values = ['stat', 'p', 'diff', 'greater', f'a<{alpha}']
columns = pd.MultiIndex.from_product([tests, values])

var_df = pd.DataFrame(columns=columns)


age_bins = np.zeros((10,))
for percentile in range(0, 100, 10):
    index = int(percentile / 10)
    age_bins[index] = np.percentile(df['interview_age.baseline_year_1_arm_1'], 
                                    float(percentile))

img_modalities = {'smri': deltasmri_complete,
                  'fmri': deltarsfmri_complete,
                  'rsi': deltarsi_complete, 
                  'dti': deltadti_complete}

for modality in img_modalities.keys():
    variables = img_modalities[modality].columns
    for var in variables:
        # compute variance across the sample
        var_df.at[var, ('variance', 'stat')] = np.var(df[var])
        
        # compare variance between male and female participants
        m = df[df['sex.baseline_year_1_arm_1'] == 'M'][var].dropna()
        f = df[df['sex.baseline_year_1_arm_1'] == 'F'][var].dropna()
        test = fligner(m, f)
        var_df.at[var, ('fligner_sex', 'stat')] = test[0]
        var_df.at[var, ('fligner_sex', 'p')] = test[1]
        var_df.at[var, ('fligner_sex', 'diff')] = np.mean(f) - np.mean(m)
        var_df.at[var, ('fligner_sex', 'greater')] = 'f'
        if test[1] < alpha:
            var_df.at[var, ('fligner_sex', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_sex', f'a<{alpha}')] = 'ns'
        
        # compare variance between pubertal stages at baseline, female ppts
        puberty = 'pds_p_ss_female_category_2.baseline_year_1_arm_1'
        one = df[df[puberty] == 1.][var].dropna()
        two = df[df[puberty] == 2.][var].dropna()
        three = df[df[puberty] == 3.][var].dropna()
        test = fligner(one, two, three)
        var_df.at[var, ('fligner_puberty_f', 'stat')] = test[0]
        var_df.at[var, ('fligner_puberty_f', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_puberty_f', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_puberty_f', f'a<{alpha}')] = 'ns'
        
        var_dict = {'1': np.mean(one), 
                    '2': np.mean(two), 
                    '3': np.mean(three)}
        max_key = max(var_dict, key=var_dict.get)
        min_val = min(var_dict.values())
        sorted_keys = {k: v for k, v in sorted(var_dict.items(), key=lambda item: item[1])}
        var_df.at[var, ('fligner_puberty_f', 'diff')] = var_dict[max_key] - min_val
        var_df.at[var, ('fligner_puberty_f', 'greater')] = [list(sorted_keys.keys())]
        
        # compare variance between pubertal stages at baseline, male ppts
        puberty = 'pds_p_ss_male_category_2.baseline_year_1_arm_1'
        one = df[df[puberty] == 1.][var].dropna()
        two = df[df[puberty] == 2.][var].dropna()
        three = df[df[puberty] == 3.][var].dropna()
        test = fligner(one, two, three)
        var_dict = {'1': np.mean(one), 
                    '2': np.mean(two), 
                    '3': np.mean(three)}
        max_key = max(var_dict, key=var_dict.get)
        min_val = min(var_dict.values())
        sorted_keys = {k: v for k, v in sorted(var_dict.items(), key=lambda item: item[1])}
        var_df.at[var, ('fligner_puberty_m', 'diff')] = var_dict[max_key] - min_val
        var_df.at[var, ('fligner_puberty_m', 'greater')] = [list(sorted_keys.keys())]
        var_df.at[var, ('fligner_puberty_m', 'stat')] = test[0]
        var_df.at[var, ('fligner_puberty_m', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_puberty_m', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_puberty_m', f'a<{alpha}')] = 'ns'
        
        # compare variance between sexes at baseline, pubertal stages
        puberty_f = 'pds_p_ss_female_category_2.baseline_year_1_arm_1'
        puberty_m = 'pds_p_ss_male_category_2.baseline_year_1_arm_1'
        one_f = df[df[puberty_f] == 1.][var].dropna()
        two_f = df[df[puberty_f] == 2.][var].dropna()
        three_f = df[df[puberty_f] == 3.][var].dropna()
        one_m = df[df[puberty_m] == 1.][var].dropna()
        two_m = df[df[puberty_m] == 2.][var].dropna()
        three_m = df[df[puberty_m] == 3.][var].dropna()
        test = fligner(one_f, one_m)
        var_df.at[var, ('fligner_puberty_1', 'stat')] = test[0]
        var_df.at[var, ('fligner_puberty_1', 'p')] = test[1]
        var_df.at[var, ('fligner_puberty_1', 'diff')] = np.mean(one_f) - np.mean(one_m)
        var_df.at[var, ('fligner_puberty_1', 'greater')] = 'f'
        if test[1] < alpha:
            var_df.at[var, ('fligner_puberty_1', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_puberty_1', f'a<{alpha}')] = 'ns'
        test = fligner(two_f, two_m)
        var_df.at[var, ('fligner_puberty_2', 'stat')] = test[0]
        var_df.at[var, ('fligner_puberty_2', 'p')] = test[1]
        var_df.at[var, ('fligner_puberty_2', 'diff')] = np.mean(two_f) - np.mean(two_m)
        var_df.at[var, ('fligner_puberty_2', 'greater')] = 'f'
        if test[1] < alpha:
            var_df.at[var, ('fligner_puberty_2', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_puberty_2', f'a<{alpha}')] = 'ns'
        test = fligner(three_f, three_m)
        var_df.at[var, ('fligner_puberty_3', 'stat')] = test[0]
        var_df.at[var, ('fligner_puberty_3', 'p')] = test[1]
        var_df.at[var, ('fligner_puberty_3', 'diff')] = np.mean(three_f) - np.mean(three_m)
        var_df.at[var, ('fligner_puberty_3', 'greater')] = 'f'
        if test[1] < alpha:
            var_df.at[var, ('fligner_puberty_3', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_puberty_3', f'a<{alpha}')] = 'ns'
        
        #compare variance across race/ethnicities
        race = 'race_ethnicity.baseline_year_1_arm_1'
        white = df[df[race] == 1.][var].dropna()
        black = df[df[race] == 2.][var].dropna()
        hispanic = df[df[race] == 3.][var].dropna()
        asian = df[df[race] == 4.][var].dropna()
        other = df[df[race] == 5.][var].dropna()
        test = fligner(white, black, hispanic, asian, other)
        var_df.at[var, ('fligner_raceth', 'stat')] = test[0]
        var_df.at[var, ('fligner_raceth', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_raceth', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_raceth', f'a<{alpha}')] = 'ns'
        
        # compare variance across income
        income = 'demo_comb_income_v2.baseline_year_1_arm_1'
        lt50k = df[df[income] < 6.][var].dropna()
        lt100 = df[df[income].between(6., 8.)][var].dropna()
        gt100 = df[df[income].between(9., 10.)][var].dropna()
        
        test = fligner(lt50k, lt100, gt100)
        var_df.at[var, ('fligner_income', 'stat')] = test[0]
        var_df.at[var, ('fligner_income', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_income', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_income', f'a<{alpha}')] = 'ns'
        
        # compare variance across education
        edu = 'demo_prnt_ed_v2.baseline_year_1_arm_1'
        one = df[df[edu] == 1][var].dropna()
        two = df[df[edu] == 2][var].dropna()
        three = df[df[edu] == 3][var].dropna()
        four = df[df[edu] == 4][var].dropna()
        five = df[df[edu] == 5][var].dropna()
        six = df[df[edu] == 6][var].dropna()
        seven = df[df[edu] == 7][var].dropna()
        eight = df[df[edu] == 8][var].dropna()
        nine = df[df[edu] == 9][var].dropna()
        ten = df[df[edu] == 10][var].dropna()
        eleven = df[df[edu] == 11][var].dropna()
        twelve = df[df[edu] == 12][var].dropna()
        thirteen = df[df[edu] == 13][var].dropna()
        fourteen = df[df[edu] == 14][var].dropna()
        fifteen = df[df[edu] == 15][var].dropna()
        sixteen = df[df[edu] == 16][var].dropna()
        seventeen = df[df[edu] == 17][var].dropna()
        eighteen = df[df[edu] == 18][var].dropna()
        nineteen = df[df[edu] == 19][var].dropna()
        twenty = df[df[edu] == 20][var].dropna()
        twentyone = df[df[edu] == 21][var].dropna()
        sevens = df[df[edu] == 777][var].dropna()
        test = fligner(one, two, three, four, five, six, 
                      seven, eight, nine, ten, sevens, 
                      eleven, twelve, thirteen, fourteen, fifteen,
                      sixteen, seventeen, eighteen, nineteen, twenty, twentyone)
        var_df.at[var, ('fligner_edu', 'stat')] = test[0]
        var_df.at[var, ('fligner_edu', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_edu', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_edu', f'a<{alpha}')] = 'ns'
        
        # compare variance across age, binned by n0th percentiles
        # age_bins are calculated above
        age = 'interview_age.baseline_year_1_arm_1'
        one = df[df[age] < age_bins[1]][var].dropna()
        two = df[df[age].between(age_bins[1], age_bins[2] - 1)][var].dropna()
        three = df[df[age].between(age_bins[2], age_bins[3] - 1.)][var].dropna()
        four = df[df[age].between(age_bins[3], age_bins[4] - 1.)][var].dropna()
        five = df[df[age].between(age_bins[4], age_bins[5] - 1.)][var].dropna()
        six = df[df[age].between(age_bins[5], age_bins[6] - 1.)][var].dropna()
        seven = df[df[age].between(age_bins[6], age_bins[7] - 1.)][var].dropna()
        eight = df[df[age].between(age_bins[7], age_bins[8] - 1.)][var].dropna()
        nine = df[df[age].between(age_bins[8], age_bins[9] - 1.)][var].dropna()
        ten = df[df[age] >= age_bins[9]][var].dropna()
                 
        test = fligner(one, two, three, four, five, six, 
                      seven, eight, nine, ten)
        var_df.at[var, ('fligner_age', 'stat')] = test[0]
        var_df.at[var, ('fligner_age', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_age', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_age', f'a<{alpha}')] = 'ns'
        
         # compare variance across scanner manufacturers
        mri = 'mri_info_manufacturer.baseline_year_1_arm_1'
        siemens = df[df[mri] == 'SIEMENS'][var].dropna()
        ge = df[df[mri] == 'GE MEDICAL SYSTEMS'][var].dropna()
        philips = df[df[mri] == 'Philips Medical Systems'][var].dropna()
        
        test = fligner(siemens, philips, ge)
        var_df.at[var, ('fligner_scanner', 'stat')] = test[0]
        var_df.at[var, ('fligner_scanner', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_scanner', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_scanner', f'a<{alpha}')] = 'ns'
        
        # compare variance across parent marital status
        marry = "demo_prnt_marital_v2.baseline_year_1_arm_1"
        married = df[df[marry] == 1.][var].dropna()
        widowed = df[df[marry] == 2.][var].dropna()
        divorced = df[df[marry] == 3.][var].dropna()
        separated = df[df[marry] == 4.][var].dropna()
        never = df[df[marry] == 5.][var].dropna()
        refuse = df[df[marry] == 777.][var].dropna()
        
        test = fligner(married, widowed, separated, divorced, never, refuse)
        var_df.at[var, ('fligner_marital', 'stat')] = test[0]
        var_df.at[var, ('fligner_marital', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_marital', f'a<{alpha}')] = '**'
        else:
            var_df.at[var, ('fligner_marital', f'a<{alpha}')] = 'ns'


var_df.dropna(how='all', axis=1, inplace=True)

# calculate what proportion of measures show significant heteroscedasticity
# just count('**') / 1776 measures
var_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'variance_flinger.csv'))

crayons = sns.crayon_palette(['Aquamarine', 'Burnt Sienna', 'Jungle Green', 'Fuchsia', 'Lavender'])


# plot the distribution of variances of all structural mri measures
smri_var = img_modalities['smri'].columns
dti_var = img_modalities['dti'].columns
rsi_var = img_modalities['rsi'].columns
fmri_var = img_modalities['fmri'].columns
fmri_cor_var = img_modalities['fmri'].filter(regex='_cor_.*').columns
fmri_var_var = img_modalities['fmri'].filter(regex='_var_.*').columns


fig,ax = plt.subplots(ncols=2, figsize=(15,5))
g = sns.kdeplot(var_df.loc[smri_var, ('variance', 'stat')], color=crayons[0], shade=True, ax=ax[0])
h = sns.kdeplot(var_df.loc[dti_var, ('variance', 'stat')], color=crayons[1], shade=True, ax=ax[0])
i = sns.kdeplot(var_df.loc[rsi_var, ('variance', 'stat')], color=crayons[2], shade=True, ax=ax[0])
#m = sns.kdeplot(var_df.loc[fmri_var, ('variance', 'stat')], color=crayons[3])
j = sns.rugplot(var_df.loc[smri_var, ('variance', 'stat')], color=crayons[0], lw=1, alpha=.1, ax=ax[0])
k = sns.rugplot(var_df.loc[dti_var, ('variance', 'stat')], color=crayons[1], lw=1, alpha=0.2, ax=ax[0])
l = sns.rugplot(var_df.loc[rsi_var, ('variance', 'stat')], color=crayons[2], lw=1, alpha=.1, ax=ax[0])
#n = sns.rugplot(var_df.loc[fmri_var, ('variance', 'stat')], color=crayons[3])
ax[0].set_xlabel('Variance')
plt.tight_layout()
ax[0].legend(['smri', 
           'dti', 
           'rsi', 
           #'fmri'
          ])
ax[0].set_title('A')

m = sns.kdeplot(var_df.loc[fmri_cor_var, ('variance', 'stat')], color=crayons[3], shade=True, ax=ax[1])
n = sns.rugplot(var_df.loc[fmri_cor_var, ('variance', 'stat')], color=crayons[3], lw=1, alpha=.1, ax=ax[1])
o = sns.kdeplot(var_df.loc[fmri_var_var, ('variance', 'stat')], color=crayons[4], shade=True, ax=ax[1])
p = sns.rugplot(var_df.loc[fmri_var_var, ('variance', 'stat')], color=crayons[4], lw=1, alpha=.1, ax=ax[1])
ax[0].set_xlabel('Variance')
plt.tight_layout()
ax[1].legend(['fmri - connectivity',
              'fmri - BOLD variance'
          ])
ax[1].set_title('B')
fig.savefig('../figures/apchange_variance.png', dpi=400)


sex_diff = {}
for modality in img_modalities.keys():
    variables = img_modalities[modality].columns
    sex_diff[modality] = np.sum(var_df.loc[variables][('fligner_sex', 'p')] < 0.01) / len(variables)

print('Proportion of measures from each modality that exhibit significant sex differences in change scores:\n',
      sex_diff)


# ## Visualizing brain heterogeneity across non-brain variables
# 1. Variability across all brain measures
# 2. Per modality
# 3. Across the brain
# 4. Across developmental variables
# 5. Across demographic variables


for var in var_df.index:
    for modality in img_modalities.keys():
        if var in img_modalities[modality]:
            var_df.at[var, 'modality'] = modality


var_df[var_df['modality'] == 'fmri'][('variance', 'stat')].sort_values()

devt = ['fligner_age', 
        'fligner_sex',
        'fligner_puberty_f',
        'fligner_puberty_m',
        'fligner_puberty_1',
        'fligner_puberty_2',
        'fligner_puberty_3']
demo =  ['fligner_raceth',
         'fligner_income',
         'fligner_edu',
         'fligner_marital', 
         #'fligner_scanner'
        ]

stats = var_df.drop(['variance'], axis=1).xs('stat', level=1, axis=1)
alphas = var_df.xs(f'a<{alpha}', level=1, axis=1)
modalities = var_df['modality']

alphas = alphas.add_suffix('_alpha')


demo_alphas = [f'{i}_alpha' for i in demo]
devt_alphas = [f'{i}_alpha' for i in devt]

mod_demo = pd.concat([stats[demo], modalities], axis=1).melt(value_vars=demo, 
                                                  value_name='Flinger-Killeen Statistic',
                                                  id_vars='modality').drop('variable', axis=1)
alpha_demo = alphas[demo_alphas].melt(value_name='Significant')
demo_flinger = pd.concat([mod_demo, alpha_demo], axis=1)

mod_devt = pd.concat([stats[devt], modalities], axis=1).melt(value_vars=devt, 
                                                  value_name='Flinger-Killeen Statistic',
                                                  id_vars='modality')
alpha_devt = alphas[devt_alphas].melt(value_name='Significant').drop('variable', axis=1)
devt_flinger = pd.concat([mod_devt, alpha_devt], axis=1)


fig,ax = plt.subplots(nrows=2, figsize=(16,18))
g = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=devt_flinger[devt_flinger['Significant'] == '**'], 
                  hue='modality',
                  marker='o',
                  size=7,
                  edgecolor='white',
                  dodge=True,
                  linewidth=0.5,
                  ax=ax[0],
                  palette=crayons,
                  hue_order=['smri', 
                           'dti', 
                           'rsi', 
                           'fmri']
                 )
k = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=devt_flinger[devt_flinger['Significant'] != '**'], 
                  hue='modality',
                  marker='P',
                  size=11,
                  linewidth=0.5,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[0],
                  palette=crayons,
                  hue_order=['smri', 
                           'dti', 
                           'rsi', 
                           'fmri']
                 )
g.get_legend().remove()
g.set_ylabel('Flinger-Killeen Statistic')
g.set_xlabel('Measure')
g.set_xticklabels(['Age', 'Sex', 'Puberty (f)', 'Puberty (m)', 'Prepubertal', 'Early\nPuberty', 'Midpubertal'])
g.set_title('Heteroscedasticity Across Developmental Variables, Per Modality')

h = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=demo_flinger[demo_flinger['Significant'] == '**'], 
                  hue='modality',
                    marker='o',
                  size=7,
                  linewidth=0.5,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[1],
                  palette=crayons,
                  hue_order=['smri', 
                           'dti', 
                           'rsi', 
                           'fmri'],
              
                 )
j = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=demo_flinger[demo_flinger['Significant'] != '**'], 
                  hue='modality',
                    marker='P',
                  size=11,
                  linewidth=0.5,
                  dodge=True,
                  edgecolor='white',
                  ax=ax[1],
                  palette=crayons,
                  hue_order=['smri', 
                           'dti', 
                           'rsi', 
                           'fmri'],
              
                 )
h.get_legend().remove()
h.set_ylabel('Flinger-Killeen Statistic')
h.set_xlabel('Measure')
h.set_title('Heteroscedasticity Across Demographic Variables, Per Modality')
h.set_xticklabels(['Race &\nEthnicity', 
                   'Household\nIncome', 
                   'Parent\nEducation', 
                   'Parent\nMarital Status', 
                   #'Scanner\nManufacturer'
                  ])
fig.show()
fig.savefig('../figures/heteroscedasticity_modality.png', dpi=400)

