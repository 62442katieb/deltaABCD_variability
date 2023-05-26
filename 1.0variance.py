#!/usr/bin/env python
# coding: utf-8

import enlighten
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt


from os.path import join
from scipy.stats import fligner
from matplotlib.gridspec import GridSpec
from utils import jili_sidak_mc, assign_region_names

sns.set(style='whitegrid', context='talk')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'


crayons = sns.crayon_palette(['Aquamarine', 'Fuchsia', 
                              'Jungle Green', 'Yellow Green'])


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data_qcd.csv"), index_col=0, header=0)


df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
df.drop(list(df.filter(regex='.*_cf12_.*').columns), axis=1, inplace=True)
no_2yfu = df[df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
df = df.drop(no_2yfu, axis=0)

deltasmri_complete = df.filter(regex='smri.*change_score')
deltarsfmri_complete = df.filter(regex='rsfmri.*change_score').dropna(how='any')
deltarsi_complete = df.filter(regex='dmri_rsi.*change_score').dropna()
deltadti_complete = df.filter(regex='dmri_dti.*change_score').dropna()

img_modalities = {'smri': deltasmri_complete,
                  'fmri': deltarsfmri_complete,
                  'rsi': deltarsi_complete, 
                  'dti': deltadti_complete}

tests = ['variance', 
         'fligner_sex',
         'fligner_puberty',
         'fligner_raceth', 
         'fligner_income', 
         'fligner_edu', 
         'fligner_marital', 
         'fligner_age', 
         'fligner_scanner'
        ]



# should I correct alpha for multiple comparisons?
alpha, _ = jili_sidak_mc(df.filter(regex='.*mri.*change_score'), 0.05)

values = ['stat', 'p', 'diff', 'greater', f'sig']
columns = pd.MultiIndex.from_product([tests, values])

var_df = pd.DataFrame(columns=columns)

ages = df['interview_age.baseline_year_1_arm_1'].sort_values().dropna().unique()
all_ages = df['interview_age.baseline_year_1_arm_1'].sort_values().dropna()
q1 = np.quantile(all_ages, 0.25)
q2 = np.quantile(all_ages, 0.5)
q3 = np.quantile(all_ages, 0.75)
age_bins = [ages[0], q1, q2, q3, ages[-1]]

# combine female and male pubertal development scores into one variable
df[['pds_p_ss_female_category_2.baseline_year_1_arm_1',
    'pds_p_ss_male_category_2.baseline_year_1_arm_1']]
df['pds_p_ss_category_2.baseline_year_1_arm_1'] = df['pds_p_ss_female_category_2.baseline_year_1_arm_1'].fillna(0) + df['pds_p_ss_male_category_2.baseline_year_1_arm_1'].fillna(0)
df['pds_p_ss_category_2.baseline_year_1_arm_1'].replace({0:np.nan}, inplace=True)

# added plots of pubertal stage by age for
# all ppts, afab ppts, and amab ppts
# per reviewer request
fig,ax = plt.subplots(figsize=(7,5))
g = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.baseline_year_1_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df, ax=ax, fill=True, multiple='stack', palette='husl')
sns.move_legend(g, "center left", bbox_to_anchor=(1.0, 0.5))
g.get_legend().set_title('Pubertal Stage')
g.set_xlabel('Age (months)')
g.set_ylabel('Density')
g.set_xlim(left=100, right=140)
fig.savefig(join(PROJ_DIR,
                   FIGS_DIR,
                   'puberty_x_age-dist.png'), dpi=500, bbox_inches='tight')

# name the pubertal & demographic variables (baseline)
sex = 'sex.baseline_year_1_arm_1'
puberty = 'pds_p_ss_category_2.baseline_year_1_arm_1' # new combined score
race = 'race_ethnicity.baseline_year_1_arm_1'
income = 'demo_comb_income_v2.baseline_year_1_arm_1'
edu = 'demo_prnt_ed_v2.baseline_year_1_arm_1'
age = 'interview_age.baseline_year_1_arm_1'
marry = "demo_prnt_marital_v2.baseline_year_1_arm_1"
mri = 'mri_info_manufacturer.baseline_year_1_arm_1'

num_vars = len(df.filter(regex='.*mri.*change_score').columns)
manager = enlighten.get_manager()
tocks = manager.counter(total=num_vars, desc='Hsk Progress', unit='variables')

for modality in img_modalities.keys():
    variables = img_modalities[modality].columns
    for var in variables:
        # compute variance across the sample
        var_df.at[var, ('variance', 'stat')] = np.var(df[var])
        
        # compare variance between male and female participants
        m = df[df[sex] == 'M'][var].dropna()
        f = df[df[sex] == 'F'][var].dropna()
        test = fligner(m, f)
        var_df.at[var, ('fligner_sex', 'stat')] = test[0]
        var_df.at[var, ('fligner_sex', 'p')] = test[1]
        var_df.at[var, ('fligner_sex', 'diff')] = np.mean(f) - np.mean(m)
        var_df.at[var, ('fligner_sex', 'greater')] = 'f'
        if test[1] < alpha:
            var_df.at[var, ('fligner_sex', f'sig')] = '**'
        else:
            var_df.at[var, ('fligner_sex', f'sig')] = 'ns'
        
        # compare variance between pubertal stages at baseline
        # do not re-bin, unequal group sizes will have to be okay
        one = df[df[puberty] == 1.][var].dropna()
        two = df[df[puberty] == 2.][var].dropna()
        three = df[df[puberty] == 3.][var].dropna()
        
        test = fligner(one, two, three)
        var_df.at[var, ('fligner_puberty', 'stat')] = test[0]
        var_df.at[var, ('fligner_puberty', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_puberty', f'sig')] = '**'
        else:
            var_df.at[var, ('fligner_puberty', f'sig')] = 'ns'
        
        var_dict = {'1': np.mean(one), 
                    '2': np.mean(two), 
                    '3': np.mean(three)}
        max_key = max(var_dict, key=var_dict.get)
        min_val = min(var_dict.values())
        #LEAST TO MOST
        sorted_keys = {k: v for k, v in sorted(var_dict.items(), key=lambda item: item[1])}
        var_df.at[var, ('fligner_puberty', 'diff')] = var_dict[max_key] - min_val
        var_df.at[var, ('fligner_puberty', 'greater')] = [list(sorted_keys.keys())]
        
        
        # compare variance across race/ethnicities
        # do not re-bin, unequal group sizes will have to be okay
        white = df[df[race] == 1.][var].dropna()
        black = df[df[race] == 2.][var].dropna()
        hispanic = df[df[race] == 3.][var].dropna()
        asian = df[df[race] == 4.][var].dropna()
        other = df[df[race] == 5.][var].dropna()
        asian_other = df[df[race].between(4., 5.)][var].dropna()
        test = fligner(white, black, hispanic, asian_other)
        var_df.at[var, ('fligner_raceth', 'stat')] = test[0]
        var_df.at[var, ('fligner_raceth', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_raceth', f'sig')] = '**'
        else:
            var_df.at[var, ('fligner_raceth', f'sig')] = 'ns'
        
        # compare variance across income
        # re-bin for approximately equal group sizes
        lt75k = df[df[income] < 7.][var].dropna()
        lt100 = df[df[income].between(7., 8.)][var].dropna()
        gt100 = df[df[income].between(9., 10.)][var].dropna()
        
        test = fligner(lt75k, lt100, gt100)
        var_df.at[var, ('fligner_income', 'stat')] = test[0]
        var_df.at[var, ('fligner_income', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_income', f'sig')] = '**'
        else:
            var_df.at[var, ('fligner_income', f'sig')] = 'ns'
        
        # compare variance across education
        # re-bin for approximately equal group sizes
        hs_ged = df[df[edu] <= 14][var].dropna()
        lt_4yuni = df[df[edu].between(15, 17)][var].dropna()
        bachelors = df[df[edu] == 18][var].dropna()
        graduate = df[df[edu].between(19, 22)][var].dropna()
        
        test = fligner(hs_ged, lt_4yuni, bachelors, graduate)
        var_df.at[var, ('fligner_edu', 'stat')] = test[0]
        var_df.at[var, ('fligner_edu', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_edu', f'sig')] = '**'
        else:
            var_df.at[var, ('fligner_edu', f'sig')] = 'ns'
        
        # compare variance across age, binned by n0th percentiles
        # age_bins are calculated above
        one = df[df[age] <= age_bins[1]][var].dropna()
        two = df[df[age].between(age_bins[1], age_bins[2], inclusive='left')][var].dropna()
        three = df[df[age].between(age_bins[2], age_bins[3], inclusive='left')][var].dropna()
        four = df[df[age].between(age_bins[3], age_bins[4], inclusive='left')][var].dropna()
                 
        test = fligner(one, two, three, four)
        var_df.at[var, ('fligner_age', 'stat')] = test[0]
        var_df.at[var, ('fligner_age', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_age', f'sig')] = '**'
        else:
            var_df.at[var, ('fligner_age', f'sig')] = 'ns'
        
         # compare variance across scanner manufacturers
         # not for the main brain change variability paper
        siemens = df[df[mri] == 'SIEMENS'][var].dropna()
        ge = df[df[mri] == 'GE MEDICAL SYSTEMS'][var].dropna()
        philips = df[df[mri] == 'Philips Medical Systems'][var].dropna()
        
        test = fligner(siemens, philips, ge)
        var_df.at[var, ('fligner_scanner', 'stat')] = test[0]
        var_df.at[var, ('fligner_scanner', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_scanner', f'sig')] = '**'
        else:
            var_df.at[var, ('fligner_scanner', f'sig')] = 'ns'
        
        # compare variance across parent marital status
        # re-bin for group sizes
        married = df[df[marry] == 1.][var].dropna()
        notmarried = df[df[marry].between(2,5, inclusive='both')][var].dropna()
        #widowed = df[df[marry] == 2.][var].dropna()
        #divorced = df[df[marry] == 3.][var].dropna()
        #separated = df[df[marry] == 4.][var].dropna()
        #never = df[df[marry] == 5.][var].dropna()
        #refuse = df[df[marry] == 777.][var].dropna()
        
        test = fligner(married, notmarried
                       #widowed, 
                       #separated, 
                       #divorced, 
                       #never, 
                       #refuse
                      )
        var_df.at[var, ('fligner_marital', 'stat')] = test[0]
        var_df.at[var, ('fligner_marital', 'p')] = test[1]
        if test[1] < alpha:
            var_df.at[var, ('fligner_marital', f'sig')] = '**'
        else:
            var_df.at[var, ('fligner_marital', f'sig')] = 'ns'
        tocks.update()


var_df.dropna(how='all', axis=1, inplace=True)

# calculate what proportion of measures show significant heteroscedasticity
# just count('**') / # measures
var_df.to_csv(join(PROJ_DIR, OUTP_DIR, f'variance_flinger-alpha<{np.round(alpha, 2)}.csv'))
new_var_df = assign_region_names(var_df)
var_df.to_csv(join(PROJ_DIR, OUTP_DIR, f'variance_flinger-alpha<{np.round(alpha, 2)}-regions.csv'))

concepts = {'morph': ['thick', 
                      'area', 
                      'vol',
                      'gmvol',
                      'dtivol'],
            'cell': ['t1wcnt', 
                     'rsirni', 
                     'rsirnd',
                     'rsirnigm', 
                     'rsirndgm',
                     'dtifa', 
                     'dtimd',
                     'dtild', 
                     'dtitd'],
            'func':['var',
                    'c',
                     'cor']}

for i in var_df.index:
    measure = var_df.loc[i]['measure']
    measure = str(measure.values[0])
    if measure in concepts['morph']:
        var_df.at[i,'concept'] = 'macrostructure'
    elif measure in concepts['cell']:
        var_df.at[i,'concept'] = 'microstructure'
    elif measure in concepts['func']:
        var_df.at[i,'concept'] = 'function'

# plot the distribution of variances of all structural mri measures
smri_var = img_modalities['smri'].columns
dti_var = img_modalities['dti'].columns
rsi_var = img_modalities['rsi'].columns
# separate wm and gm rsi
rsi_gm = list(img_modalities['rsi'].filter(regex='.*gm').columns) + list(img_modalities['rsi'].filter(regex='.*scs').columns)
rsi_wm = list(set(rsi_var) - set(rsi_gm))
fmri_var = img_modalities['fmri'].columns
fc_cort_var = img_modalities['fmri'].filter(regex='_c_.*').columns
fc_scor_var = img_modalities['fmri'].filter(regex='_cor_.*').columns
fmri_var_var = img_modalities['fmri'].filter(regex='_var_.*').columns

morph_var = var_df[var_df['concept'] == 'macrostructure'].index
cell_var = var_df[var_df['concept'] == 'microstructure'].index
func_var = list(fmri_var_var) 
conn_var = list(fc_cort_var) + list(fc_scor_var)

btwn_fc = []
wthn_fc = []
for var in fc_cort_var:
    var_list = var[:-13].split('_')
    #print(var_list)
    if var_list[3] == var_list[5]:
        #print(var, 'within-network')
        wthn_fc.append(var)
    else:
        btwn_fc.append(var)
        #print(var, 'between-network')

# plot variance in brain changes by biological concept
fig,ax = plt.subplots(ncols=2, figsize=(15,5))
g = sns.kdeplot(var_df.loc[morph_var, ('variance', 'stat')], color=crayons[0], shade=True, ax=ax[0])
h = sns.kdeplot(var_df.loc[cell_var, ('variance', 'stat')], color=crayons[1], shade=True, ax=ax[0])
#i = sns.kdeplot(var_df.loc[func_var, ('variance', 'stat')], color=crayons[2], shade=True, ax=ax[0])
#m = sns.kdeplot(var_df.loc[fmri_var, ('variance', 'stat')], color=crayons[3])
j = sns.rugplot(var_df.loc[morph_var, ('variance', 'stat')], color=crayons[0], lw=1, alpha=.1, ax=ax[0])
k = sns.rugplot(var_df.loc[cell_var, ('variance', 'stat')], color=crayons[1], lw=1, alpha=0.2, ax=ax[0])
#l = sns.rugplot(var_df.loc[rsi_var, ('variance', 'stat')], color=crayons[2], lw=1, alpha=.1, ax=ax[0])
#n = sns.rugplot(var_df.loc[fmri_var, ('variance', 'stat')], color=crayons[3])
ax[0].set_xlabel('Variance')
plt.tight_layout()
ax[0].legend(['macrostructure', 
           'microstructure', 
           #'rsi', 
           #'fmri'
          ])
ax[0].set_title('A')

m = sns.kdeplot(var_df.loc[func_var, ('variance', 'stat')], color=crayons[2], shade=True, ax=ax[1])
o = sns.kdeplot(var_df.loc[conn_var, ('variance', 'stat')], color=crayons[3], shade=True, ax=ax[1])
n = sns.rugplot(var_df.loc[func_var, ('variance', 'stat')], color=crayons[2], lw=1, alpha=.1, ax=ax[1])
p = sns.rugplot(var_df.loc[conn_var, ('variance', 'stat')], color=crayons[3], lw=1, alpha=.1, ax=ax[1])
ax[1].set_xlabel('Variance')
plt.tight_layout()
ax[1].legend(['function',
              'functional connectivity'
          ])
ax[1].set_title('B')
fig.savefig(f'{PROJ_DIR}/figures/apchange_variance_concept.png', dpi=400)



# ## Visualizing brain heterogeneity across non-brain variables
# 1. Variability across all brain measures
# 2. Per modality
# 3. Across the brain
# 4. Across developmental variables
# 5. Across demographic variables

## SECOND COPY PASTA ##
# add region names


var_df[var_df['modality'] == 'fmri'][('variance', 'stat')].sort_values()

#need to separate scanner into its own thing
devt = ['fligner_age', 
        'fligner_sex',
        'fligner_puberty']
demo =  ['fligner_raceth',
         'fligner_income',
         'fligner_edu',
         'fligner_marital', 
         #'fligner_scanner'
        ]
scan = ['fligner_scanner']

stats = var_df.drop(['variance'], axis=1).xs('stat', level=1, axis=1)
alphas = var_df.xs('sig', level=1, axis=1)
modalities = var_df['concept']

alphas = alphas.add_suffix('_alpha')

demo_alphas = [f'{i}_alpha' for i in demo]
devt_alphas = [f'{i}_alpha' for i in devt]
scan_alphas = [f'{i}_alpha' for i in scan]

mod_demo = pd.concat([stats[demo], modalities], axis=1).melt(value_vars=demo, 
                                                  value_name='Flinger-Killeen Statistic',
                                                  id_vars='concept').drop('variable', axis=1)
alpha_demo = alphas[demo_alphas].melt(value_name='Significant')
demo_flinger = pd.concat([mod_demo, alpha_demo], axis=1)

mod_devt = pd.concat([stats[devt], modalities], axis=1).melt(value_vars=devt, 
                                                  value_name='Flinger-Killeen Statistic',
                                                  id_vars='concept')
alpha_devt = alphas[devt_alphas].melt(value_name='Significant').drop('variable', axis=1)
devt_flinger = pd.concat([mod_devt, alpha_devt], axis=1)

mod_scan = pd.concat([stats[scan], modalities], axis=1).melt(value_vars=scan, 
                                                  value_name='Flinger-Killeen Statistic',
                                                  id_vars='concept')
alpha_scan = alphas[scan_alphas].melt(value_name='Significant').drop('variable', axis=1)
scan_flinger = pd.concat([mod_scan, alpha_scan], axis=1)

# need to remake this with a third columns for scanner heteroscedasticity.
# and redo the spacing between columns with a gridspec
fig = plt.figure(figsize=(25,6))

gs = GridSpec(1, 8)

ax0 = fig.add_subplot(gs[0:3])
ax1 = fig.add_subplot(gs[3:7])
ax2 = fig.add_subplot(gs[7])
plt.tight_layout(w_pad=3, h_pad=1)

g = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=devt_flinger[devt_flinger['Significant'] == '**'], 
                  hue='concept',
                  marker='o',
                  size=7,
                  edgecolor='white',
                  dodge=True,
                  linewidth=0.5,
                  ax=ax0,
                  palette=crayons,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
k = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=devt_flinger[devt_flinger['Significant'] != '**'], 
                  hue='concept',
                  marker='P',
                  size=11,
                  linewidth=0.5,
                  edgecolor='white',
                  dodge=True,
                  ax=ax0,
                  palette=crayons,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
g.get_legend().remove()
g.set_ylabel('Flinger-Killeen Statistic')
g.set_xlabel('')
g.set_xticklabels(['Age', 'Sex', 'Puberty'])


h = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=demo_flinger[demo_flinger['Significant'] == '**'], 
                  hue='concept',
                    marker='o',
                  size=7,
                  linewidth=0.5,
                  edgecolor='white',
                  dodge=True,
                  ax=ax1,
                  palette=crayons,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function'],
              
                 )
j = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=demo_flinger[demo_flinger['Significant'] != '**'], 
                  hue='concept',
                    marker='P',
                  size=11,
                  linewidth=0.5,
                  dodge=True,
                  edgecolor='white',
                  ax=ax1,
                  palette=crayons,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function'],
              
                 )
handles, labels = h.get_legend_handles_labels()
h.legend(handles[:3], labels [:3], bbox_to_anchor=(0.5, -0.15), ncol=3)
h.set_ylabel('')
h.set_xlabel('')

h.set_xticklabels(['Race &\nEthnicity', 
                   'Household\nIncome', 
                   'Caregiver\nEducation', 
                   'Caregiver\nMarital\nStatus', 
                   #'Scanner\nManufacturer'
                  ])

l = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=scan_flinger[scan_flinger['Significant'] == '**'], 
                  hue='concept',
                  marker='o',
                  size=7,
                  edgecolor='white',
                  dodge=True,
                  linewidth=0.5,
                  ax=ax2,
                  palette=crayons,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
k = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=scan_flinger[scan_flinger['Significant'] != '**'], 
                  hue='concept',
                  marker='P',
                  size=11,
                  linewidth=0.5,
                  edgecolor='white',
                  dodge=True,
                  ax=ax2,
                  palette=crayons,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
l.get_legend().remove()
l.set_ylabel('')
l.set_xlabel('')
l.set_xticklabels(['MRI\nManufacturer'])
fig.show()
fig.savefig(f'{PROJ_DIR}/figures/heteroscedasticity_concept.png', dpi=400, bbox_inches="tight")