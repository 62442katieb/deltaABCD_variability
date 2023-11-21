#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import fligner
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from os.path import join, exists

PROJ_DIR = '/Volumes/Projects_Herting/LABDOCS/Personnel/Katie/deltaABCD_clustering'
DATA_DIR = 'data'
OUTP_DIR = 'output'
FIGS_DIR = 'figures'

ABCD_DIR =  "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/csv"

crayons = sns.crayon_palette(['Aquamarine', 'Fuchsia', 
                              'Jungle Green', 'Yellow Green'])
dark_crayons = ['#4d6c80', '#7f4d80', '#4d8071']
dark = sns.color_palette(dark_crayons)

sns.set(style='white', context='talk')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'

variables = [
         "pds_p_ss_female_category_2",
         "pds_p_ss_male_category_2"
         ]
timepoints = [
    "baseline_year_1_arm_1",
    "1_year_follow_up_y_arm_1", 
    "2_year_follow_up_y_arm_1"
    ]

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

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd.pkl'))

df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
df.drop(list(df.filter(regex='.*_cf12_.*').columns), axis=1, inplace=True)
df.drop(list(df.filter(regex='.*_cortgordon_.*').columns), axis=1, inplace=True)

puberty = pd.read_csv(join(ABCD_DIR, "abcd_ssphp01.csv"), 
                      index_col="subjectkey", 
                      header=0)
temp_df = pd.DataFrame()
for timepoint in timepoints[1:]:
    print(timepoint)
    for variable in variables:
        if not f'{variable}.{timepoint}' in df.columns:
            temp_df[f'{variable}.{timepoint}'] = puberty[puberty['eventname'] == timepoint][variable]
        else:
            pass
temp_df = temp_df.astype(float)
df = pd.concat([df, temp_df], axis=1)
deltasmri_complete = df.filter(regex='smri.*change_score')
deltarsfmri_complete = df.filter(regex='rsfmri.*change_score').dropna(how='any')
deltarsi_complete = df.filter(regex='dmri_rsi.*change_score').dropna()
deltadti_complete = df.filter(regex='dmri_dti.*change_score').dropna()

img_modalities = {'smri': deltasmri_complete,
                  'fmri': deltarsfmri_complete,
                  'rsi': deltarsi_complete, 
                  'dti': deltadti_complete}

print(df.filter(regex="pds_p_ss_.*", axis=1).describe())

temp_df = pd.DataFrame()
for timepoint in timepoints:
    print('\n\n\n\n', timepoint, 
          '\n\n\nfemale mean', np.mean(df[f'pds_p_ss_female_category_2.{timepoint}'].dropna()), 
          '\n\n\nmale mean', np.mean(df[f'pds_p_ss_male_category_2.{timepoint}'].dropna()))
    temp_2 = df[f'pds_p_ss_female_category_2.{timepoint}'].fillna(0) + df[f'pds_p_ss_male_category_2.{timepoint}'].fillna(0)
    temp_2.name = f'pds_p_ss_category_2.{timepoint}'
    temp_2.replace({0: np.nan}, inplace=True)
    temp_df = pd.concat([temp_df, temp_2], axis=1)
print('\n\n', temp_df.columns, '\n\n')
df = pd.concat((df, temp_df), axis=1)
stages = [1., 2., 3.]

stage_idx = {
    "baseline_year_1_arm_1": {
        1: df[df['pds_p_ss_category_2.baseline_year_1_arm_1'] == 1].index,
        2: df[df['pds_p_ss_category_2.baseline_year_1_arm_1'] == 2].index,
        3: df[df['pds_p_ss_category_2.baseline_year_1_arm_1'] == 3].index,
        4: df[df['pds_p_ss_category_2.baseline_year_1_arm_1'] == 4].index,
        5: df[df['pds_p_ss_category_2.baseline_year_1_arm_1'] == 5].index
    },
    "1_year_follow_up_y_arm_1": {
        1: df[df['pds_p_ss_category_2.1_year_follow_up_y_arm_1'] == 1].index,
        2: df[df['pds_p_ss_category_2.1_year_follow_up_y_arm_1'] == 2].index,
        3: df[df['pds_p_ss_category_2.1_year_follow_up_y_arm_1'] == 3].index,
        4: df[df['pds_p_ss_category_2.1_year_follow_up_y_arm_1'] == 4].index,
        5: df[df['pds_p_ss_category_2.1_year_follow_up_y_arm_1'] == 5].index
    },
    "2_year_follow_up_y_arm_1": {
        1: df[df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] == 1].index,
        2: df[df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] == 2].index,
        3: df[df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] == 3].index,
        4: df[df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] == 4].index,
        5: df[df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] == 5].index
    },
}

fig,ax = plt.subplots(ncols=3, nrows=3, sharey=True, figsize=(21,12))
plt.tight_layout(w_pad=1., h_pad=3.)
g = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.baseline_year_1_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df, ax=ax[0,0], fill=True, multiple='stack', palette='husl')
g.set_title('All participants')
g.get_legend().remove()
g.set_xlabel('Age (months)')
g.set_ylabel('Density')
g.set_xlim(left=100, right=140)

h = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.1_year_follow_up_y_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df, ax=ax[1,0], fill=True, multiple='stack', palette='husl')
h.get_legend().remove()
h.set_xlabel('Age (months)')
h.set_ylabel('Density')
h.set_xlim(left=100, right=140)
h.set_title('Year 1')

i = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.2_year_follow_up_y_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df, ax=ax[2,0], fill=True, multiple='stack', palette='husl')

i.get_legend().remove()
i.set_xlabel('Age (months)')
i.set_ylabel('Density')
i.set_xlim(left=100, right=140)
i.set_title('Year 2')

g = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.baseline_year_1_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df[df['sex.baseline_year_1_arm_1'] == 'F'], ax=ax[0,1], 
                fill=True, multiple='stack', palette='husl')

g.get_legend().remove()
g.set_title('Female participants')
g.set_xlabel('Age (months)')
g.set_ylabel('Density')
g.set_xlim(left=100, right=140)

h = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.1_year_follow_up_y_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df[df['sex.baseline_year_1_arm_1'] == 'F'], ax=ax[1,1], 
                fill=True, multiple='stack', palette='husl')
h.get_legend().remove()
h.set_xlabel('Age (months)')
h.set_ylabel('Density')
h.set_xlim(left=100, right=140)


i = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.2_year_follow_up_y_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df[df['sex.baseline_year_1_arm_1'] == 'F'], ax=ax[2,1], fill=True, multiple='stack', palette='husl')


i.set_xlabel('Age (months)')
i.set_ylabel('Density')
i.set_xlim(left=100, right=140)
i.get_legend().set_title('Pubertal Stage')

sns.move_legend(i, "center left", bbox_to_anchor=(1.0, 0.5))

g = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.baseline_year_1_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df[df['sex.baseline_year_1_arm_1'] == 'M'], ax=ax[0,2], fill=True, multiple='stack', palette='husl')

g.get_legend().remove()
g.set_title('Male participants')
g.set_xlabel('Age (months)')
g.set_ylabel('Density')
g.set_xlim(left=100, right=140)

h = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.1_year_follow_up_y_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df[df['sex.baseline_year_1_arm_1'] == 'M'], ax=ax[1,2], fill=True, multiple='stack', palette='husl')
h.get_legend().remove()
h.set_xlabel('Age (months)')
h.set_ylabel('Density')
h.set_xlim(left=100, right=140)

i = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.2_year_follow_up_y_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df[df['sex.baseline_year_1_arm_1'] == 'M'], ax=ax[2,2], fill=True, multiple='stack', palette='husl')

i.get_legend().remove()
i.set_xlabel('Age (months)')
i.set_ylabel('Density')
i.set_xlim(left=100, right=140)

sns.despine()
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'puberty_x_age_x_sex-dist_timepoints.png'), dpi=500, bbox_inches='tight')

change_scores = df.filter(regex='mri.*change_score').columns

meas = ['stat', 'p']


f_timepoints = [f'f_{timepoint}' for timepoint in timepoints]
m_timepoints = [f'm_{timepoint}' for timepoint in timepoints]
changes = ['2yfu-1yfu', '1yfu-base', '2yfu-base']
f_changes = [f'f_{change}' for change in changes]
m_changes = [f'm_{change}' for change in changes]

all_timepoints = timepoints + f_timepoints + m_timepoints + changes + f_changes + m_changes

columns = pd.MultiIndex.from_product((all_timepoints, meas))
var_df = pd.DataFrame(columns=columns, index=change_scores)

for timepoint in timepoints:
    for var in change_scores:
        one = df[df[f'pds_p_ss_category_2.{timepoint}'] == 1.][var].dropna()
        two = df[df[f'pds_p_ss_category_2.{timepoint}'] == 2.][var].dropna()
        three = df[df[f'pds_p_ss_category_2.{timepoint}'] == 3.][var].dropna()

        test = fligner(one, two, three)
        var_df.at[var, (timepoint,'stat')] = test[0]
        var_df.at[var, (timepoint,'p')] = test[1]

for var in var_df.index:
    #print(var)
    if 'mrisdp' in var:
        var_num = int(var.split('.')[0].split('_')[-1])
        var_df.at[var, 'modality'] = 'smri'
        var_df.at[var, 'atlas'] = 'dtx'
        if var_num <= 148:
            var_df.at[var, 'measure'] = 'thick'
        elif var_num <= 450 and var_num >= 303:
            var_df.at[var, 'measure'] = 'area'
        elif var_num < 604 and var_num >= 450:
            var_df.at[var, 'measure'] = 'vol'
        elif var_num <= 1054 and var_num >= 907:
            var_df.at[var, 'measure'] = 't1wcnt'
        elif var_num == 604:
            var_df.at[var, 'measure'] = 'gmvol'
    elif '_' in var:
        var_list = var.split('.')[0].split('_')
        var_df.at[var, 'variable'] = var.split('.')[0]
        var_df.at[var, 'modality'] = var_list[0]
        var_df.at[var, 'measure'] = var_list[1]
        var_df.at[var, 'atlas'] = var_list[2]
        var_df.at[var, 'region'] = '_'.join(var_list[3:])

var_df = var_df[var_df['measure'] != 't1w']
var_df = var_df[var_df['measure'] != 't2w']
for i in var_df.index:
    measure = var_df.loc[i]['measure']
    measure = str(measure.values[0])
    if measure in concepts['morph']:
        var_df.at[i,'concept'] = 'macrostructure'
    elif measure in concepts['cell']:
        var_df.at[i,'concept'] = 'microstructure'
    elif measure in concepts['func']:
        var_df.at[i,'concept'] = 'function'

# need to remake this with a third columns for scanner heteroscedasticity.
# and redo the spacing between columns with a gridspec
fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(18,6))

plt.tight_layout(w_pad=1, h_pad=1)
i = sns.stripplot(y=('baseline_year_1_arm_1','stat'),
                  data=var_df[var_df[('baseline_year_1_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  ax=ax[0],
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
if len(var_df[var_df[('baseline_year_1_arm_1', 'p')] < 0.000096].index) > 0:
    g = sns.stripplot(y=('baseline_year_1_arm_1','stat'),
                      data=var_df[var_df[('baseline_year_1_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      #fill=True,
                      ax=ax[0],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    g.set_xlabel('')
    g.set_title('Pubertal stage at ages 9-10')
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles[3:], labels [3:], 
             bbox_to_anchor=(0., -0.15), 
             loc="center left",
             ncol=3, title='Significant')
else:
    pass

ax[0].set_ylabel('Density')

#k.set_ylabel('')
i.set_ylabel('Greater difference in \nvariability between levels')


j = sns.stripplot(y=('1_year_follow_up_y_arm_1','stat'),
                  data=var_df[var_df[('1_year_follow_up_y_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  ax=ax[1],
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function'],
              
                 )
if len(var_df[var_df[('1_year_follow_up_y_arm_1', 'p')] < 0.000096].index) > 0:
    h = sns.stripplot(y=('1_year_follow_up_y_arm_1','stat'),
                      data=var_df[var_df[('1_year_follow_up_y_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ax=ax[1],
                    #fill=True,
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                   )
else:
    pass

j.set_title('Pubertal stage at ages 10-11')
j.get_legend().remove()
j.set_xlabel('')
j.set_ylabel('')
k = sns.stripplot(y=('2_year_follow_up_y_arm_1','stat'),
                  data=var_df[var_df[('2_year_follow_up_y_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[2],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
if len(var_df[var_df[('2_year_follow_up_y_arm_1', 'p')] < 0.000096].index) > 0:
    l = sns.stripplot(y=('2_year_follow_up_y_arm_1','stat'),
                      data=var_df[var_df[('2_year_follow_up_y_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      #fill=True,
                      ax=ax[2],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                     hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    l.get_legend().remove()
    l.set_ylabel('')
    #l.set_xlabel('')
else: 
    pass


k.set_ylabel('')
k.set_xlabel('')
k.set_title('Pubertal stage at ages 11-12')

handles, labels = k.get_legend_handles_labels()
k.legend(handles[:3], labels [:3], 
         bbox_to_anchor=(1., -0.15), 
         loc="center right",
         ncol=3, title='Not Significant')

sns.despine()
#fig.show()
fig.savefig(f'{PROJ_DIR}/figures/heteroscedasticity_puberty.png', dpi=400, bbox_inches="tight")

fig,ax = plt.subplots(ncols=3, figsize=(21,5))
plt.tight_layout(w_pad=1.)
g = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.baseline_year_1_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df, ax=ax[0], fill=True, multiple='stack', palette='husl')
sns.move_legend(g, "center left", bbox_to_anchor=(1.0, 0.5))
g.get_legend().set_title('Pubertal Stage')
g.set_xlabel('Age (months)')
g.set_ylabel('Density')
g.set_xlim(left=100, right=140)

h = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.1_year_follow_up_y_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df, ax=ax[1], fill=True, multiple='stack', palette='husl')
sns.move_legend(g, "center left", bbox_to_anchor=(1.0, 0.5))
h.get_legend().set_title('Pubertal Stage')
h.set_xlabel('Age (months)')
h.set_ylabel('Density')
h.set_xlim(left=100, right=140)

i = sns.kdeplot(x='interview_age.baseline_year_1_arm_1', 
                hue='pds_p_ss_category_2.2_year_follow_up_y_arm_1', 
                #hue='sex.baseline_year_1_arm_1', 
                 data=df, ax=ax[2], fill=True, multiple='stack', palette='husl')
sns.move_legend(g, "center left", bbox_to_anchor=(1.0, 0.5))
i.get_legend().set_title('Pubertal Stage')
i.set_xlabel('Age (months)')
i.set_ylabel('Density')
i.set_xlim(left=100, right=140)

fig.savefig(join(PROJ_DIR,
                 FIGS_DIR,
                 'puberty_x_age-dist_timepoints.png'), 
            dpi=500, 
            bbox_inches='tight')

change_scores = df.filter(regex='mri.*change_score').columns

f_df = df[df['sex.baseline_year_1_arm_1'] == 'F']
m_df = df[df['sex.baseline_year_1_arm_1'] == 'M']

for timepoint in timepoints:
    for var in change_scores:
        one = df[df[f'pds_p_ss_category_2.{timepoint}'] == 1.][var]
        one = one.dropna()
        two = df[df[f'pds_p_ss_category_2.{timepoint}'] == 2.][var]
        two = two.dropna()
        three = df[df[f'pds_p_ss_category_2.{timepoint}'] == 3.][var]
        three = three.dropna()

        test = fligner(one, two, three)
        var_df.at[var, (f'{timepoint}','stat')] = test[0]
        var_df.at[var, (f'{timepoint}','p')] = test[1]

        one = m_df[m_df[f'pds_p_ss_category_2.{timepoint}'] == 1.][var].dropna()
        two = m_df[m_df[f'pds_p_ss_category_2.{timepoint}'] == 2.][var].dropna()
        three = m_df[m_df[f'pds_p_ss_category_2.{timepoint}'] == 3.][var].dropna()

        test = fligner(one, two, three)
        var_df.at[var, (f'm_{timepoint}','stat')] = test[0]
        var_df.at[var, (f'm_{timepoint}','p')] = test[1]
        
        one = f_df[f_df[f'pds_p_ss_category_2.{timepoint}'] == 1.][var].dropna()
        two = f_df[f_df[f'pds_p_ss_category_2.{timepoint}'] == 2.][var].dropna()
        three = f_df[f_df[f'pds_p_ss_category_2.{timepoint}'] == 3.][var].dropna()

        test = fligner(one, two, three)
        var_df.at[var, (f'f_{timepoint}','stat')] = test[0]
        var_df.at[var, (f'f_{timepoint}','p')] = test[1]

prop_hsk_df = pd.DataFrame(index=var_df[('measure', '')].unique(), 
                           columns=all_timepoints)
for measure in var_df[('measure', '')].unique():
    temp_df = var_df[var_df[('measure', '')] == measure]
    for timepoint in all_timepoints:
        prop_hsk = np.sum(temp_df[(timepoint, 'p')] < 0.000096) / len(temp_df.index)
        prop_hsk_df.at[measure,timepoint] = prop_hsk
prop_hsk_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'puberty_proportion_heteroscedastic.csv'))

# need to remake this with a third columns for scanner heteroscedasticity.
# and redo the spacing between columns with a gridspec
fig, ax = plt.subplots(ncols=3, nrows=2, sharey=True, sharex=True, figsize=(18,12))

plt.tight_layout(w_pad=1, h_pad=3)
i = sns.stripplot(y=('f_baseline_year_1_arm_1','stat'),
                  data=var_df[var_df[('f_baseline_year_1_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  ##fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[0,0],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
if len(var_df[var_df[('f_baseline_year_1_arm_1', 'p')] < 0.000096].index) > 0:
    g = sns.stripplot(y=('f_baseline_year_1_arm_1','stat'),
                      data=var_df[var_df[('f_baseline_year_1_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ##fill=True,
                      ax=ax[0,0],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles[3:], labels [3:], 
             bbox_to_anchor=(0., -1.33), 
             loc="center left",
             ncol=3, title='Significant')
    g.set_title('Female participants\nPubertal stage at ages 9-10,')
else:
    pass
i.set_ylabel('Greater difference in \nvariability between stages')

j = sns.stripplot(y=('f_1_year_follow_up_y_arm_1','stat'),
                  data=var_df[var_df[('f_1_year_follow_up_y_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  ##fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[0,1],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function'],
              
                 )
if len(var_df[var_df[('f_1_year_follow_up_y_arm_1', 'p')] < 0.000096].index) > 0:
    h = sns.stripplot(y=('f_1_year_follow_up_y_arm_1','stat'),
                      data=var_df[var_df[('f_1_year_follow_up_y_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ax=ax[0,1],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                    ##fill=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                   )
else:
    pass

j.set_title('ages 10-11,')
j.get_legend().remove()
j.set_xlabel('')
j.set_ylabel('')

if len(var_df[var_df[('f_2_year_follow_up_y_arm_1', 'p')] < 0.000096].index) > 0:
    l = sns.stripplot(y=('f_2_year_follow_up_y_arm_1','stat'),
                      data=var_df[var_df[('f_2_year_follow_up_y_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ##fill=True,
                      ax=ax[0,2],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                     hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
else:
    pass
k = sns.stripplot(y=('f_2_year_follow_up_y_arm_1','stat'),
                  data=var_df[var_df[('f_2_year_follow_up_y_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  ##fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[0,2],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )

k.set_ylabel('')
k.set_xlabel('')
k.set_title('& ages 11-12')
k.get_legend().remove()
l.set_ylabel('')



i = sns.stripplot(y=('m_baseline_year_1_arm_1','stat'),
                  data=var_df[var_df[('m_baseline_year_1_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[1,0],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
i.get_legend().remove()
i.set_ylabel('Greater difference in \nvariability between stages')
i.set_title('Male participants\nPubertal stage at ages 9-10')
if len(var_df[var_df[('m_baseline_year_1_arm_1', 'p')] < 0.000096].index) > 0:
    g = sns.stripplot(y=('m_baseline_year_1_arm_1','stat'),
                      data=var_df[var_df[('m_baseline_year_1_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      #fill=True,
                      ax=ax[1,0],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    
    g.get_legend().remove()
else:
    pass



j = sns.stripplot(y=('m_1_year_follow_up_y_arm_1','stat'),
                  data=var_df[var_df[('m_1_year_follow_up_y_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[1,1],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function'],
              
                 )
if len(var_df[var_df[('m_1_year_follow_up_y_arm_1', 'p')] < 0.000096].index):
    h = sns.stripplot(y=('m_1_year_follow_up_y_arm_1','stat'),
                      data=var_df[var_df[('m_1_year_follow_up_y_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ax=ax[1,1],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                    #fill=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                   )
else:
    pass

j.set_title('ages 10-11,')
j.get_legend().remove()
j.set_xlabel('')
j.set_ylabel('')
k = sns.stripplot(y=('m_2_year_follow_up_y_arm_1','stat'),
                  data=var_df[var_df[('m_2_year_follow_up_y_arm_1', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[1,2],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
if len(var_df[var_df[('m_2_year_follow_up_y_arm_1', 'p')] < 0.000096].index):
    l = sns.stripplot(y=('m_2_year_follow_up_y_arm_1','stat'),
                      data=var_df[var_df[('m_2_year_follow_up_y_arm_1', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      #fill=True,
                      ax=ax[1,2],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                     hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    l.get_legend().remove()
    l.set_ylabel('')
else:
    pass

k.set_ylabel('')
k.set_xlabel('')
k.set_title('& ages 11-12')

handles, labels = k.get_legend_handles_labels()
k.legend(handles[:3], labels [:3], 
         bbox_to_anchor=(1., -0.14), 
         loc="center right",
         ncol=3, title='Not Significant')


#l.set_xlabel('')
sns.despine()
#fig.show()
fig.savefig(f'{PROJ_DIR}/figures/heteroscedasticity_puberty_by_sex.png', dpi=400, bbox_inches="tight")

# changes in puberty
y2_y1 = df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] - df['pds_p_ss_category_2.1_year_follow_up_y_arm_1']
y1_b = df['pds_p_ss_category_2.1_year_follow_up_y_arm_1'] - df['pds_p_ss_category_2.baseline_year_1_arm_1']
y2_b = df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] - df['pds_p_ss_category_2.baseline_year_1_arm_1']

y2_y1.name = '2yfu-1yfu'
y1_b.name = '1yfu-base'
y2_b.name = '2yfu-base'

puberty_changes = pd.concat([y2_y1, y1_b, y2_b], axis=1)

long = puberty_changes.melt(value_name='deltaPDS', var_name='change')

sexes = pd.concat([df['sex.baseline_year_1_arm_1'], 
           df['sex.baseline_year_1_arm_1']], axis=1)

pds_sex = pd.concat([long, sexes.melt(value_name='sex', 
                              var_name='drop')], 
            axis=1).drop('drop', 
                         axis=1)

temp = pd.concat([puberty_changes["2yfu-base"], 
           df['sex.baseline_year_1_arm_1']], axis=1)


fig,ax = plt.subplots(ncols=2, sharey=False, sharex=True, figsize=(12.5,5))
g = sns.histplot(x='2yfu-base', 
                 data=temp, 
                 hue='sex.baseline_year_1_arm_1', 
                 multiple="fill", 
                 binwidth=1, 
                 palette='husl',
                fill=True,
                
                 ax=ax[1])
h = sns.histplot(x='2yfu-base', 
                 data=temp, 
                 #hue='sex.baseline_year_1_arm_1', 
                 multiple="stack", 
                 binwidth=1, 
                 
                fill=True,
                 ax=ax[0])

g.set_xlabel('PDS Change')
h.set_xlabel('PDS Change')
g.set_ylabel('Proportion')
g.set_xlim(-2,4)
h.set_xlim(-2,4)
h.set_title('All participants')
g.set_xticks([-2, -1, 0, 1, 2, 3, 4])
g.get_legend().set_title('Sex')


sns.despine()
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'change_in_puberty_across_timepoints_x_sex.png'), dpi=500, bbox_inches='tight')

# var_df for deltaPDS
#meas = ['stat', 'p']
#f_timepoints = [f'f_{timepoint}' for timepoint in timepoints]
#m_timepoints = [f'm_{timepoint}' for timepoint in timepoints]

#all_timepoints = f_timepoints + m_timepoints
#columns = pd.MultiIndex.from_product((all_timepoints, meas))
#s_var_df = pd.DataFrame(columns=columns)
temp = pd.concat([df, puberty_changes], axis=1)

for change in puberty_changes.columns:
    for var in change_scores:
        zero = temp[temp[change] == 0.][var].dropna()
        one = temp[temp[change] == 1.][var].dropna()
        two = temp[temp[change] == 2.][var].dropna()

        test = fligner(zero, one, two)
        var_df.at[var, (f'{change}','stat')] = test[0]
        var_df.at[var, (f'{change}','p')] = test[1]

# need to remake this with a third columns for scanner heteroscedasticity.
# and redo the spacing between columns with a gridspec
fig, ax = plt.subplots(ncols=3, sharey=True, figsize=(18,6))

plt.tight_layout(w_pad=1, h_pad=1)
i = sns.stripplot(y=('1yfu-base','stat'),
                  data=var_df[var_df[('1yfu-base', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  ax=ax[0],
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
i.set_xlabel('')
i.set_title('Pubertal change between ages 9-10 and 10-11,')
if len(var_df[var_df[('1yfu-base', 'p')] < 0.000096].index) > 0:
    g = sns.stripplot(y=('1yfu-base','stat'),
                  data=var_df[var_df[('1yfu-base', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      #fill=True,
                      ax=ax[0],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles[3:], labels [3:], 
             bbox_to_anchor=(0., -0.15), 
             loc="center left",
             ncol=3, title='Significant')
else:
    pass

ax[0].set_ylabel('Density')

#k.set_ylabel('')
i.set_ylabel('Greater difference in \nvariability between changes')
i.get_legend().remove()

j = sns.stripplot(y=('2yfu-1yfu','stat'),
                  data=var_df[var_df[('2yfu-1yfu', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  ax=ax[1],
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function'],
              
                 )
if len(var_df[var_df[('2yfu-1yfu', 'p')] < 0.000096].index) > 0:
    h = sns.stripplot(y=('2yfu-1yfu','stat'),
                      data=var_df[var_df[('2yfu-1yfu', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ax=ax[1],
                    #fill=True,
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                   )
    handles, labels = h.get_legend_handles_labels()
    h.legend(handles[3:4], labels [3:4], 
             bbox_to_anchor=(0.5, -0.15), 
             loc="center right",
             ncol=3, title='Significant')
else:
    pass

j.set_title('between ages 10-11 and 11-12,')

j.set_xlabel('')
j.set_ylabel('')
k = sns.stripplot(y=('2yfu-base','stat'),
                  data=var_df[var_df[('2yfu-base', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[2],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
if len(var_df[var_df[('2yfu-base', 'p')] < 0.000096].index) > 0:
    l = sns.stripplot(y=('2yfu-base','stat'),
                      data=var_df[var_df[('2yfu-base', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      #fill=True,
                      ax=ax[2],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                     hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    l.get_legend().remove()
    l.set_ylabel('')
    #l.set_xlabel('')
else: 
    pass


k.set_ylabel('')
k.set_xlabel('')
k.set_title('& between ages 9-10 and 11-12 years.')

handles, labels = k.get_legend_handles_labels()
k.legend(handles[:3], labels [:3], 
         bbox_to_anchor=(1., -0.15), 
         loc="center right",
         ncol=3, title='Not Significant')

sns.despine()
#fig.show()
fig.savefig(f'{PROJ_DIR}/figures/heteroscedasticity_puberty_change.png', dpi=400, bbox_inches="tight")

#meas = ['stat', 'p']
#f_timepoints = [f'f_{timepoint}' for timepoint in timepoints]
#m_timepoints = [f'm_{timepoint}' for timepoint in timepoints]

#all_timepoints = f_timepoints + m_timepoints
#columns = pd.MultiIndex.from_product((all_timepoints, meas))
#s_var_df = pd.DataFrame(columns=columns)

f_df = temp[temp['sex.baseline_year_1_arm_1'] == 'F']
m_df = temp[temp['sex.baseline_year_1_arm_1'] == 'M']

for change in changes:
    for var in change_scores:
        one = m_df[m_df[change] == 1.][var].dropna()
        two = m_df[m_df[change] == 2.][var].dropna()
        three = m_df[m_df[change] == 3.][var].dropna()

        test = fligner(one, two, three)
        var_df.at[var, (f'm_{change}','stat')] = test[0]
        var_df.at[var, (f'm_{change}','p')] = test[1]
        
        one = f_df[f_df[change] == 1.][var].dropna()
        two = f_df[f_df[change] == 2.][var].dropna()
        three = f_df[f_df[change] == 3.][var].dropna()

        test = fligner(one, two, three)
        var_df.at[var, (f'f_{change}','stat')] = test[0]
        var_df.at[var, (f'f_{change}','p')] = test[1]

# need to remake this with a third columns for scanner heteroscedasticity.
# and redo the spacing between columns with a gridspec
fig, ax = plt.subplots(ncols=3, nrows=2, sharey=True, sharex=True, figsize=(18,12))

plt.tight_layout(w_pad=1, h_pad=3)
i = sns.stripplot(y=('f_1yfu-base','stat'),
                  data=var_df[var_df[('f_1yfu-base', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  ##fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[0,0],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
if len(var_df[var_df[('f_1yfu-base', 'p')] < 0.000096].index) > 0:
    g = sns.stripplot(y=('f_1yfu-base','stat'),
                      data=var_df[var_df[('f_1yfu-base', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ##fill=True,
                      ax=ax[0,0],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    
    handles, labels = g.get_legend_handles_labels()
    g.legend(handles[3:], labels [3:], 
             bbox_to_anchor=(0., -1.33), 
             loc="center left",
             ncol=3, title='Significant')
    
else:
    pass
i.set_ylabel('Greater difference in \nvariability between changes')
i.get_legend().remove()
i.set_title('Female participants\nPubertal change between ages 9-10 and 10-11,')

j = sns.stripplot(y=('f_2yfu-1yfu','stat'),
                  data=var_df[var_df[('f_2yfu-1yfu', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  ##fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[0,1],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function'],
              
                 )
if len(var_df[var_df[('f_2yfu-1yfu', 'p')] < 0.000096].index) > 0:
    h = sns.stripplot(y=('f_2yfu-1yfu','stat'),
                      data=var_df[var_df[('f_2yfu-1yfu', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ax=ax[0,1],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                    ##fill=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                   )
else:
    pass

j.set_title('between ages 10-11 and 11-12,')
j.get_legend().remove()
j.set_xlabel('')
j.set_ylabel('')

if len(var_df[var_df[('f_2yfu-base', 'p')] < 0.000096].index) > 0:
    l = sns.stripplot(y=('f_2yfu-base','stat'),
                      data=var_df[var_df[('f_2yfu-base', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ##fill=True,
                      ax=ax[0,2],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                     hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
else:
    pass
k = sns.stripplot(y=('f_2yfu-base','stat'),
                  data=var_df[var_df[('f_2yfu-base', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  ##fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[0,2],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )

k.set_ylabel('')
k.set_xlabel('')
k.set_title('& between ages 9-10 and 11-12 years.')
k.get_legend().remove()
l.set_ylabel('')



i = sns.stripplot(y=('m_1yfu-base','stat'),
                  data=var_df[var_df[('m_1yfu-base', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[1,0],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
i.get_legend().remove()
i.set_ylabel('Greater difference in \nvariability between changes')
i.set_title('Male participants\nPubertal change between ages 9-10 and 10-11')
if len(var_df[var_df[('m_1yfu-base', 'p')] < 0.000096].index) > 0:
    g = sns.stripplot(y=('m_1yfu-base','stat'),
                      data=var_df[var_df[('m_1yfu-base', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      #fill=True,
                      ax=ax[1,0],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    
    g.get_legend().remove()
else:
    pass



j = sns.stripplot(y=('m_2yfu-1yfu','stat'),
                  data=var_df[var_df[('m_2yfu-1yfu', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[1,1],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function'],
              
                 )
if len(var_df[var_df[('m_2yfu-1yfu', 'p')] < 0.000096].index):
    h = sns.stripplot(y=('m_2yfu-1yfu','stat'),
                      data=var_df[var_df[('m_2yfu-1yfu', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      ax=ax[1,1],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                    #fill=True,
                      palette=crayons,
                      hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                   )
else:
    pass

j.set_title('between ages 10-11 and 11-12,')
j.get_legend().remove()
j.set_xlabel('')
j.set_ylabel('')
k = sns.stripplot(y=('m_2yfu-base','stat'),
                  data=var_df[var_df[('m_2yfu-base', 'p')] > 0.000096], 
                  hue=('concept', ''),
                  #fill=True,
                  marker='o',
                  size=6,
                  linewidth=1,
                  edgecolor='white',
                  dodge=True,
                  ax=ax[1,2],
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
if len(var_df[var_df[('m_2yfu-base', 'p')] < 0.000096].index):
    l = sns.stripplot(y=('m_2yfu-base','stat'),
                      data=var_df[var_df[('m_2yfu-base', 'p')] < 0.000096], 
                      hue=('concept', ''),
                      #fill=True,
                      ax=ax[1,2],
                      marker='o',
                      size=8,
                      linewidth=0.5,
                      edgecolor='white',
                      dodge=True,
                      palette=crayons,
                     hue_order=['macrostructure', 
                               'microstructure', 
                               'function']
                     )
    l.get_legend().remove()
    l.set_ylabel('')
else:
    pass

k.set_ylabel('')
k.set_xlabel('')
k.set_title('& between ages 9-10 and 11-12 years.')

handles, labels = k.get_legend_handles_labels()
k.legend(handles[:3], labels [:3], 
         bbox_to_anchor=(1., -0.14), 
         loc="center right",
         ncol=3, title='Not Significant')


#l.set_xlabel('')
sns.despine()
#fig.show()
fig.savefig(f'{PROJ_DIR}/figures/heteroscedasticity_puberty_change_by_sex.png', dpi=400, bbox_inches="tight")


fig = plt.figure(figsize=(25,6))
plt.tight_layout(w_pad=3, h_pad=1)
g = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=var_df[var_df['p'] < 0.000096], 
                  hue='concept',
                  marker='o',
                  size=7,
                  edgecolor='white',
                  dodge=True,
                  linewidth=0.5,
                  ax=ax,
                  palette=crayons,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
k = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=var_df[var_df['p'] > 0.000096], 
                  hue='concept',
                  marker='0',
                  size=7,
                  linewidth=0.5,
                  edgecolor='white',
                  dodge=True,
                  ax=ax,
                  palette=dark,
                  hue_order=['macrostructure', 
                           'microstructure', 
                           'function']
                 )
g.get_legend().remove()
g.set_ylabel('Flinger-Killeen Statistic')
g.set_xlabel('')
g.set_xticklabels(['Age', 'Sex', 'Puberty'])

fig.show()
fig.savefig(f'{PROJ_DIR}/figures/heteroscedasticity_concept.png', dpi=400, bbox_inches="tight")