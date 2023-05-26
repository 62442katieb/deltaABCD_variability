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

variables = [
         "pds_p_ss_female_category_2",
         "pds_p_ss_male_category_2"
         ]
timepoints = [
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

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, 'data_qcd.csv'), 
                 index_col=0, 
                 header=0)

df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
df.drop(list(df.filter(regex='.*_cf12_.*').columns), axis=1, inplace=True)

puberty = pd.read_csv(join(ABCD_DIR, "abcd_ssphp01.csv"), 
                      index_col="subjectkey", 
                      header=0)
temp_df = pd.DataFrame()
for timepoint in timepoints:
    for variable in variables:
        temp_df[f'{variable}.{timepoint}'] = puberty[puberty['eventname'] == timepoint][variable]
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

timepoints.append('baseline_year_1_arm_1')
print(df.filter(regex="pds_p_ss_.*", axis=1).describe())

temp_df = pd.DataFrame()
for timepoint in timepoints:
    print(timepoint, 
          np.mean(df[f'pds_p_ss_female_category_2.{timepoint}'].dropna()), 
          np.mean(df[f'pds_p_ss_male_category_2.{timepoint}'].dropna()))
    temp_df[f'pds_p_ss_category_2.{timepoint}'] = df[f'pds_p_ss_female_category_2.{timepoint}'].fillna(0) + df[f'pds_p_ss_male_category_2.{timepoint}'].fillna(0)
    temp_df[f'pds_p_ss_category_2.{timepoint}'].replace({0: np.nan}, inplace=True)
print(temp_df.describe())
df = pd.concat((df, temp_df), axis=1)
stages = [1., 2., 3.]

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

meas = ['stat', 'p']
columns = pd.MultiIndex.from_product((timepoints, meas))
var_df = pd.DataFrame(columns=columns)

for var in change_scores:
    one = df[df[puberty] == 1.][var].dropna()
    two = df[df[puberty] == 2.][var].dropna()
    three = df[df[puberty] == 3.][var].dropna()

    test = fligner(one, two, three)
    var_df.at[var, 'stat'] = test[0]
    var_df.at[var, 'p'] = test[1]

for i in var_df.index:
    measure = var_df.loc[i]['measure']
    measure = str(measure.values[0])
    if measure in concepts['morph']:
        var_df.at[i,'concept'] = 'macrostructure'
    elif measure in concepts['cell']:
        var_df.at[i,'concept'] = 'microstructure'
    elif measure in concepts['func']:
        var_df.at[i,'concept'] = 'function'

fig = plt.figure(figsize=(25,6))

plt.tight_layout(w_pad=3, h_pad=1)

g = sns.stripplot(x='variable', y='Flinger-Killeen Statistic',
                  data=var_df[var_df['p'] < 7.441394990670425e-05], 
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
                  data=var_df[var_df['p'] > 7.441394990670425e-05], 
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