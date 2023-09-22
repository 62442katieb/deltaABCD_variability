#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl

from os.path import join
from matplotlib.gridspec import GridSpec

from scipy.stats import fligner, mannwhitneyu
from nilearn import plotting, datasets, surface

from utils import jili_sidak_mc, plot_surfaces, assign_region_names

sns.set(style='white', context='talk')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'

crayons = sns.crayon_palette(['Aquamarine', 'Fuchsia', 
                              'Jungle Green', 'Fern'])

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))

df.drop(list(df.filter(regex='lesion.*', axis=1).columns), axis=1, inplace=True)
df.drop(list(df.filter(regex='.*_cf12_.*', axis=1).columns), axis=1, inplace=True)
df.drop(df.filter(regex='.*cortgordon.*', axis=1).columns, axis=1, inplace=True)
no_2yfu = df[df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
df = df.drop(no_2yfu, axis=0)

deltasmri_complete = pd.concat([df.filter(regex='smri.*change_score'), 
                                df.filter(regex='mrisdp.*change_score')], axis=1).dropna()
deltarsfmri_complete = df.filter(regex='rsfmri.*change_score').dropna(how='any')
deltarsi_complete = df.filter(regex='dmri_rsi.*change_score').dropna()
deltadti_complete = df.filter(regex='dmri_dti.*change_score').dropna()

df[['pds_p_ss_female_category_2.baseline_year_1_arm_1',
    'pds_p_ss_male_category_2.baseline_year_1_arm_1']]
df['pds_p_ss_category_2.baseline_year_1_arm_1'] = df['pds_p_ss_female_category_2.baseline_year_1_arm_1'].fillna(0) + df['pds_p_ss_male_category_2.baseline_year_1_arm_1'].fillna(0)

tests = ['variance', 
         'fligner_sex',
         'fligner_puberty',
         #'fligner_puberty_4',
         #'fligner_puberty_5',
         'fligner_raceth', 
         'fligner_income', 
         'fligner_edu', 
         'fligner_marital', 
         'fligner_age', 
         'fligner_scanner'
        ]

var_df = pd.read_pickle(join(PROJ_DIR, 
                          OUTP_DIR, 
                          'variance_flinger-alpha<0.0.pkl'))
var_df.drop(var_df.filter(regex='.*cortgordon.*', axis=1).columns, axis=1, inplace=True)


img_modalities = {'smri': deltasmri_complete,
                  'fmri': deltarsfmri_complete,
                  'rsi': deltarsi_complete, 
                  'dti': deltadti_complete}


# plot the distribution of variances of all structural mri measures
smri_var = img_modalities['smri'].columns
dti_var = img_modalities['dti'].columns
rsi_var = img_modalities['rsi'].columns
fmri_var = img_modalities['fmri'].columns
fmri_cor_var = img_modalities['fmri'].filter(regex='_c.*').columns
fmri_var_var = img_modalities['fmri'].filter(regex='_var_.*').columns
morph = img_modalities['smri'].filter(regex='.*vol.*').columns

fc_cort_var = img_modalities['fmri'].filter(regex='_c_.*').columns
fc_scor_var = img_modalities['fmri'].filter(regex='_cor_.*').columns
fmri_var_var = img_modalities['fmri'].filter(regex='_var_.*').columns

#morph_var = df[df['concept'] == 'macrostructure'].index
#cell_var = df[df['concept'] == 'microstructure'].index
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

puberty = 'pds_p_ss_category_2.baseline_year_1_arm_1'
race = 'race_ethnicity.baseline_year_1_arm_1'
income = 'demo_comb_income_v2.baseline_year_1_arm_1'
edu = 'demo_prnt_ed_v2.baseline_year_1_arm_1'
age = 'interview_age.baseline_year_1_arm_1'
sex = 'sex.baseline_year_1_arm_1'
mri = 'mri_info_manufacturer.baseline_year_1_arm_1'
marry = "demo_prnt_marital_v2.baseline_year_1_arm_1"

# ## Visualizing brain heterogeneity across non-brain variables
# 1. Variability across all brain measures
# 2. Per modality
# 3. Across the brain
# 4. Across developmental variables
# 5. Across demographic variables

#print(var_df.index)

devt = ['fligner_age', 
        'fligner_sex',
        'fligner_puberty']
demo =  ['fligner_raceth',
         'fligner_income',
         'fligner_edu',
         'fligner_marital', 
         'fligner_scanner'
        ]

demo_alphas = [f'{i}_alpha' for i in demo]
devt_alphas = [f'{i}_alpha' for i in devt]

# build a df that categorizes measures (i.e., 'smri', 'cortical volume', 'region')
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
        #elif var_num == 604:
            #var_df.at[var, 'measure'] = 'gmvol'
    elif '_' in var:
        var_list = var.split('.')[0].split('_')
        var_df.at[var, 'modality'] = var_list[0]
        var_df.at[var, 'atlas'] = var_list[2]
        var_df.at[var, 'region'] = '_'.join(var_list[3:])
        if 'scs' in var and 'rsi' in var:
            var_df.at[var, 'measure'] = f'{var_list[1]}gm'
        else:
            var_df.at[var, 'measure'] = var_list[1]

#print(var_df['measure'].unique())

#print(var_df['atlas'].unique())

var_df = var_df[var_df['measure'] != 't1w']
var_df = var_df[var_df['measure'] != 't2w']
var_df = var_df[var_df['atlas'] != 'cortgordon']

atlases = list(np.unique(list(var_df['atlas'])))
measures = list(np.unique(list(var_df['measure'])))

modalities = list(np.unique(list(var_df['modality'])))

concepts = {'morph': ['thick', 
                      'area', 
                      'vol',
                      'dtivol'],
            'cell': ['t1wcnt',
                     'rsirnigm',
                     'rsirndgm', 
                     'dtifa', 
                     'dtimd'
                    'dtitd', 
                     'dtild',
                     'rsirni',
                     'rsirnd'
                     ],
            'func':['var',
                    'c',
                     'cor']}

row_order = ['cdk',
             #'cortgordon',
             'fib',
             'fiberat',
             'ngd',
             'scs',
             'morph', 
             'vol',
             'thick', 
             'area', 
             'dtivol',
             'cell', 
             't1wcnt',
             'rsirnigm',
             'rsirndgm',
             'dtifa',
             'dtimd',
             'dtitd',
             'dtild',
             'rsirni', 
             'rsirnd', 
             'func', 
             'var',
             'within-network fc',
             'between-network fc', 
             'cor']

row_order2 = ['all', 
             'macro', 
             'vol',
             'thick', 
             'area', 
             'dtivol',
             'micro', 
             't1wcnt',
             'rsirnigm',
             'rsirndgm',
             'dtifa',
             'dtimd',
             'dtitd',
             'dtild',
             'rsirni',
             'rsirnd',
             'func', 
             'var',
             'within-network fc',
             'between-network fc', 
             'subcortical-network fc']

columns = atlases + measures + modalities + list(concepts.keys())
prop_heterosked = pd.DataFrame(index=tests[1:], 
                                          columns=columns)
for category in tests[1:]:
    #print(category)
    for atlas in atlases:
        temp_df = var_df[var_df['atlas'] == atlas]
        prop_heterosked.at[category, atlas] = sum(temp_df[category]['sig'] == '**') / len(temp_df.index)
    for measure in measures:
        if measure == 'c':
            temp_df = var_df.loc[wthn_fc]
            prop_heterosked.at[category, 'within-network fc'] = sum(temp_df[category]['sig'] == '**') / len(temp_df.index)
            
            temp_df = var_df.loc[btwn_fc]
            prop_heterosked.at[category, 'between-network fc'] = sum(temp_df[category]['sig'] == '**') / len(temp_df.index)
        else:
            temp_df = var_df[var_df['measure'] == measure]
            prop_heterosked.at[category, measure] = sum(temp_df[category]['sig'] == '**') / len(temp_df.index)
    for modality in modalities:
        temp_df = var_df[var_df['modality'] == modality]
        prop_heterosked.at[category, modality] = sum(temp_df[category]['sig'] == '**') / len(temp_df.index)
    for concept in concepts.keys():
        con_df = var_df[var_df['measure'] == concepts[concept][0]]
        for measure in concepts[concept][1:]:
            temp_df = var_df[var_df['measure'] == measure]
            con_df = pd.concat([con_df, temp_df], axis=0)
        prop = sum(con_df[category]['sig'] == '**') / len(con_df.index)
        prop_heterosked.at[category, concept] = prop
prop_heterosked = prop_heterosked.T
prop_heterosked.columns = [i[8:] for i in prop_heterosked.columns]
prop_heterosked = prop_heterosked.loc[row_order]
prop_heterosked.dropna(axis=1,how='all').to_csv(join(PROJ_DIR, OUTP_DIR,'heteroscedasticity_atlas_measures.csv'))

prop_heterosked.to_csv(join(PROJ_DIR, OUTP_DIR, 'proportion_heteroscedastic_brain_regions.csv'))

all_var = demo + devt
prop_hsk_demo = pd.Series()
for var in all_var:
    v = var.split('_')[1]
    #print(var, sum(var_df[demo_var]['sig'] == '**') / len(var_df.index))
    prop_hsk_demo.at[v] = sum(var_df[var]['sig'] == '**') / len(var_df.index)
prop_hsk_demo.to_csv(join(PROJ_DIR, OUTP_DIR, 'proportion_heteroscedastic_devtdemo.csv'))

macro_var = []
micro_var = []

for i in var_df.index:
    #print(i)
    measure = var_df.loc[i]['measure']
    #print(measure)
    measure = str(measure.values[0])
    #print(measure)
    if measure in concepts['morph']:
        var_df.at[i,'concept'] = 'macrostructure'
        macro_var.append(i)
    elif measure in concepts['cell']:
        var_df.at[i,'concept'] = 'microstructure'
        micro_var.append(i)
    elif measure in concepts['func']:
        var_df.at[i,'concept'] = 'function'
    #elif measure in concepts['conn']:
    #    var_df.at[i,'concept'] = 'functional connectivity'

#print(var_df['measure'].unique())

var_description = var_df[['modality', 
                          'atlas', 
                          'measure', 
                          'region', 
                          'concept']]
var_description.columns = var_description.columns.get_level_values(0)

hsk_sig_demo = var_description
for var in demo:
    temp_df = pd.Series(var_df[(var,'stat')], name=var)
    hsk_sig_demo = pd.concat([hsk_sig_demo, temp_df], axis=1)


hsk_sig_devt = var_description
for var in devt:
    temp_df = pd.Series(var_df[(var,'stat')], name=var)
    hsk_sig_devt = pd.concat([hsk_sig_devt, temp_df], axis=1)

func_var = list(btwn_fc) + list(wthn_fc) + list(fc_scor_var) + list(fmri_var_var) 

all_vars = func_var + macro_var + micro_var
# # Parsing variance across heteroscedastic categories
# 

for i in var_df.index:
    if i in btwn_fc:
        var_df.at[i, 'measure'] = 'between-network fc'
    elif i in wthn_fc:
        var_df.at[i, 'measure'] = 'within-network fc'
    elif i in fc_scor_var:
        var_df.at[i, 'measure'] = 'subcortical-network fc'

concepts = var_df['concept'].dropna().unique()

# added plots of brain variance by measure 
#(i.e., across participants, point per region)
# per pubertal stage - per reviewer request

long_names = {'thick': 'Cortical Thickness', 
                      'area': 'Cortical Area', 
                      'vol': 'Gray Matter (GM) Volume',
                      'gmvol': 'Gray Matter (GM) Volume',
                      'dtivol': 'White Matter (WM) Volume',
            't1wcnt': 'GM/WM Contrast', 
                     'rsirni': 'Isotropic Intracellular Diffusion (WM)', 
                     'rsirnd': 'Directional Intracellular Diffusion (WM)',
                     'rsirnigm': 'Isotropic Intracellular Diffusion (GM)', 
                     'rsirndgm': 'Directional Intracellular Diffusion (GM)',
                     'dtifa': 'Fractional Anisotropy (WM)', 
                     'dtimd': 'Mean Diffusivity (WM)',
                    'dtitd': 'Transverse Diffusivity (WM)', 
                     'dtild': 'Longitudinal Diffusivity (WM)',
            'var': 'BOLD Variance',
                    #'c': 'Network Connectivity',
                    #'cor': 'Subcortical-to-Network Connectivity',
              'within-network fc': 'Cortical Network FC',
              'between-network fc': 'Cortical Network FC',
             'subcortical-network fc': 'Subcortical-to-Network FC'}

new_measure_order = [
    'Cortical Thickness',
    'Cortical Area', 
    'Gray Matter (GM) Volume',
    'White Matter (WM) Volume',
    'GM/WM Contrast', 
    'Isotropic Intracellular Diffusion (WM)', 
    'Directional Intracellular Diffusion (WM)',
    'Isotropic Intracellular Diffusion (GM)',
    'Directional Intracellular Diffusion (GM)',
    'Fractional Anisotropy (WM)', 
    'Mean Diffusivity (WM)',
    'Transverse Diffusivity (WM)', 
    'Longitudinal Diffusivity (WM)',
    'BOLD Variance',
    'Cortical Network FC',
    'Subcortical-to-Network FC'
    ]

hetero = {
    'fligner_age':{
                    'var': age,
                    'levels': [(107.,112.), (113.,119.), (120.,125.), (126.,133.)],
                    'strings': ['9-9.3', '9.4-9.9', '10-10.4', '10.5-11']},
               'fligner_sex':{
                   'var': sex,
                   'levels': ['F', 'M'],
                   'strings': ['female', 'male']},
               'fligner_puberty': {
                   'var': puberty, 
                   'levels': [1., 2., 3.],
                   'strings': ['pre', 'early', 'mid']},
                #'fligner_income':{
                #    'var': income,
                #    'levels': [(0,6), (7,8), (9,10)],
                #    'strings': ['<$75k', '$75k-100k', '>$100k']},
            #   'fligner_scanner':{
            #       'var': mri,
            #       'levels': ['SIEMENS', 
            #               'GE MEDICAL SYSTEMS', 
            #               'Philips Medical Systems'],
            #       'strings': ['Siemens', 'GE', 'Philips']},
               #'fligner_edu': {
               #    'var': edu, 
               #    'levels': [(0,14), (15,17), 18, (19,22)],
               #    'strings': ['HS/GED', 'AA/Some', 'Bach', 'Grad']}, 
               #'fligner_raceth': {
               #    'var': race,
               #    'levels': [1,2,3,(4,5)], 
               #    'strings': ['White', 'Black', 'Hispanic', 'Asian/Oth.']},
               #'fligner_marital': {
               #    'var': marry, 
               #    'levels': [1,(2,5)], 
               #    'strings': ['Married', 'Not Married']}, 
               }
#print(var_df['measure'].unique())
#print('duplicate indices:', sum(var_df.index.duplicated()))

for fligner_var in list(hetero.keys())[:3]:
    var_name = hetero[fligner_var]['var']
    subsamples = {}
    strings = hetero[fligner_var]['strings']
    sig_strings = [f"{string}*" for string in strings]
    levels = hetero[fligner_var]['levels']
    cols = strings + sig_strings + ['measure', 'modality']
    variance = pd.DataFrame(index=var_df.index,
                            columns=cols)
    for i in range(0, len(hetero[fligner_var]['levels'])):
        level_val = levels[i]
        level_name = strings[i]
        #print(level_name, level_val, type(level_val))
        if type(level_val) == int:
            subsamples[level_name] = df[df[var_name] == level_val].index
        elif type(level_val) == float:
            subsamples[level_name] = df[df[var_name] == level_val].index
        elif type(level_val) == tuple:
            subsamples[level_name] = df[df[var_name].between(level_val[0], level_val[1], inclusive='both')].index
        elif type(level_val) == str:
            subsamples[level_name] = df[df[var_name] == level_val].index

    sig_hsk = var_df[var_df[(fligner_var,'sig')] == '**'].index
    for region in var_df.index:
        #variance.at[region, 'measure'] = 
        variance.at[region, 'modality'] = var_df.loc[region]['modality'][0]
        measure = var_df.loc[region]['measure'][0]
        #print(measure)
        variance.at[region, 'measure'] = measure
        variance.at[region, 'long_measure'] = long_names[measure]
        for string in strings:
            ppts = subsamples[string]
            if region in sig_hsk:
                variance.at[region, f'{string}*'] = np.var(df.loc[ppts][region])
            else:
                variance.at[region, string] = np.var(df.loc[ppts][region])
    variance.to_csv(join(PROJ_DIR, OUTP_DIR, f'variance_by_level-{var_name}.csv'))
    new_df = pd.DataFrame(index=var_df['measure'].unique(), columns=strings)
    for measure in var_df['measure'].unique():
        for string in strings:
            if variance[variance['measure'] == measure].iloc[0][string] > 0:
                new_df.at[measure, string] = np.mean(variance[variance['measure'] == measure][string])
            else:
                new_df.at[measure, string] = np.mean(variance[variance['measure'] == measure][f'{string}*'])
    new_df.to_csv(join(PROJ_DIR, OUTP_DIR, f'variance_by_level-{var_name}_measures.csv'))
    colors = sns.husl_palette(n_colors=len(levels), h=0.01, s=0.8, l=0.65, as_cmap=False)
    darks = sns.husl_palette(n_colors=len(levels), h=0.01, s=0.9, l=0.40, as_cmap=False)
    color_list = colors.as_hex()
    dark_list = darks.as_hex()

    #measure_list = list(variance['measure'].dropna().unique())
    #print(variance.describe())
    fig,ax = plt.subplots(nrows=4, ncols=4, sharex=False, sharey=True,  figsize=(29,20))
    #fig,ax = plt.subplots(ncols=3, sharex=False, sharey=True,  figsize=(21,7))
    plt.tight_layout(h_pad=2)
    new_measure_list = list(variance['long_measure'].unique())
    
    for measure in new_measure_list:
        if measure == np.nan:
            pass
        else:
            i = new_measure_order.index(measure)
            #print(measure,i)
            if i in [0,1,2,3]:
                #axis = ax[0,i]
                axis = ax[0,i]
                #print(0, i)
            elif i in [4,5,6,7]:
                axis = ax[1,i-4]
                #print(1, i-4)
            elif i in [8,9,10,11]:
                axis = ax[2,i-8]
                #print(2,i-8)
            elif i in [12,13, 14, 15]:
                axis = ax[3, i-12]
                #print(3, i-12)
            melt = pd.melt(variance[variance['long_measure'] == measure], value_vars=strings)
            #print('measure\n', melt.describe())
            g = sns.stripplot(data=melt, x='value', y='variable',hue='variable',  ax=axis, dodge=False, alpha=0.5, palette=colors)
            g.set_xlabel('Variance')
            if i in [0,4,8,12]:
                g.set_ylabel(f'{fligner_var.split("_")[-1]}'.capitalize())
            g.set_title(measure)
            is_legend = g.get_legend()
            if is_legend is not None:
                is_legend.remove()
            #print(sig_strings, np.mean(variance[variance['measure'] == measure][sig_strings[0]]))
            if np.sum(np.sum(variance[variance['long_measure'] == measure][sig_strings])) > 0:
                melt_sig = pd.melt(variance[variance['long_measure'] == measure], value_vars=sig_strings)
                #print('measure\n', melt_sig.describe())
                h = sns.pointplot(data=melt_sig, x='value', y='variable', ax=axis, join=False,  
                                markers='X', hue='variable', dodge=.1, errorbar=None, palette=darks)
                h = sns.stripplot(data=melt_sig, x='value', y='variable', hue='variable',  ax=axis, dodge=False, alpha=0.75, palette=darks)
                #means = {}
                for string in sig_strings:
                    j = sig_strings.index(string)
                    mean = np.mean(variance[variance['long_measure'] == measure][string])
                    axis.axvline(mean, lw=2, ls='--', color=dark_list[j], alpha=0.75)
                h.get_legend().remove()
                h.set_xlabel('Variance')
            else:
                pass
            
            
    variance.to_csv(join(PROJ_DIR, 
                     OUTP_DIR, 
                     f'variance_by_level-{fligner_var.split("_")[-1]}.csv'))         
    fig.savefig(join(PROJ_DIR, 
                     FIGS_DIR, 
                     f'{fligner_var.split("_")[-1]}_variance.png'), 
                dpi=500, 
                bbox_inches='tight')      
print('\n\nVARIANCE PLOTS ARE DONE\n\n')

for fligner_var in hetero.keys():
    #print('\n\n',fligner_var)
    variable = fligner_var.split('_')[-1]
    var = hetero[fligner_var]['var']
    levels = hetero[fligner_var]['levels']
    strings = hetero[fligner_var]['strings']
    #print(levels, strings)
    
    sig_measures = var_df[var_df[(fligner_var, 'sig')] == '**'].index
    nsig_measures = var_df[var_df[(fligner_var, 'sig')] != '**'].index
    #top_50 = var_df[(fligner_var, 'stat')].sort_values()[-50:].index
    #highest_heterosced = var_description.loc[top_50].describe()
    #bot_50 = var_df[(fligner_var, 'stat')].sort_values()[:50].index
    #lowest_heterosced = var_description.loc[bot_50].describe()
    #print('HIGHEST-----\n', highest_heterosced)
    #print('LOWEST-----\n', lowest_heterosced)
    
    if type(levels[0]) == int or type(levels[0]) == str or type(levels[0]) == float:
        fligner_df = df[df[var] == levels[0]]
    elif type(levels[0]) == tuple:
        it = levels[0]
        fligner_df = df[df[var].between(it[0], it[1])]
    fligner_df = pd.Series(fligner_df[var_df.index].var(), name=str(strings[0]))
    #print(levels[0])
    

    for i in range(0, len(levels[1:])):
        level = levels[i+1]
        string = strings[i+1]
        #print(string)
        if type(level) == int or type(level) == str or type(level) == float:
            temp_df = df[df[var] == level]
            #print(len(temp_df.index))
        elif type(level) == tuple:
            temp_df = df[df[var].between(level[0], level[1])]
            #print(len(temp_df.index))
        #print(level)
        temp_df = pd.Series(temp_df[var_df.index].var(), name=string)
        fligner_df = pd.concat([temp_df, fligner_df], axis=1)
    print('in fligner_df but not significant', 
          len(set(fligner_df.index) - set(sig_measures)), 
          '\nsignificant but not in fligner df', 
          len(set(sig_measures) - set(fligner_df.index)))
    temp = fligner_df.loc[sig_measures]
    #fligner_df = fligner_df.dropna(thresh=len(levels) - 1)
    #top_50_df = fligner_df.loc[top_50]
    #not_present = list(set(levels) - set(fligner_df.columns))
    mann_whitney_u = pd.DataFrame()
    #temp = fligner_df.loc[sig_measures]
    for string in strings:
        dat = fligner_df[string].dropna()
        #print(dat.index, '\n\n\n\n')
        for string1 in strings:
            dat1 = fligner_df[string1].dropna()
            if string != string1:
                if len(dat.index) > 0 and len(dat1.index) > 0:
                    column = f'{string} > {string1}'
                    res = mannwhitneyu(dat, dat1, alternative='greater')
                    #mann_whitney_u.at[f'{string} > {string1}', ('all', 'stat')] = res.statistic
                    mann_whitney_u.at['all', column] = res.pvalue

                    # run mann-whitney for each biological concept (macro/microstructure & function)
                    res = mannwhitneyu(dat.loc[func_var], dat1.loc[func_var], alternative='greater')
                    #mann_whitney_u.at[f'{string} > {string1}', ('func', 'stat')] = res.statistic
                    mann_whitney_u.at['func', column] = res.pvalue

                    res = mannwhitneyu(dat.loc[macro_var], dat1.loc[macro_var], alternative='greater')
                    #mann_whitney_u.at[f'{string} > {string1}', ('macro', 'stat')] = res.statistic
                    mann_whitney_u.at['macro', column] = res.pvalue
                    
                    res = mannwhitneyu(dat.loc[micro_var], dat1.loc[micro_var], alternative='greater')
                    #mann_whitney_u.at[f'{string} > {string1}', ('micro', 'stat')] = res.statistic
                    mann_whitney_u.at['micro', column] = res.pvalue
                    
                    for measure in var_df['measure'].unique():
                        print(measure)
                        variables = var_df[var_df['measure'] == measure].index
                        sig_variables = list(set(variables) - set(nsig_measures))
                        if len(sig_variables) == 0:
                            pass
                        else:
                            res = mannwhitneyu(dat.loc[sig_variables], dat1.loc[sig_variables], alternative='greater')
                            mann_whitney_u.at[measure, column] = res.pvalue
                            print(res.pvalue)
                else:
                    mann_whitney_u.at['all', column] = np.nan
                    mann_whitney_u.at['all', column] = np.nan
            else:
                pass
    #mann_whitney_u = mann_whitney_u.loc[row_order2]
    mann_whitney_u.to_csv(join(PROJ_DIR, OUTP_DIR,f'mann_whitney-{variable}-variance_diff.csv'))
    #convert from variance to coefficient of variation (sdev / mean)
    heteroskedasticity = pd.Series(var_df[(fligner_var, 'stat')], 
                                   name='heteroscedasticity')
    fligner_cov = np.sqrt(fligner_df) / fligner_df.mean(axis=0)
    fligner_cov = pd.concat([fligner_cov, var_description], axis=1)
    fligner_df = pd.concat([fligner_df, 
                            heteroskedasticity, 
                            var_description], axis=1)
    fligner_df.to_csv(join(PROJ_DIR, OUTP_DIR,f'heteroscedasticity_{variable}.csv'))
    
    str_levels = [str(level) for level in levels]
    long_fligner = fligner_df.loc[sig_measures].melt(value_vars=strings, 
                                    value_name='variance',
                                    var_name=variable,
                                    id_vars=var_description.columns)
    
    fligner_df = None
    
    n_colors = len(np.unique(long_fligner[variable]))
    morph_pal = sns.cubehelix_palette(n_colors=n_colors, start=0.6, rot=-0.6, 
                                      gamma=1.0, hue=0.7, light=0.6, dark=0.4)
    cell_pal = sns.cubehelix_palette(n_colors=n_colors, start=1.7, rot=-0.8, 
                                     gamma=1.0, hue=0.7, light=0.6, dark=0.4)
    func_pal = sns.cubehelix_palette(n_colors=n_colors, start=3.0, rot=-0.6, 
                                     gamma=1.0, hue=0.7, light=0.6, dark=0.4)
    
sns.set(style='white', context='paper', font_scale=2.)
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'    

# let's visualize this variability!
destrieux = datasets.fetch_atlas_destrieux_2009()
desikan = datasets.fetch_neurovault_ids(image_ids=(23262, ))
subcort = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')

scs_1 = ['x',
         'x',
         'x',
         'x',
        "tplh",
        "caudatelh",
        "putamenlh",
        "pallidumlh",
        "bstem",
        "hpuslh",
        "amygdalalh",
        "aal",
        'x',
        'x',
         'x',
        "tprh",
        "caudaterh",
        "putamenrh",
        "pallidumrh",
        "hpusrh",
        "amygdalarh",
        "aar"]

# looks like I'm not using this code anywhere else?
abcd_to_harvox = pd.DataFrame(columns=['label', 'Volume', 
                                       'T1w Intensity', 
                                       'Restricted Normalized Directional Diffusion', 
                                       'Restricted Normalized Isotropic Diffusion'])

for i in range(0, len(subcort.labels)):
    label = subcort.labels[i]
    abcd_to_harvox.at[i, 'label'] = label
    if scs_1[i] == 'x':
        abcd_to_harvox.at[i, 'Volume'] = np.nan
        abcd_to_harvox.at[i, 'T1w Intensity'] = np.nan
        abcd_to_harvox.at[i, 'Restricted Normalized Directional Diffusion'] = np.nan
        abcd_to_harvox.at[i, 'Restricted Normalized Isotropic Diffusion'] = np.nan
    else:
        abcd_to_harvox.at[i, 'Volume'] = f'smri_vol_scs_{scs_1[i]}'
        abcd_to_harvox.at[i, 'T1w Intensity'] = f'smri_t1w_scs_{scs_1[i]}'
        abcd_to_harvox.at[i, 'Restricted Normalized Directional Diffusion'] = f'dmri_rsirnd_scs_{scs_1[i]}'
        abcd_to_harvox.at[i, 'Restricted Normalized Isotropic Diffusion'] = f'dmri_rsirni_scs_{scs_1[i]}'

morph_pal = sns.cubehelix_palette(start=0.6, rot=-0.6, gamma=1.0, hue=1, light=0.7, dark=0.3)
morph_cmap = sns.cubehelix_palette(n_colors=4, start=0.6, rot=-0.6, gamma=1.0, hue=1, light=0.9, dark=0.4, 
                                   as_cmap=True, reverse=True)
cell_pal = sns.cubehelix_palette(start=1.7, rot=-0.8, gamma=1.0, hue=1, light=0.7, dark=0.3)
cell_cmap = sns.cubehelix_palette(n_colors=7, start=1.7, rot=-0.8, gamma=1.0, hue=1, light=0.9, dark=0.4, 
                                  as_cmap=True, reverse=True)
func_pal = sns.cubehelix_palette(start=3.0, rot=-0.6, gamma=1.0, hue=1, light=0.7, dark=0.3)
func_cmap = sns.cubehelix_palette(n_colors=4, start=3.0, rot=-0.6, gamma=1.0, hue=1, light=0.8, dark=0.4, 
                                  as_cmap=True, reverse=True)
big_pal = morph_pal + cell_pal + func_pal
#sns.palplot(big_pal)
sns.set(style='white', context='paper', font_scale=2.)
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'    

fsaverage = datasets.fetch_surf_fsaverage()

nifti_mapping = pd.read_csv(join(PROJ_DIR, 
                                 DATA_DIR, 
                                 'variable_to_nifti_mapping.csv'), 
                            header=0, 
                            index_col=0)

vol_mapping = {'smri_vol_cdk_banksstslh.change_score': 1001.0,
    'smri_vol_cdk_cdacatelh.change_score': 1002.0,
    'smri_vol_cdk_cdmdfrlh.change_score': 1003.0,
    'smri_vol_cdk_cuneuslh.change_score': 1005.0,
    'smri_vol_cdk_ehinallh.change_score': 1006.0,
    'smri_vol_cdk_fusiformlh.change_score': 1007.0,
    'smri_vol_cdk_ifpllh.change_score': 1008.0,
    'smri_vol_cdk_iftmlh.change_score': 1009.0,
    'smri_vol_cdk_ihcatelh.change_score': 1010.0,
    'smri_vol_cdk_locclh.change_score': 1011.0,
    'smri_vol_cdk_lobfrlh.change_score': 1012.0,
    'smri_vol_cdk_linguallh.change_score': 1013.0,
    'smri_vol_cdk_mobfrlh.change_score': 1014.0,
    'smri_vol_cdk_mdtmlh.change_score': 1015.0,
    'smri_vol_cdk_parahpallh.change_score': 1016.0,
    'smri_vol_cdk_paracnlh.change_score': 1017.0,
    'smri_vol_cdk_parsopclh.change_score': 1018.0,
    'smri_vol_cdk_parsobislh.change_score': 1019.0,
    'smri_vol_cdk_parstgrislh.change_score': 1020.0,
    'smri_vol_cdk_pericclh.change_score': 1021.0,
    'smri_vol_cdk_postcnlh.change_score': 1022.0,
    'smri_vol_cdk_ptcatelh.change_score': 1023.0,
    'smri_vol_cdk_precnlh.change_score': 1024.0,
    'smri_vol_cdk_pclh.change_score': 1025.0,
    'smri_vol_cdk_rracatelh.change_score': 1026.0,
    'smri_vol_cdk_rrmdfrlh.change_score': 1027.0,
    'smri_vol_cdk_sufrlh.change_score': 1028.0,
    'smri_vol_cdk_supllh.change_score': 1029.0,
    'smri_vol_cdk_sutmlh.change_score': 1030.0,
    'smri_vol_cdk_smlh.change_score': 1031.0,
    'smri_vol_cdk_frpolelh.change_score': 1032.0,
    'smri_vol_cdk_tmpolelh.change_score': 1033.0,
    'smri_vol_cdk_trvtmlh.change_score': 1034.0,
    'smri_vol_cdk_insulalh.change_score': 1035.0,
    'smri_vol_cdk_banksstsrh.change_score': 2001.0,
    'smri_vol_cdk_cdacaterh.change_score': 2002.0,
    'smri_vol_cdk_cdmdfrrh.change_score': 2003.0,
    'smri_vol_cdk_cuneusrh.change_score': 2005.0,
    'smri_vol_cdk_ehinalrh.change_score': 2006.0,
    'smri_vol_cdk_fusiformrh.change_score': 2007.0,
    'smri_vol_cdk_ifplrh.change_score': 2008.0,
    'smri_vol_cdk_iftmrh.change_score': 2009.0,
    'smri_vol_cdk_ihcaterh.change_score': 2010.0,
    'smri_vol_cdk_loccrh.change_score': 2011.0,
    'smri_vol_cdk_lobfrrh.change_score': 2012.0,
    'smri_vol_cdk_lingualrh.change_score': 2013.0,
    'smri_vol_cdk_mobfrrh.change_score': 2014.0,
    'smri_vol_cdk_mdtmrh.change_score': 2015.0,
    'smri_vol_cdk_parahpalrh.change_score': 2016.0,
    'smri_vol_cdk_paracnrh.change_score': 2017.0,
    'smri_vol_cdk_parsopcrh.change_score': 2018.0,
    'smri_vol_cdk_parsobisrh.change_score': 2019.0,
    'smri_vol_cdk_parstgrisrh.change_score': 2020.0,
    'smri_vol_cdk_periccrh.change_score': 2021.0,
    'smri_vol_cdk_postcnrh.change_score': 2022.0,
    'smri_vol_cdk_ptcaterh.change_score': 2023.0,
    'smri_vol_cdk_precnrh.change_score': 2024.0,
    'smri_vol_cdk_pcrh.change_score': 2025.0,
    'smri_vol_cdk_rracaterh.change_score': 2026.0,
    'smri_vol_cdk_rrmdfrrh.change_score': 2027.0,
    'smri_vol_cdk_sufrrh.change_score': 2028.0,
    'smri_vol_cdk_suplrh.change_score': 2029.0,
    'smri_vol_cdk_sutmrh.change_score': 2030.0,
    'smri_vol_cdk_smrh.change_score': 2031.0,
    'smri_vol_cdk_frpolerh.change_score': 2032.0,
    'smri_vol_cdk_tmpolerh.change_score': 2033.0,
    'smri_vol_cdk_trvtmrh.change_score': 2034.0,
    'smri_vol_cdk_insularh.change_score': 2035.0,}

for vol_var in vol_mapping.keys():
    var = vol_var.split('.')[0]
    nifti_mapping.at[var,'atlas_value'] = vol_mapping[vol_var]

# list of measures to plot
measures = {'cortical-thickness': 'smri_thick_cdk_*',
            'cortical-gwcontrast': 'smri_t1wcnt_cdk_*',
            'cortical-area': 'smri_area_cdk_.*',
            'cortical-volume': 'smri_vol_cdk_.*', 
            'subcortical-volume': 'smri_vol_scs_.*', 
            'subcortical-RND': 'dmri_rsirnd_scs_.*',
            'subcortical-RNI': 'dmri_rsirni_scs_.*',
            'cortical-RND': 'dmri_rsirndgm_.*',
            'cortical-RNI': 'dmri_rsirnigm_.*',
            'cortical-BOLD-variance': 'rsfmri_var_cdk_.*',
            'tract-volume': 'dmri_dtivol_fiberat_.*', 
            'tract-FA': 'dmri_dtifa_fiberat_.*', 
            'tract-MD': 'dmri_dtimd_fiberat_.*',
            'tract-LD': 'dmri_dtild_fiberat_.*', 
            'tract-TD': 'dmri_dtitd_fiberat_.*', 
            'tract-RND': 'dmri_rsirnd_fib_.*',
            'tract-RNI': 'dmri_rsirni_fib_.*'}
tract_measures = {'tract-volume': 'dmri_dtivol_fiberat_.*', 
            'tract-FA': 'dmri_dtifa_fiberat_.*', 
            'tract-MD': 'dmri_dtimd_fiberat_.*',
            'tract-LD': 'dmri_dtild_fiberat_.*', 
            'tract-TD': 'dmri_dtitd_fiberat_.*', 
            'tract-RND': 'dmri_rsirnd_fib_.*',
            'tract-RNI': 'dmri_rsirni_fib_.*'}
vmax_other = 30
#vmax_scanner = 800
#cmap = 'viridis'

conn_measures = {'cortical-network-connectivity': 'rsfmri_c_ngd_.*',
            'subcortical-network-connectivity': 'rsfmri_cor_ngd_.*_scs_.*',}

morph_cmap = sns.diverging_palette(250, 256.3, s=90, l=60, center="dark", as_cmap=True)
func_cmap = sns.diverging_palette(250, 140.9, s=90, l=60, center="dark", as_cmap=True)
cell_cmap = sns.diverging_palette(250, 294.3, s=90, l=60, center="dark", as_cmap=True)
#cell_pal = sns.diverging_palette(250, 294.3, s=80, l=55, center="dark", as_cmap=False)
sns.set(style='white', context='paper', font_scale=2.)
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'   

pals = {'cortical-thickness': morph_cmap,
        'cortical-gwcontrast': cell_cmap,
            'cortical-area': morph_cmap,
            'cortical-volume': morph_cmap, 
            'subcortical-volume': morph_cmap, 
            'subcortical-RND': cell_cmap,
            'subcortical-RNI': cell_cmap,
            'cortical-RND': cell_cmap,
            'cortical-RNI': cell_cmap,
            'cortical-BOLD-variance': func_cmap,
            'tract-volume': morph_cmap, 
            'tract-FA': cell_cmap, 
            'tract-MD': cell_cmap, 
            'tract-TD': cell_cmap, 
            'tract-LD': cell_cmap, 
            'tract-RND': cell_cmap,
            'tract-RNI': cell_cmap,
        'cortical-network-connectivity': func_cmap,
            'subcortical-network-connectivity': func_cmap}


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

# test run plotting wm tracts not on a t1 (until I can get them properly registered)

corrs = var_df.filter(regex='rsfmri_c_ngd.*', axis=0).index
corrs = [i.split('.')[0] for i in corrs]
networks = list(np.unique([i.split('_')[-1] for i in corrs]))

corrs = var_df.filter(regex='rsfmri_c_ngd.*', axis=0).index
corrs = [i.split('.')[0] for i in corrs]
networks = list(np.unique([i.split('_')[-1] for i in corrs]))

btwn_fc_src = [i.split('.')[0].split('_')[3] for i in btwn_fc]
btwn_fc_trgt = [i.split('.')[0].split('_')[-1] for i in btwn_fc]

# okay, now we're plotting between and within network connectivity
for fligner_var in list(hetero.keys()):
    fligner = fligner_var.split('_')[-1]
    print(fligner)
    vmax = vmax_other
    #within-network fc is easy to plot bc there's only one HSK value per network (per fligner_var)
    meas_df = var_df.loc[wthn_fc]
    meas_vars = [i.split('.')[0] for i in meas_df.index]
    atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
    #print(atlas_fname)
    atlas_nii = nib.load(atlas_fname)
    atlas_arr = atlas_nii.get_fdata()
    plotting_arr = np.zeros(atlas_arr.shape)
    nv_arr = np.zeros(atlas_arr.shape)
    sig = 0
    for i in meas_df.index:
        j = i.split('.')[0]
        value = nifti_mapping.loc[j]['atlas_value']
        #print(i, value)
        if value is np.nan:
            pass
        else:
            nv_arr[np.where(atlas_arr == value)] = - np.log10(var_df.at[i,(fligner_var, 'p')])
            
            if var_df.at[i,(fligner_var, 'sig')] == '**':
                sig += 1
                plotting_arr[np.where(atlas_arr == value)] = var_df.at[i,(fligner_var, 'stat')]
            else:
                plotting_arr[np.where(atlas_arr == value)] = 0
    nv_nimg = nib.Nifti1Image(nv_arr, atlas_nii.affine).to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_nlogp_FCw.nii')
    print('\t\tplotting...', f'{sig} out of {len(meas_df.index)} heteroskedastic regions')
    meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
    meas_nimg.to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_FCw.nii')
    figure = plot_surfaces(meas_nimg, fsaverage, func_cmap, vmax, 9)
    figure.savefig(f'{PROJ_DIR}/figures/{fligner}_FCw_fk.png', dpi=600)
    plt.close()
    
    # between-network FC is tough bc we have to average all of a networks HSK values
    # but only the significantly HSK connections
    
    meas_df = var_df.loc[btwn_fc]
    meas_df.loc[btwn_fc, 'from_ntwk'] = btwn_fc_src
    meas_df.loc[btwn_fc, 'to_ntwk'] = btwn_fc_trgt
    avgs = pd.DataFrame()
    nlogps = pd.DataFrame()
    # for each network
    
    for ntwk in np.unique(btwn_fc_src):
        sig = []
        # grab only FC values including that network
        temp_df = meas_df[meas_df['from_ntwk'] == ntwk]
        temp_df2 = meas_df[meas_df['to_ntwk'] == ntwk]
        temp_df = pd.concat([temp_df, temp_df2], axis=0)
        # grab nlog p for all of that network's connections
        # calculate average heteroscedasticity of all 
        # significantly heteroscedastic network connections
        atlas_fname = nifti_mapping.loc[temp_df.index[0].split('.')[0]]['atlas_fname']
        atlas_nii = nib.load(atlas_fname)
        atlas_arr = atlas_nii.get_fdata()
        nv_arr = np.zeros(atlas_arr.shape)
        ntwk_arr = np.zeros(atlas_arr.shape)
        for i in temp_df.index:
            #
            target_ntwk = i.split('.')[0]
            value = nifti_mapping.loc[target_ntwk]['atlas_value']
            nv_arr[np.where(atlas_arr == value)] = - np.log10(temp_df.loc[i, (fligner_var, 'p')])
            if temp_df.loc[i, (fligner_var, 'sig')] == '**':
                sig.append(temp_df.loc[i,(fligner_var, 'stat')])
                ntwk_arr[np.where(atlas_arr == value)] = temp_df.loc[i,(fligner_var, 'stat')]
            else:
                pass
        nv_nimg = nib.Nifti1Image(nv_arr, atlas_nii.affine)
        ntwk_nimg = nib.Nifti1Image(ntwk_arr, atlas_nii.affine)
        nv_nimg.to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_{ntwk}FCb-nlogp.nii')
        figure = plot_surfaces(nv_nimg, fsaverage, func_cmap, vmax, 3)
        figure.savefig(f'{PROJ_DIR}/figures/{fligner}x{ntwk}FC_fk.png', dpi=600)
        mean_hsk = np.mean(sig)
        # grab name of first conn var for this network for plotting
        avgs.at[temp_df.index[0], 'fk'] = mean_hsk
    meas_vars = [i.split('.')[0] for i in avgs.index]
    atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
    atlas_nii = nib.load(atlas_fname)
    atlas_arr = atlas_nii.get_fdata()
    #print(atlas_fname)
    plotting_arr = np.zeros(atlas_arr.shape)
    sig = 0
    for i in avgs.index:
        j = i.split('.')[0]
        value = nifti_mapping.loc[j]['atlas_value']
        #print(i, value)
        if value is np.nan:
            pass
        else:
            
            plotting_arr[np.where(atlas_arr == value)] = avgs.at[i,'fk']        
    meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
    meas_nimg.to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_FCb.nii')
    figure = plot_surfaces(meas_nimg, fsaverage, func_cmap, vmax, 9)
    figure.savefig(f'{PROJ_DIR}/figures/{fligner}xFCb_fk.png', dpi=600)
    plt.close()


fc_scor_var
scs_varnames = [i.split('.')[0].split('_')[-1] for i in fc_scor_var]


for fligner_var in list(hetero.keys()):
    fligner = fligner_var.split('_')[-1]
    #if fligner == 'scanner':
    #    vmax = vmax_scanner
    #else:
    #    vmax = vmax_other
    
    meas_df = var_df.loc[fc_scor_var]
    
    meas_df.loc[fc_scor_var, 'scs'] = scs_varnames
    avgs = pd.DataFrame()
    nsig = 0
    for scs in np.unique(scs_varnames):
        sig = []
        temp_df = meas_df[meas_df['scs'] == scs]
        # calculate average heteroscedasticity of all 
        # significantly heteroscedastic network connections
        
        for i in temp_df.index:
            if temp_df.loc[i, (fligner_var, 'sig')] == '**':
                sig.append(temp_df.loc[i,(fligner_var, 'stat')])
                nsig += 1
            else:
                pass
        mean_hsk = np.mean(sig)
        #print(mean_hsk)
        # grab name of first conn var for this network for plotting
        avgs.at[temp_df.index[0], 'fk'] = mean_hsk
    #print(nsig)
    meas_vars = [i.split('.')[0] for i in avgs.index]
    atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
    #print(atlas_fname)
    atlas_nii = nib.load(atlas_fname)
    atlas_arr = atlas_nii.get_fdata()
    plotting_arr = np.zeros(atlas_arr.shape)
    sig = 0
    for i in avgs.index:
        j = i.split('.')[0]
        value = nifti_mapping.loc[j]['atlas_value']
        #print(i, value)
        if value is np.nan:
            pass
        else:
            plotting_arr[np.where(atlas_arr == value)] = avgs.at[i,'fk']        
    meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
    meas_nimg.to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_FCscs.nii')
    fig,ax = plt.subplots(figsize=(4,8)
                         )
    plt.figure(layout='tight')
    q = plotting.plot_stat_map(meas_nimg, display_mode='z',  threshold=9,
                           #cut_coords=[-20, -10, 0, 10], 
                           cut_coords=4,
                           vmax=vmax*1.1, 
                           annotate=True, cmap=func_cmap, colorbar=False,
                           symmetric_cbar=False, axes=ax)
    #q.add_edges(meas_nimg)
    #ax[1].set_visible(False)
    fig.savefig(f'{PROJ_DIR}/figures/{fligner}xFCs_fk_scs.png', dpi=600, bbox_inches='tight')
    plt.close(fig)

vmaxes = [vmax_other]
for vmax in vmaxes:
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

    cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
                                cmap=func_cmap, 
                                values=range(-int(vmax*1.1),int(vmax*1.1)), 
                                )
    ax.set_xlabel('Heteroscedasticity (F-K Statistic)')

    plt.savefig(f'{PROJ_DIR}/figures/func-cmap_1-{vmax}.png', bbox_inches='tight', dpi=600)

    cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
                                cmap=cell_cmap, 
                                values=range(-int(vmax*1.1),int(vmax*1.1)), 
                                )
    ax.set_xlabel('Heteroscedasticity (F-K Statistic)')

    plt.savefig(f'{PROJ_DIR}/figures/cell-cmap_1-{vmax}.png', bbox_inches='tight', dpi=600)
    cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
                                cmap=morph_cmap, 
                                values=range(-int(vmax*1.1),int(vmax*1.1)), 
                                )
    ax.set_xlabel('Heteroscedasticity (F-K Statistic)')

    plt.savefig(f'{PROJ_DIR}/figures/morph-cmap_1-{vmax}.png', bbox_inches='tight', dpi=600)
    plt.close()

sns.set(style='white', context='poster')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'   

for fligner_var in list(hetero.keys()):
    fligner = fligner_var.split('_')[-1]
    print(fligner)
    #if fligner == 'scanner':
    #    vmax = vmax_scanner
    #else:
    #    vmax = vmax_other
    vmax = vmax_other
    for measure in measures:
        print(f'\t{measure}')
        meas_df = var_df.filter(regex=measures[measure], axis=0)
        if 'tract' in measure:
            fibers = nifti_mapping.filter(regex=tract_measures[measure], axis=0).index
            var = fibers[0]
            tract_fname = nifti_mapping.loc[var]['atlas_fname']
            tract_nii = nib.load(tract_fname)
            tract_arr = tract_nii.get_fdata()
            #print(np.unique(tract_arr))
            if var_df.at[f'{var}.change_score',(fligner_var, 'sig')] == '**':
                tract_arr *= var_df.at[f'{var}.change_score',(fligner_var, 'stat')]
            else:
                tract_arr *= 0
            all_tracts_arr = np.zeros(tract_arr.shape)
            nv_arr = np.zeros(tract_arr.shape)
            all_tracts_arr += tract_arr
            for var in fibers[1:]:    
                tract_fname = nifti_mapping.loc[var]['atlas_fname']
                if type(tract_fname) is str:
                    try:
                        tract_nii = nib.load(tract_fname)
                        tract_arr = tract_nii.get_fdata()
                        #print(np.unique(tract_arr))
                        nv_arr += tract_arr * (np.log10(var_df.at[i,(fligner_var, 'p')]) * -1)
                        if var_df.at[f'{var}.change_score',(fligner_var, 'sig')] == '**':
                            tract_arr *= var_df.at[f'{var}.change_score',(fligner_var, 'stat')]
                        else:
                            tract_arr *= 0
                        all_tracts_arr += tract_arr
                    except Exception as e:
                        pass
                else:
                    pass
            meas_nimg = nib.Nifti1Image(all_tracts_arr, tract_nii.affine)
            nv_nimg = nib.Nifti1Image(nv_arr, atlas_nii.affine).to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_nlogp_{measure}.nii')
            meas_nimg.to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_{measure}.nii')
            fig2,ax2 = plt.subplots(figsize=(4,2)
                                    )
            plt.figure(layout='tight')
            q = plotting.plot_stat_map(meas_nimg, display_mode='z',  threshold=1,
                                    #cut_coords=[-20, 0, 18, 40], 
                                    cut_coords=4,
                                    vmax=vmax*1.1, 
                                    annotate=True, cmap=pals[measure], colorbar=False,
                                    symmetric_cbar=False, axes=ax2)
            #q.add_edges(meas_nimg)
            fig2.savefig(f'{PROJ_DIR}/figures/{fligner}x{measure}_fk.png', dpi=600, bbox_inches='tight')
            plt.close(fig2)
        else:
            #print(len(meas_df.index))
            meas_vars = [i.split('.')[0] for i in meas_df.index]
            atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
            #print(atlas_fname)
            atlas_nii = nib.load(atlas_fname)
            atlas_arr = atlas_nii.get_fdata()
            plotting_arr = np.zeros(atlas_arr.shape)
            nv_arr = np.zeros(atlas_arr.shape)
            sig = 0
            for i in meas_df.index:
                if 'cdk_total' in i:
                    pass
                else:
                    j = i.split('.')[0]
                    value = nifti_mapping.loc[j]['atlas_value']
                    #print(i, value)
                    if value is np.nan:
                        pass
                    else:
                        nv_arr[np.where(atlas_arr == value)] = - np.log10(var_df.at[i,(fligner_var, 'p')])
                        if var_df.at[i,(fligner_var, 'sig')] == '**':
                            sig += 1
                            plotting_arr[np.where(atlas_arr == value)] = var_df.at[i,(fligner_var, 'stat')]
                        else:
                            plotting_arr[np.where(atlas_arr == value)] = 0
            nv_nimg = nib.Nifti1Image(nv_arr, atlas_nii.affine).to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_nlogp_{measure}.nii')
            print('\t\tplotting...', f'{sig} out of {len(meas_df.index)} heteroskedastic regions\n\t\tavg val: {np.mean(plotting_arr)}')
            meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
            meas_nimg.to_filename(f'{PROJ_DIR}/{OUTP_DIR}/{fligner}_{measure}.nii')
            if 'subcortical' in measure:
                grid_kw = dict(width_ratios=[3,1])
                
                fig,ax = plt.subplots(#ncols=2, gridspec_kw=grid_kw, figsize=(24,4)
                                    )
                plt.figure(layout='tight')
                v = plotting.plot_stat_map(meas_nimg, display_mode='z',  threshold=1,
                                    #cut_coords=[-20, -10, 0, 10], 
                                    cut_coords=4,
                                    vmax=vmax*1.1, 
                                    annotate=True, cmap=pals[measure], colorbar=False,
                                    symmetric_cbar=False, axes=ax)

                #ax[1].set_visible(False)
                fig.savefig(f'{PROJ_DIR}/figures/{fligner}x{measure}_fk_scs.png', dpi=600, bbox_inches='tight')
                plt.close(fig)
            
            elif 'cortical' in measure:
                figure = plot_surfaces(meas_nimg, fsaverage, pals[measure], vmax, 1)
                #texture_l = surface.vol_to_surf(meas_nimg, fsaverage.pial_left, interpolation='nearest')
                #texture_r = surface.vol_to_surf(meas_nimg, fsaverage.pial_right, interpolation='nearest')

                #figure = plotting.plot_surf_stat_map(fsaverage.pial_left, texture_l, symmetric_cbar=False, threshold=1,
                                                    #cmap=pals[measure], view='medial', colorbar=False, vmax=vmax)
                #plt.tight_layout(pad=2)
                #figure.savefig(f'../figures/{measure}x{fligner}_fk_leftmed.png', dpi=600)
                #figure = plotting.plot_surf_stat_map(fsaverage.pial_left, texture_l, symmetric_cbar=False, threshold=1,
                                                    #cmap=pals[measure], view='lateral', colorbar=False, vmax=vmax)
                #plt.tight_layout(pad=2)
                #figure.savefig(f'../figures/{measure}x{fligner}_fk_leftlat.png', dpi=600)
                #figure = plotting.plot_surf_stat_map(fsaverage.pial_right, texture_r, symmetric_cbar=False, threshold=1,
                                                    #cmap=pals[measure], view='medial', colorbar=False, vmax=vmax)
                #plt.tight_layout(pad=2)
                #figure.savefig(f'../figures/{measure}x{fligner}_fk_rightlat.png', dpi=600)
                #figure = plotting.plot_surf_stat_map(fsaverage.pial_right, texture_r, symmetric_cbar=False, threshold=1,
                                                    #cmap=pals[measure], view='lateral', colorbar=False, vmax=vmax)
                #plt.tight_layout(pad=2)
                figure.savefig(f'{PROJ_DIR}/figures/{fligner}x{measure}_fk.png', dpi=600)
                plt.close()
                

