# %%
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl

from os.path import join
from pingouin import partial_corr
from scipy.stats import fligner, t, spearmanr, pearsonr
from nilearn import plotting#, datasets, surface
from utils import assign_region_names, jili_sidak_mc, plot_surfaces


morph_pal = sns.cubehelix_palette(n_colors=4, start=0.6, rot=-0.6, gamma=1.0, hue=0.9, light=0.6, dark=0.3)
morph_cmap = sns.cubehelix_palette(n_colors=4, start=0.6, rot=-0.6, gamma=1.0, hue=0.9, light=0.3, dark=0.2, 
                                   as_cmap=True, reverse=True)
cell_pal = sns.cubehelix_palette(n_colors=9, start=1.7, rot=-0.8, gamma=1.0, hue=0.9, light=0.6, dark=0.3)
cell_cmap = sns.cubehelix_palette(n_colors=9, start=1.7, rot=-0.8, gamma=1.0, hue=0.9, light=0.3, dark=0.2, 
                                  as_cmap=True, reverse=True)
func_pal = sns.cubehelix_palette(n_colors=4, start=3.0, rot=-0.6, gamma=1.0, hue=0.9, light=0.6, dark=0.3)
func_cmap = sns.cubehelix_palette(n_colors=4, start=3.0, rot=-0.6, gamma=1.0, hue=0.9, light=0.3, dark=0.2, 
                                  as_cmap=True, reverse=True)

age_cmap = sns.cubehelix_palette(n_colors=27, start=0, rot=-0.8, gamma=1.0, hue=0.9, light=0.3, dark=0.2, 
                                  as_cmap=True, reverse=True)
age_pal = sns.cubehelix_palette(n_colors=27, start=5, rot=-0.8, gamma=1.0, hue=0.9, light=0.7, dark=0.3)

big_pal = morph_pal + cell_pal + func_pal
sns.palplot(age_pal)
sns.set(style="white", 
        context="poster", 
        font_scale=0.8,
        rc={"axes.facecolor": (0, 0, 0, 0),
            "font.monospace": 'Courier New',
            "font.family": 'monospace'})
crayons = sns.crayon_palette(['Aquamarine', 'Jungle Green', 'Fuchsia'])


sns.palplot(crayons)


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data_qcd.csv"), index_col=0, header=0)

df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
df.drop(list(df.filter(regex='.*_cf12_.*').columns), axis=1, inplace=True)
no_2yfu = df[df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
df = df.drop(no_2yfu, axis=0)


deltasmri_complete = pd.concat([df.filter(regex='smri.*change_score'), 
                                df.filter(regex='mrisdp.*change_score')], axis=1).dropna()
deltarsfmri_complete = df.filter(regex='rsfmri.*change_score').dropna(how='any')
deltarsi_complete = df.filter(regex='dmri_rsi.*change_score').dropna()
deltadti_complete = df.filter(regex='dmri_dti.*change_score').dropna()

img_modalities = {'smri': deltasmri_complete,
                  'fmri': deltarsfmri_complete,
                  'rsi': deltarsi_complete, 
                  'dti': deltadti_complete}

#morph = img_modalities['smri'].filter(regex='.*vol.*').columns

# plot the distribution of variances of all structural mri measures
smri_var = img_modalities['smri'].columns
dti_var = img_modalities['dti'].columns
rsi_var = img_modalities['rsi'].columns
# separate wm and gm rsi
rsi_gm = list(img_modalities['rsi'].filter(regex='.*gm').columns) + list(img_modalities['rsi'].filter(regex='.*scs').columns)
rsi_wm = list(set(rsi_var) - set(rsi_gm))
rsi_scs = list(img_modalities['rsi'].filter(regex='.*scs').columns)
fmri_var = img_modalities['fmri'].columns
fc_cort_var = img_modalities['fmri'].filter(regex='_c_.*').columns
fc_scor_var = img_modalities['fmri'].filter(regex='_cor_.*').columns
fmri_var_var = img_modalities['fmri'].filter(regex='_var_.*').columns

#morph_var = df[df['concept'] == 'macrostructure'].index
#cell_var = df[df['concept'] == 'microstructure'].index
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


imaging_apd = list(deltasmri_complete.columns) + list(deltadti_complete.columns) + list(deltarsi_complete.columns) + list(deltarsfmri_complete.columns)

alpha, _ = jili_sidak_mc(df[imaging_apd], 0.05)

concepts = {'morph': ['thick', 
                      'area', 
                      'vol',
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
                    'cor',
                    #'subcortical-network fc'
                   ]}


analyses = ['Corr: Age, APC', 'Corr: Baseline, APC', 'PCorr: Age, APC', 'PCorr: Baseline, APC']
metrics = ['r', 'p(r)', 'concept', 'measure']
columns = pd.MultiIndex.from_product((analyses, metrics))
correlations = pd.DataFrame(columns=columns)
for var in imaging_apd:
    base_var = var.split('.')[0]
    temp_df = df[[var, 
                  f'{base_var}.baseline_year_1_arm_1', 
                  'interview_age.baseline_year_1_arm_1']].dropna()
    # correlation between start and change
    r, p = spearmanr(temp_df[var], temp_df[f'{base_var}.baseline_year_1_arm_1'])
    correlations.at[base_var, 
                    ('Corr: Baseline, APC', 'r')] = r
    correlations.at[base_var, 
                    ('Corr: Baseline, APC', 'p(r)')] = p
    # correlation between baseline age and change
    r, p = spearmanr(temp_df[var], temp_df['interview_age.baseline_year_1_arm_1'])
    correlations.at[base_var, 
                  ('Corr: Age, APC', 'r')] = r
    correlations.at[base_var, 
                  ('Corr: Age, APC', 'p(r)')] = p
    # partial correlation between start and change, controlling for age
    pcorr_df = partial_corr(temp_df, 
                        var,  
                        f'{base_var}.baseline_year_1_arm_1',
                        'interview_age.baseline_year_1_arm_1',)
    correlations.at[base_var, 
                  ('PCorr: Baseline, APC', 'r')] = pcorr_df.iloc[0]['r']
    correlations.at[base_var, 
                  ('PCorr: Baseline, APC', 'p(r)')] = pcorr_df.iloc[0]['p-val']
    # partial correlation between age and change, controlling for start
    pcorr_df = partial_corr(temp_df, 
                        var, 
                        'interview_age.baseline_year_1_arm_1', 
                        f'{base_var}.baseline_year_1_arm_1')
    correlations.at[base_var, 
                  ('PCorr: Age, APC', 'r')] = pcorr_df.iloc[0]['r']
    correlations.at[base_var, 
                  ('PCorr: Age, APC', 'p(r)')] = pcorr_df.iloc[0]['p-val']
    if 'mrisdp' in var:
        pass
    elif '_' in var:
        var_list = base_var.split('_')
        correlations.at[base_var, 
                        ('Corr: Age, APC', 'measure')] = var_list[1]
        if var_list[1] in concepts['morph']:
            correlations.at[base_var, 
                            ('Corr: Age, APC', 'concept')] = 'macrostructure'
        elif var_list[1] in concepts['cell']:
            correlations.at[base_var, 
                            ('Corr: Age, APC', 'concept')] = 'microstructure'
        if var_list[1] in concepts['func']:
            correlations.at[base_var, 
                            ('Corr: Age, APC', 'concept')] = 'function'
        if var in btwn_fc:
            correlations.at[base_var, 
                            ('Corr: Age, APC', 'measure')] = 'between-network fc'
        elif var in wthn_fc:
            correlations.at[base_var, 
                            ('Corr: Age, APC', 'measure')] = 'within-network fc'
        elif var in fc_scor_var:
            correlations.at[base_var, 
                            ('Corr: Age, APC', 'measure')] = 'subcortical-network fc'
        elif var in rsi_scs:
            if 'rsirni' in var:
                correlations.at[base_var, 
                                ('Corr: Age, APC', 'measure')] = 'rsirnigm'
            elif 'rsirnd' in var:
                correlations.at[base_var, 
                                ('Corr: Age, APC', 'measure')] = 'rsirndgm'
correlations.dropna(how='all', axis=1, inplace=True)


long_names = {'var': 'BOLD Variance',
              'between-network fc': 'Between FC',
              'within-network fc': 'Within FC',
              'subcortical-network fc': 'Subcortical FC',
              'dtivol': 'WM Volume',
              'vol': 'GM Volume',
              'thick': 'Cortical Thickness',
              'area': 'Cortical Area',
              'dtifa': 'WM FA',
              'dtimd': 'WM MD',
              'dtild': 'WM LD',
              'dtitd': 'WM TD',
              'rsirni': 'WM RNI',
              'rsirnd': 'WM RND',
              'rsirnigm': 'GM RNI',
              'rsirndgm': 'GM RND',
              't1wcnt': 'GM/WM'}

row_order = ['GM Volume', 'Cortical Thickness', 'Cortical Area', 'WM Volume',
             'GM/WM', 'WM FA', 'WM MD', 'WM LD', 'WM TD', 
             'WM RNI', 'GM RNI','WM RND','GM RND',
             'BOLD Variance', 'Between FC', 'Within FC', 'Subcortical FC']


correlations.replace(long_names, inplace=True)
correlations.columns = correlations.columns.to_flat_index()


correlations.rename({('Corr: Age, APC', 'measure'): 'measure',
                     ('Corr: Age, APC', 'concept'): 'concept',
                     ('PCorr: Baseline, APC', 'r'): 'Partial correlation: baseline, APΔ',
                     ('PCorr: Age, APC', 'r'): 'Partial correlation: age, APΔ'}, axis=1, inplace=True)

correlations = assign_region_names(correlations, missing=False)

correlations.to_csv(join(PROJ_DIR, OUTP_DIR,'start_change_region_names.csv'))

g = sns.FacetGrid(correlations, 
                  row="measure", row_order=row_order,
                  hue="measure", hue_order=row_order,
                  aspect=12.5, 
                  height=.6, palette=big_pal, xlim=(-1.1,0.1))

# Draw the densities in a few steps
g.map(sns.kdeplot, ('Corr: Baseline, APC', 'r'),
      bw_adjust=.55, clip_on=False,
      fill=True, alpha=1, linewidth=1.)
g.map(sns.kdeplot, ('Corr: Baseline, APC', 'r'), clip_on=False, color="w", lw=2.5, bw_adjust=.55)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .15, label, #fontweight="bold", 
            color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, ('Corr: Baseline, APC', 'r'))

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-0.5)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.savefig(f'../{FIGS_DIR}/start_change_corr.png', dpi=400)


g = sns.FacetGrid(correlations, 
                  row="measure", row_order=row_order,
                  hue="measure", hue_order=row_order,
                  aspect=10, 
                  height=.8, palette=big_pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, ('Corr: Age, APC', 'r'),
      bw_adjust=.55, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, ('Corr: Age, APC', 'r'), clip_on=False, color="w", lw=2.5, bw_adjust=.55)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .1, label, #fontweight="bold", 
            color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, ('Corr: Age, APC', 'r'))

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-0.4)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.savefig(f'../{FIGS_DIR}/age_change_corr.png', dpi=400)


g = sns.FacetGrid(correlations, 
                  row="measure", row_order=row_order,
                  hue="measure", hue_order=row_order,
                  aspect=10, 
                  height=.8, palette=big_pal, xlim=(-1,0.1))

# Draw the densities in a few steps
g.map(sns.kdeplot, 'Partial correlation: baseline, APΔ',
      bw_adjust=.55, clip_on=False,
      fill=True, alpha=1, linewidth=1.)
g.map(sns.kdeplot, 'Partial correlation: baseline, APΔ', clip_on=False, color="w", lw=2.5, bw_adjust=.55)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .1, label, #fontweight="bold", 
            color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, 'Partial correlation: baseline, APΔ')

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-0.5)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.savefig(f'../{FIGS_DIR}/start_change_minus_age_pcorr.png', dpi=400)


#sns.set_theme(context='talk', style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
g = sns.FacetGrid(correlations, 
                  row="measure", row_order=row_order,
                  hue="measure", hue_order=row_order,
                  aspect=10, 
                  height=.8, palette=big_pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, 'Partial correlation: age, APΔ',
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, 'Partial correlation: age, APΔ', clip_on=False, color="w", lw=2.5, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, #fontweight="bold", 
            color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, 'Partial correlation: age, APΔ')

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-0.3)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.savefig(f'../{FIGS_DIR}/age_change_minus_start_pcorr.png', dpi=400)

temp_df = correlations[correlations['measure'] == 'FC (sc)']
prop_sig = np.sum(temp_df[('Corr: Age, APC', 'p(r)')] < alpha) / len(temp_df.index)
temp_df.describe(percentiles=[.25,.75]).loc['75%']['Partial correlation: baseline, APΔ']

stats = ['mean', '(Q1,Q3)', '%sig']
meas = ['Pearson: age, APΔ', 'Pearson: baseline, APΔ', 'Partial: age, APΔ', 'Partial: baseline, APΔ']
columns = pd.MultiIndex.from_product([meas, stats])
corr_stats = pd.DataFrame(columns=columns)
for measure in correlations['measure'].unique():
    temp_df = correlations[correlations['measure'] == measure]
    descriptives = temp_df.describe()
    
    # I want mean correlation, quartiles, and proportion of regions that are significantly correlated
    corr_stats.at[measure, ('Pearson: age, APΔ', 'mean')] = descriptives.loc['mean'][('Corr: Age, APC', 'r')]
    corr_stats.at[measure, ('Pearson: baseline, APΔ', 'mean')] = descriptives.loc['mean'][('Corr: Baseline, APC', 'r')]
    corr_stats.at[measure, ('Partial: age, APΔ', 'mean')] = descriptives.loc['mean']['Partial correlation: age, APΔ']
    corr_stats.at[measure, ('Partial: baseline, APΔ', 'mean')] = descriptives.loc['mean']['Partial correlation: baseline, APΔ']
    
    # quartiles are already calculated in `descriptives`
    q1 = descriptives.loc['25%']
    q3 = descriptives.loc['75%']
    # need to round these (:)
    corr_stats.at[measure, ('Pearson: age, APΔ', '(Q1,Q3)')] = (np.round(q1[('Corr: Age, APC', 'r')],2), np.round(q3[('Corr: Age, APC', 'r')],2))
    corr_stats.at[measure, ('Pearson: baseline, APΔ', '(Q1,Q3)')] = (np.round(q1[('Corr: Baseline, APC', 'r')],2), np.round(q3[('Corr: Baseline, APC', 'r')],2))
    corr_stats.at[measure, ('Partial: age, APΔ', '(Q1,Q3)')] = (np.round(q1['Partial correlation: age, APΔ'],2), np.round(q3['Partial correlation: age, APΔ'],))
    corr_stats.at[measure, ('Partial: baseline, APΔ', '(Q1,Q3)')] = (np.round(q1[('Partial correlation: baseline, APΔ')],2), np.round(q3[('Partial correlation: baseline, APΔ')],2))
    
    # proportion of regions with significant (p)corrs
    prop_sig = np.sum(temp_df[('Corr: Age, APC', 'p(r)')] < alpha) / len(temp_df.index)
    corr_stats.at[measure, ('Pearson: age, APΔ', '%sig')] = prop_sig
    
    prop_sig = np.sum(temp_df[('Corr: Baseline, APC', 'p(r)')] < alpha) / len(temp_df.index)
    corr_stats.at[measure, ('Pearson: baseline, APΔ', '%sig')] = prop_sig
    
    prop_sig = np.sum(temp_df['Partial correlation: age, APΔ'] < alpha) / len(temp_df.index)
    corr_stats.at[measure, ('Partial: age, APΔ', '%sig')] = prop_sig
    
    prop_sig = np.sum(temp_df['Partial correlation: baseline, APΔ'] < alpha) / len(temp_df.index)
    corr_stats.at[measure, ('Partial: baseline, APΔ', '%sig')] = prop_sig
    

corr_stats = corr_stats.loc[row_order]
corr_stats.to_csv(join(PROJ_DIR, OUTP_DIR, 'APC_startchange_summary.csv'))
