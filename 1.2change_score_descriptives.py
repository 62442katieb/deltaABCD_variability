import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib.gridspec import GridSpec

from os.path import join

from scipy.ndimage import binary_erosion
from nilearn import plotting, datasets, surface
from utils import plot_surfaces, assign_region_names

# set up colormaps & palettes for plotting later
morph_pal = sns.cubehelix_palette(n_colors=4, start=0.6, rot=-0.6, gamma=1.0, hue=0.7, light=0.6, dark=0.3)
morph_cmap = sns.cubehelix_palette(n_colors=4, start=0.6, rot=-0.6, gamma=1.0, hue=0.7, light=0.3, dark=0.2, 
                                   as_cmap=True, reverse=True)
cell_pal = sns.cubehelix_palette(n_colors=9, start=1.7, rot=-0.8, gamma=1.0, hue=0.7, light=0.6, dark=0.3)
cell_cmap = sns.cubehelix_palette(n_colors=9, start=1.7, rot=-0.8, gamma=1.0, hue=0.7, light=0.3, dark=0.2, 
                                  as_cmap=True, reverse=True)
func_pal = sns.cubehelix_palette(n_colors=4, start=3.0, rot=-0.6, gamma=1.0, hue=0.7, light=0.6, dark=0.3)
func_cmap = sns.cubehelix_palette(n_colors=4, start=3.0, rot=-0.6, gamma=1.0, hue=0.7, light=0.3, dark=0.2, 
                                  as_cmap=True, reverse=True)
big_pal = morph_pal + cell_pal + func_pal
morph_cell_pal = morph_pal + cell_pal


sns.set(style="white", 
        context="talk", 
        font_scale=0.8,
        rc={"axes.facecolor": (0, 0, 0, 0),
            "font.monospace": 'Courier New',
            "font.family": 'monospace'})
crayons = sns.crayon_palette(['Aquamarine', 'Burnt Sienna', 'Jungle Green', 'Fuchsia', 'Lavender'])

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

# need to calculate mean & sd for each imaging variable change score
descriptives = pd.DataFrame(columns=['annualized percent change', 'sdev', 'concept', 'atlas', 'measure'])
for var in imaging_apd:
    descriptives.at[var,'annualized percent change'] = df[var].mean()
    descriptives.at[var,'sdev'] = df[var].std()
    if 'mrisdp' in var:
        var_num = int(var.split('.')[0].split('_')[-1])
        descriptives.at[var, 'atlas'] = 'dtx'
        if var_num <= 148:
            descriptives.at[var, 'concept'] = 'macrostructure'
            descriptives.at[var, 'measure'] = 'thick'
        elif var_num <= 450 and var_num >= 303:
            descriptives.at[var, 'concept'] = 'macrostructure'
            descriptives.at[var, 'measure'] = 'area'
        elif var_num < 604 and var_num >= 450:
            descriptives.at[var, 'concept'] = 'macrostructure'
            descriptives.at[var, 'measure'] = 'vol'
        elif var_num <= 1054 and var_num >= 907:
            descriptives.at[var, 'concept'] = 'cellular architecture'
            descriptives.at[var, 'measure'] = 't1wcnt'
        elif var_num == 604:
            descriptives.at[var, 'concept'] = 'macrostructure'
            descriptives.at[var, 'measure'] = 'vol'
    elif '_' in var:
        var_list = var.split('.')[0].split('_')
        descriptives.at[var, 'measure'] = var_list[1]
        descriptives.at[var, 'atlas'] = var_list[2]
        if var_list[1] in concepts['morph']:
            descriptives.at[var, 'concept'] = 'macrostructure'
        elif var_list[1] in concepts['cell']:
            descriptives.at[var, 'concept'] = 'cellular architecture'
        if var_list[1] in concepts['func']:
            descriptives.at[var, 'concept'] = 'function'
        if var in btwn_fc:
            descriptives.at[var, 'measure'] = 'between-network fc'
        elif var in wthn_fc:
            descriptives.at[var, 'measure'] = 'within-network fc'
        elif var in fc_scor_var:
            descriptives.at[var, 'measure'] = 'subcortical-network fc'
        elif var in rsi_scs:
            if 'rsirni' in var:
                descriptives.at[var, 'measure'] = 'rsirnigm'
            elif 'rsirnd' in var:
                descriptives.at[var, 'measure'] = 'rsirndgm'

# Calculate mean and variance APdelta for each measure, 
# separately for each level of each categorical variable
# add into supplement with significance stars for differences in _variance_


# remove all non-brain or non-desikan-killany variables
drop = ['dtx', 'meanmotion',
       'subthreshnvols', 'subtcignvols', 'ntpoints']
drop_var = []
for var in drop:
    drop_var += list(descriptives[descriptives['atlas'] == var].index)

descriptives.drop(list(drop_var), axis=0, inplace=True)

measures = list(descriptives['measure'].unique())
concepts = ['macrostructure', 'microstructure', 'function']
long_names = {'var': 'BOLD',
              'between-network fc': 'FC (btwn)',
              'within-network fc': 'FC (wthn)',
              'subcortical-network fc': 'FC (sc)',
              'dtivol': 'WMV',
              'vol': 'GMV',
              'thick': 'CT',
              'area': 'CA',
              'dtifa': 'FA',
              'dtimd': 'MD',
              'dtild': 'LD',
              'dtitd': 'TD',
              'rsirni': 'RNI',
              'rsirnd': 'RND',
              'rsirnigm': 'RNI (gm)',
              'rsirndgm': 'RND (gm)',
              't1wcnt': 'G/W'}

sub_df = descriptives[descriptives['concept'] == 'macrostructure']
sub2_df = descriptives[descriptives['concept'] == 'cellular architecture']
sub_df = pd.concat([sub_df, sub2_df])
sub_df.replace(long_names, inplace=True)

# apc macro- and microstructure ridge plot
g = sns.FacetGrid(sub_df, 
                  row="measure", row_order=['GMV', 'CT', 'CA', 'WMV', 
                                            'G/W', 'FA','MD', 'LD', 'TD',
                                            'RNI', 'RNI (gm)','RND','RND (gm)'],
                  hue="measure", hue_order=['GMV', 'CT', 'CA', 'WMV', 
                                            'G/W', 'FA','MD', 'LD', 'TD',
                                            'RNI', 'RNI (gm)','RND','RND (gm)'],
                  aspect=15, height=.5, palette=morph_cell_pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "annualized percent change",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "annualized percent change", clip_on=False, color="w", lw=2.5, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, #fontweight="bold", 
            color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "annualized percent change")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.savefig(f'../{FIGS_DIR}/apr_morpcell.png', dpi=400)

sub_df = descriptives[descriptives['concept'] == 'function']
sub_df2 = descriptives[descriptives['concept'] == 'function']
sub_df.replace(long_names, inplace=True)

# apc function ridge plot (separate bc scale is much larger)
g = sns.FacetGrid(sub_df, row="measure", hue="measure", aspect=16, height=.5, palette=func_pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "annualized percent change",
      bw_adjust=.5, clip_on=False,
      fill=True, alpha=1, linewidth=1.5)
g.map(sns.kdeplot, "annualized percent change", clip_on=False, color="w", lw=2, bw_adjust=.5)

# passing color=None to refline() uses the hue mapping
g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, color=color,
            ha="left", va="center", transform=ax.transAxes)


g.map(label, "annualized percent change")

# Set the subplots to overlap
g.figure.subplots_adjust(hspace=-.25)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[], ylabel="")
g.despine(bottom=True, left=True)
g.savefig(f'../{FIGS_DIR}/apr_function.png', dpi=400)

# calculate descriptives
desc_summ = pd.DataFrame(index=measures, columns=['mean', 'sdev', '(Q1, Q3)', '(min, max)'])
for measure in measures:
    temp_df = descriptives[descriptives['measure'] == measure]
    mean = np.mean(temp_df['annualized percent change'])
    desc_summ.at[measure, 'mean'] = np.round(mean,2)
    sdev = np.mean(temp_df['sdev'])
    desc_summ.at[measure, 'sdev'] = np.round(sdev,2)
    desc_summ.at[measure, '(Q1, Q3)'] = (np.round(np.quantile(temp_df['annualized percent change'], 0.25),2), 
                                         np.round(np.quantile(temp_df['annualized percent change'], 0.75),2))
    desc_summ.at[measure, '(min, max)'] = (np.round(np.min(temp_df['annualized percent change']),2), 
                                         np.round(np.max(temp_df['annualized percent change']),2))
desc_summ.to_csv(join(PROJ_DIR, OUTP_DIR, 'apchange_descriptives.csv'))

descriptives2 = assign_region_names(descriptives)
descriptives2.to_csv(join(PROJ_DIR, OUTP_DIR, 'apchange_descriptives_regions.csv'))

# read in mapping from tabular data to nifti parcellations
nifti_mapping = pd.read_csv(join(PROJ_DIR, 
                                 DATA_DIR, 
                                 'variable_to_nifti_mapping.csv'), 
                            header=0, 
                            index_col=0)

# list of measures to plot
measures = {'tract-volume': 'dmri_dtivol_fiberat_.*', 
            'tract-FA': 'dmri_dtifa_fiberat_.*', 
            'tract-MD': 'dmri_dtimd_fiberat_.*',
            'tract-LD': 'dmri_dtild_fiberat_.*', 
            'tract-TD': 'dmri_dtitd_fiberat_.*', 
            'tract-RND': 'dmri_rsirnd_fib_.*',
            'tract-RNI': 'dmri_rsirni_fib_.*',
            'cortical-thickness': 'smri_thick_cdk_*',
            'cortical-gwcontrast': 'smri_t1wcnt_cdk_*',
            'cortical-area': 'smri_area_cdk_.*',
            'cortical-volume': 'smri_vol_cdk_.*', 
            'subcortical-volume': 'smri_vol_scs_.*', 
            'subcortical-RND': 'dmri_rsirnd_scs_.*',
            'subcortical-RNI': 'dmri_rsirni_scs_.*',
            'cortical-RND': 'dmri_rsirndgm_.*',
            'cortical-RNI': 'dmri_rsirnigm_.*',
            'cortical-BOLD-variance': 'rsfmri_var_cdk_.*',
            }
tract_measures = {'tract-volume': 'dmri_dtivol_fiberat_.*', 
            'tract-FA': 'dmri_dtifa_fiberat_.*', 
            'tract-MD': 'dmri_dtimd_fiberat_.*',
            'tract-LD': 'dmri_dtild_fiberat_.*', 
            'tract-TD': 'dmri_dtitd_fiberat_.*', 
            'tract-RND': 'dmri_rsirnd_fib_.*',
            'tract-RNI': 'dmri_rsirni_fib_.*'}


conn_measures = {'cortical-network-connectivity': 'rsfmri_c_ngd_.*',
            'subcortical-network-connectivity': 'rsfmri_cor_ngd_.*_scs_.*',}

fsaverage = datasets.fetch_surf_fsaverage()

morph_cmap = sns.diverging_palette(12, 256.3, s=70, l=50, center="light", as_cmap=True)
func_cmap = sns.diverging_palette(343, 140.9, s=70, l=50, center="light", as_cmap=True)
cell_cmap = sns.diverging_palette(31, 294.3, s=70, l=50, center="light", as_cmap=True)
morph_pal = sns.diverging_palette(12, 256.3, s=70, l=50, center="light", as_cmap=False)
func_pal = sns.diverging_palette(343, 140.9, s=70, l=50, center="light", as_cmap=False)
cell_pal = sns.diverging_palette(31.0, 294.3, s=70, l=50, center="light", as_cmap=False)
sns.palplot(morph_pal + func_pal + cell_pal)

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
            'tract-LD': cell_cmap, 
            'tract-TD': cell_cmap,
            'tract-RND': cell_cmap,
            'tract-RNI': cell_cmap,
        'cortical-network-connectivity': func_cmap,
            'subcortical-network-connectivity': func_cmap}

# let's plot APC on brains pls
for measure in measures.keys():
    #print(measure, measures[measure])
    if 'BOLD' in measure:
        vmax = 3.5
    else: 
        vmax = 1.5
    #print(measure)
    meas_df = descriptives.filter(regex=measures[measure], axis=0)
    meas_vars = [i.split('.')[0] for i in meas_df.index]
    if 'tract' in measure:
        fibers = nifti_mapping.filter(regex=measures[measure], axis=0).index
        var = fibers[0]
        #print(var)
        tract_fname = nifti_mapping.loc[var]['atlas_fname']
        tract_nii = nib.load(tract_fname)
        tract_arr = tract_nii.get_fdata()
        tract_arr = binary_erosion(tract_arr, iterations=10).astype(tract_arr.dtype)
        #print(np.unique(tract_arr))
        tract_arr *= meas_df.at[f'{var}.change_score', 'annualized percent change']
        all_tracts_arr = np.zeros(tract_arr.shape)
        all_tracts_arr += tract_arr
        for fiber in fibers[1:]:    
            tract_fname = nifti_mapping.loc[fiber]['atlas_fname']
            if type(tract_fname) is str:
                try:
                    tract_nii = nib.load(tract_fname)
                    tract_arr = tract_nii.get_fdata()
                    #print(np.unique(tract_arr))
                    
                    tract_arr *= meas_df.at[f'{fiber}.change_score', 'annualized percent change']
                    all_tracts_arr += tract_arr
                except Exception as e:
                    pass
            else:
                pass
        meas_nimg = nib.Nifti1Image(all_tracts_arr, tract_nii.affine)
        plt.figure(layout='tight')
        #fig,ax = plt.subplots(ncols=2, gridspec_kw=grid_kw, figsize=(24,4))
        q = plotting.plot_stat_map(meas_nimg, display_mode='z',  threshold=0.01,
                               cut_coords=[-10,3,18,40], 
                               black_bg=False,
                                   vmax=vmax*1.1, 
                                   annotate=True, cmap=pals[measure], colorbar=False,
                                   #axes=ax[0]
                              )
        q.savefig(f'{PROJ_DIR}/figures/APC_{measure}.png', dpi=400)
        r = plotting.plot_glass_brain(meas_nimg, display_mode='lyrz',  threshold=.01,
                               #cut_coords=[35,50,65,85], 
                               black_bg=False, alpha=0.4,
                                   vmax=vmax*1.1, 
                                   annotate=False, cmap=pals[measure], colorbar=False,
                                   #axes=ax[0]
                              )
        #ax[1].set_visible(False)
        r.savefig(f'{PROJ_DIR}/figures/APC_{measure}-glass.png', dpi=400)
    else:
        #print(nifti_mapping.loc[meas_vars]['atlas_fname'])
        atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
        print(atlas_fname)
        atlas_nii = nib.load(atlas_fname)
        atlas_arr = atlas_nii.get_fdata()
        plotting_arr = np.zeros(atlas_arr.shape)
        for i in meas_df.index:
            j = i.split('.')[0]
            value = nifti_mapping.loc[j]['atlas_value']
            #print(i, value)
            if value is np.nan:
                pass
            else:
                avg = descriptives.at[i, 'annualized percent change']
                plotting_arr[np.where(atlas_arr == value)] = avg
        meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
        if 'subcortical' in measure:
            fig,ax = plt.subplots()
            #plt.figure(layout='tight')
            q = plotting.plot_stat_map(meas_nimg, display_mode='z',  threshold=0.01,
                                   cut_coords=[-20, -10, 0, 10], vmax=vmax*1.1, 
                                   annotate=False, cmap=pals[measure], colorbar=False,
                                   symmetric_cbar=False, axes=ax)

            fig.savefig(f'{PROJ_DIR}/figures/APC_{measure}.png', dpi=400, bbox_inches='tight')
            plt.close(fig)
        elif 'cortical' in measure:
            figure = plot_surfaces(meas_nimg, fsaverage, pals[measure], vmax, 0.01)
            figure.savefig(f'{PROJ_DIR}/figures/APC_{measure}.png', dpi=400, bbox_inches='tight')
            plt.close(figure)

# gather variables (network names) for plotting connectivity
corrs = descriptives.filter(regex='rsfmri_c_ngd.*', axis=0).index
corrs = [i.split('.')[0] for i in corrs]
networks = list(np.unique([i.split('_')[-1] for i in corrs]))

corrs = descriptives.filter(regex='rsfmri_c_ngd.*', axis=0).index
corrs = [i.split('.')[0] for i in corrs]
networks = list(np.unique([i.split('_')[-1] for i in corrs]))

btwn_fc_src = [i.split('.')[0].split('_')[3] for i in btwn_fc]
btwn_fc_trgt = [i.split('.')[0].split('_')[-1] for i in btwn_fc]

vmax = 3.5

# okay, now we're plotting between and within network connectivity
#within-network fc is easy to plot bc there's only one HSK value per network (per fligner_var)
meas_df = descriptives.loc[wthn_fc]
meas_vars = [i.split('.')[0] for i in meas_df.index]
atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
#print(atlas_fname)
atlas_nii = nib.load(atlas_fname)
atlas_arr = atlas_nii.get_fdata()
plotting_arr = np.zeros(atlas_arr.shape)
for i in meas_df.index:
    j = i.split('.')[0]
    value = nifti_mapping.loc[j]['atlas_value']
    #print(i, value)
    if value is np.nan:
        pass
    else:
        plotting_arr[np.where(atlas_arr == value)] = descriptives.at[i,'annualized percent change']

meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
figure = plot_surfaces(meas_nimg, fsaverage, func_cmap, vmax, .01)
figure.savefig(f'{PROJ_DIR}/figures/APCxFCw.png', dpi=400)

scs_varnames = [i.split('.')[0].split('_')[-1] for i in fc_scor_var]


# now subcortical-cortical functional connectivity
sig = []
meas_df = descriptives.loc[fc_scor_var]

meas_df.loc[fc_scor_var, 'scs'] = scs_varnames
avgs = pd.DataFrame()
for scs in np.unique(scs_varnames):
    temp_df = meas_df[meas_df['scs'] == scs]
    # calculate average change of all 
    # significantly heteroscedastic network connections

    for i in temp_df.index:
        sig.append(temp_df.loc[i,'annualized percent change'])
    mean_apc = np.mean(sig)
    #print(mean_hsk)
    # grab name of first conn var for this network for plotting
    avgs.at[temp_df.index[0], 'apc'] = mean_apc
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
        plotting_arr[np.where(atlas_arr == value)] = avgs.at[i,'apc']        
meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
fig,ax = plt.subplots(#ncols=2, gridspec_kw=grid_kw, figsize=(24,4)
                     )
#plt.figure(layout='tight')
q = plotting.plot_stat_map(meas_nimg, display_mode='z',  threshold=.01,
                       cut_coords=[-20, -10, 0, 10], vmax=vmax*1.1, 
                       annotate=False, cmap=func_cmap, colorbar=False,
                       symmetric_cbar=False, axes=ax)

#ax[1].set_visible(False)
fig.savefig(f'{PROJ_DIR}/figures/APCxFCs_scs.png', dpi=400, bbox_inches='tight')
plt.close(fig)

# between-network FC is tough bc we have to average all of a networks HSK values
# but only the significantly HSK connections
sig = []
meas_df = descriptives.loc[btwn_fc]
meas_df.loc[btwn_fc, 'from_ntwk'] = btwn_fc_src
meas_df.loc[btwn_fc, 'to_ntwk'] = btwn_fc_trgt
avgs = pd.DataFrame()
for ntwk in np.unique(btwn_fc_src):
    temp_df = meas_df[meas_df['from_ntwk'] == ntwk]
    temp_df2 = meas_df[meas_df['to_ntwk'] == ntwk]
    temp_df = pd.concat([temp_df, temp_df2], axis=0)
    # calculate average heteroscedasticity of all 
    # significantly heteroscedastic network connections
    for i in temp_df.index:
        sig.append(temp_df.loc[i,'annualized percent change'])
    mean_hsk = np.mean(sig)
    # grab name of first conn var for this network for plotting
    avgs.at[temp_df.index[0], 'apc'] = mean_hsk
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
        plotting_arr[np.where(atlas_arr == value)] = avgs.at[i,'apc']        
meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
figure = plot_surfaces(meas_nimg, fsaverage, func_cmap, vmax, 0.01)
figure.savefig(f'{PROJ_DIR}/figures/APCxFCb.png', dpi=400)


morph_cmap = sns.diverging_palette(22, 256.3, s=70, l=50, center="light", n=6, as_cmap=True)
func_cmap = sns.diverging_palette(343, 140.9, s=70, l=50, center="light", n=6, as_cmap=True)
cell_cmap = sns.diverging_palette(71, 294.3, s=70, l=50, center="light", n=6, as_cmap=True)


fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

vmax = 1.5
range_ = np.arange(-int(vmax*5), int(vmax*5)) / 5.

cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
                               cmap=morph_cmap, 
                               values=range_, 
                              )
ax.set_xlabel('Annualized percent change')

plt.savefig(f'{PROJ_DIR}/figures/morph-cmap_1-{-vmax,vmax}.png', bbox_inches='tight', dpi=400)

cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
                               cmap=cell_cmap, 
                               values=range_, 
                              )
ax.set_xlabel('Annualized percent change')
plt.savefig(f'{PROJ_DIR}/figures/cell-cmap_1-{-vmax,vmax}.png', bbox_inches='tight', dpi=400)

vmax = 3.5
range_ = np.arange(-int(vmax*5), int(vmax*5)) / 5.

cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
                               cmap=func_cmap, 
                               values=range_, 
                              )
ax.set_xlabel('Annualized percent change')
plt.savefig(f'{PROJ_DIR}/figures/func-cmap_1-{-vmax,vmax}.png', bbox_inches='tight', dpi=400)
