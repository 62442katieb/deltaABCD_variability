import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl

from os.path import join

from scipy.stats import fligner, t
from nilearn import plotting, datasets, surface

def plot_surfaces(nifti, surf, cmap, vmax, threshold):
    '''
    Plots of medial and lateral left and right surface views from nifti volume
    '''
    
    texture_l = surface.vol_to_surf(nifti, surf.pial_left, interpolation='nearest')
    texture_r = surface.vol_to_surf(nifti, surf.pial_right, interpolation='nearest')
    
    fig = plt.figure(figsize=(12,4))
    gs = GridSpec(1, 4)

    ax0 = fig.add_subplot(gs[0], projection='3d')
    ax1 = fig.add_subplot(gs[1], projection='3d')
    ax2 = fig.add_subplot(gs[2], projection='3d')
    ax3 = fig.add_subplot(gs[3], projection='3d')
    plt.tight_layout(w_pad=-1, h_pad=-1)
    figure = plotting.plot_surf_stat_map(surf.pial_left, 
                                         texture_l, 
                                         symmetric_cbar=False, 
                                         threshold=threshold,
                                         cmap=cmap, 
                                         view='lateral', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax0)
    figure = plotting.plot_surf_stat_map(surf.pial_left, 
                                         texture_l, 
                                         symmetric_cbar=False, 
                                         threshold=threshold,     
                                         cmap=cmap, 
                                         view='medial', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax1)
    figure = plotting.plot_surf_stat_map(surf.pial_right, 
                                         texture_r, 
                                         symmetric_cbar=False, 
                                         threshold=threshold,
                                         cmap=cmap, 
                                         view='lateral', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax2)
    figure = plotting.plot_surf_stat_map(surf.pial_right, 
                                         texture_r, 
                                         symmetric_cbar=False, 
                                         threshold=threshold,     
                                         cmap=cmap, 
                                         view='medial', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax3)
    return figure

# set up colormaps & palettes for plotting later
morph_pal = sns.cubehelix_palette(start=0.6, rot=-0.6, gamma=1.0, hue=1, light=0.7, dark=0.3)
morph_cmap = sns.cubehelix_palette(n_colors=4, start=0.6, rot=-0.6, gamma=1.0, hue=0.7, light=1, dark=0.2, 
                                   as_cmap=True, reverse=True)
cell_pal = sns.cubehelix_palette(start=1.7, rot=-0.8, gamma=1.0, hue=1, light=0.7, dark=0.3)
cell_cmap = sns.cubehelix_palette(n_colors=9, start=1.7, rot=-0.8, gamma=1.0, hue=0.7, light=0.5, dark=0.2, 
                                  as_cmap=True, reverse=True)
func_pal = sns.cubehelix_palette(start=3.0, rot=-0.6, gamma=1.0, hue=1, light=0.7, dark=0.3)
func_cmap = sns.cubehelix_palette(n_colors=4, start=3.0, rot=-0.6, gamma=1.0, hue=0.7, light=0.6, dark=0.2, 
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
desc_summ = pd.DataFrame(index=measures, columns=['mean', 'sdev', '95%CI'])
for measure in measures:
    temp_df = descriptives[descriptives['measure'] == measure]
    mean = np.mean(temp_df['annualized percent change'])
    desc_summ.at[measure, 'mean'] = np.round(mean,2)
    sdev = np.mean(temp_df['sdev'])
    desc_summ.at[measure, 'sdev'] = np.round(sdev,2)
    dof = len(temp_df.index)-1 
    confidence = 0.95
    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    CI = (np.round(mean - sdev *t_crit / np.sqrt(dof + 1), 2), np.round(mean + sdev * t_crit / np.sqrt(dof + 1),2)) 
    desc_summ.at[measure, '95%CI'] = CI
desc_summ.to_csv(join(PROJ_DIR, OUTP_DIR, 'apchange_descriptives.csv'))

# read in mapping from tabular data to nifti parcellations
nifti_mapping = pd.read_csv(join(PROJ_DIR, 
                                 DATA_DIR, 
                                 'variable_to_nifti_mapping.csv'), 
                            header=0, 
                            index_col=0)

# list of measures to plot
measures = {'cortical-thickness': 'smri_thick_cdk_*change_score',
            'cortical-gwcontrast': 'smri_t1wcnt_cdk_*change_score',
            'cortical-area': 'smri_area_cdk_.*change_score',
            'cortical-volume': 'smri_vol_cdk_.*change_score', 
            'subcortical-volume': 'smri_vol_scs_.*change_score', 
            'subcortical-RND': 'dmri_rsirnd_scs_.*change_score',
            'subcortical-RNI': 'dmri_rsirni_scs_.*change_score',
            'cortical-RND': 'dmri_rsirndgm_.*change_score',
            'cortical-RNI': 'dmri_rsirnigm_.*change_score',
            'cortical-BOLD-variance': 'rsfmri_var_cdk_.*change_score',
            'tract-volume': 'dmri_dtivol_fiberat_.*change_score', 
            'tract-FA': 'dmri_dtifa_fiberat_.*change_score', 
            'tract-MD': 'dmri_dtimd_fiberat_.*change_score',
            'tract-LD': 'dmri_dtild_fiberat_.*change_score', 
            'tract-TD': 'dmri_dtitd_fiberat_.*change_score', 
            'tract-RND': 'dmri_rsirnd_fib_.*change_score',
            'tract-RNI': 'dmri_rsirni_fib_.*change_score'}

conn_measures = {'cortical-network-connectivity': 'rsfmri_c_ngd_.*change_score',
            'subcortical-network-connectivity': 'rsfmri_cor_ngd_.*_scs_.*change_score',}

fsaverage = datasets.fetch_surf_fsaverage()

morph_cmap = sns.diverging_palette(22, 256.3, s=70, l=50, center="light", as_cmap=True)
func_cmap = sns.diverging_palette(343, 140.9, s=70, l=50, center="light", as_cmap=True)
cell_cmap = sns.diverging_palette(71, 294.3, s=70, l=50, center="light", as_cmap=True)
morph_pal = sns.diverging_palette(22, 256.3, s=70, l=50, center="light", as_cmap=False)
func_pal = sns.diverging_palette(343, 140.9, s=70, l=50, center="light", as_cmap=False)
cell_pal = sns.diverging_palette(71.0, 294.3, s=70, l=50, center="light", as_cmap=False)
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
for measure in measures:
    if 'BOLD' in measure:
        vmax = 3.5
    else: 
        vmax = 1.5
    print(measure)
    meas_df = descriptives.filter(regex=measures[measure], axis=0)
    meas_vars = [i.split('.')[0] for i in meas_df.index]
    #print(nifti_mapping.loc[meas_vars]['atlas_fname'])
    atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
    print(atlas_fname)
    atlas_nii = nib.load(atlas_fname)
    atlas_arr = atlas_nii.get_fdata()
    plotting_arr = np.zeros(atlas_arr.shape)
    if 'tract' in measure:
        fibers = nifti_mapping.filter(regex=measures[measure], axis=0).index
        var = fibers[0]
        tract_fname = nifti_mapping.loc[var]['atlas_fname']
        tract_nii = nib.load(tract_fname)
        tract_arr = tract_nii.get_fdata()
        #print(np.unique(tract_arr))
        avg = descriptives.at[f'{var}.change_score', 'annualized percent change']
        tract_arr *= avg
        all_tracts_arr = np.zeros(tract_arr.shape)
        all_tracts_arr += tract_arr
        for var in fibers[1:]:    
            tract_fname = nifti_mapping.loc[var]['atlas_fname']
            if type(tract_fname) is str:
                try:
                    tract_nii = nib.load(tract_fname)
                    tract_arr = tract_nii.get_fdata()
                    #print(np.unique(tract_arr))
                    avg = descriptives.at[f'{var}.change_score', 'annualized percent change']
                    tract_arr *= avg
                    all_tracts_arr += tract_arr
                except Exception as e:
                    pass
            else:
                pass
        meas_nimg = nib.Nifti1Image(all_tracts_arr, tract_nii.affine)
        #plt.figure(layout='tight')
        #fig,ax = plt.subplots(ncols=2, gridspec_kw=grid_kw, figsize=(24,4))
        q = plotting.plot_anat(meas_nimg, display_mode='z',  threshold=0.01,
                            cut_coords=[35,50,65,85], 
                            black_bg=False,
                                vmax=vmax*1.1, 
                                vmin=-vmax*1.1,
                                annotate=False, cmap=pals[measure], colorbar=False,
                                #axes=ax[0]
                            )
        q.close()
        q.savefig(f'{PROJ_DIR}/figures/APC_{measure}.png', dpi=400)
        q = None
    else:
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

            q.savefig(f'{PROJ_DIR}/figures/APC_{measure}.png', dpi=400)
        elif 'cortical' in measure:
            figure = plot_surfaces(meas_nimg, fsaverage, pals[measure], vmax, 0.01)
            figure.savefig(f'{PROJ_DIR}/figures/APC_{measure}.png', dpi=400, bbox_inches='tight')


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