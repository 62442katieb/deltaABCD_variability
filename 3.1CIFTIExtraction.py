#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns

import nibabel as nib
from os.path import join
from nilearn import surface, plotting, datasets
from nilearn.image import resample_to_img


def series_2_nifti(series_in, out_dir, save=False):
    nifti_mapping = pd.read_pickle('/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/data/variable_to_nifti_mapping.pkl')
    series = series_in.copy()
    series.index = [x.split('.')[0] for x in series.index]
    
    #vmin = series.quantile(0.25)

    # list of measures to plot
    measures = {'cortical-thickness': 'smri_thick_cdk_.*',
                'cortical-gwcontrast': 'smri_t1wcnt_cdk_.*',
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
    fc_cort_var = series.filter(regex='.*fmri.*_c_.*').index
    fc_scor_var = series.filter(regex='.*fmri.*_cor_.*').index
    fmri_var_var = series.filter(regex='.*fmri.*_var_.*').index

    #morph_var = df[df['concept'] == 'macrostructure'].index
    #cell_var = df[df['concept'] == 'microstructure'].index
    func_var = list(fmri_var_var) 
    conn_var = list(fc_cort_var) + list(fc_scor_var)

    conn_measures = {'cortical-network-connectivity': 'rsfmri_c_ngd_.*',
                'subcortical-network-connectivity': 'rsfmri_cor_ngd_.*_scs_.*',}

    # let's plot APC on brains pls
    for measure in measures.keys():
        #print(measure, measures[measure])
        #print(measure)

        meas_df = series.filter(regex=measures[measure], axis=0)
        meas_vars = meas_df.index

        #meas_df.drop_duplicates(inplace=True)
        #print(len(meas_df.index))
        #print(meas_df.head())
        if len(meas_df[meas_df != 0]) == 0:
            pass
        else:
            if 'tract' in measure:
                print('tract')
                fibers = nifti_mapping.filter(regex=measures[measure], axis=0).index
                var = fibers[0]
                tract_fname = nifti_mapping.loc[var]['atlas_fname']
                tract_nii = nib.load(tract_fname)
                tract_arr = tract_nii.get_fdata()
                #print(np.unique(tract_arr))
                avg = series.loc[f'{var}']
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
                            avg = series.loc[f'{var}']
                            tract_arr *= avg
                            all_tracts_arr += tract_arr
                        except Exception as e:
                            pass
                    else:
                        pass
                meas_nimg = nib.Nifti1Image(all_tracts_arr, tract_nii.affine)
                if save:
                    meas_nimg.to_filename(f'{out_dir}/{series.name}.nii')
                
            else:
                print('cortex')
                #print(nifti_mapping.loc[meas_vars]['atlas_fname'])
                atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
                #print(atlas_fname)
                atlas_nii = nib.load(atlas_fname)
                atlas_arr = atlas_nii.get_fdata()
                plotting_arr = np.zeros(atlas_arr.shape)
                for i in meas_df.index:
                    if i in nifti_mapping.index:
                        value = nifti_mapping.loc[i]['atlas_value']
                        
                        #print(i, value)
                        if value is np.nan:
                            pass
                        
                        else:
                            val = series.at[i]
                            #print(avg, value, atlas_arr.shape)
                            plotting_arr[np.where(atlas_arr == value)] = val
                    else:
                        pass
                
                meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
                #print(np.mean(plotting_arr))
                if save:
                    meas_nimg.to_filename(f'{out_dir}/{series.name}.nii')

    
    return meas_nimg


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

age_10 = '/Users/katherine.b/Dropbox/Projects/deltaABCD_clustering/resources/Age10_DevelopmentMap_Derivatives.pscalar.nii'
sa_axis_path = '/Users/katherine.b/Dropbox/Mac/Downloads/SensorimotorAssociation.Axis.Glasser360.pscalar.nii'
glasser = '/Users/katherine.b/Dropbox/Mac/Downloads/glasser360MNI.nii.gz'
desikan = '/Users/katherine.b/nilearn_data/neurovault/collection_1446/image_23262.nii.gz'
nifti_mapping = pd.read_pickle('/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/data/variable_to_nifti_mapping.pkl')

glasser_nii = nib.load(glasser)
glasser_arr = glasser_nii.get_fdata()

cifti = nib.load(age_10)

age_fx = np.asanyarray(cifti.get_fdata())

age_arr = np.zeros_like(glasser_arr)
for i in range(0,age_fx.shape[1]):
    temp = np.where(glasser_arr == i+1, age_fx[0,i], 0)
    age_arr += temp

age_nimg = nib.Nifti1Image(age_arr, glasser_nii.affine)

pal = sns.blend_palette(['#FCD975', '#FFFFFF', '#926EAE'])
cmap = sns.blend_palette(['#FCD975', '#FFFFFF', '#926EAE'], as_cmap=True)

plotting.plot_img_on_surf(
    age_nimg, 
    threshold=0.0001, 
    cmap=cmap, 
    vmax=0.02, 
    output_file=join(
        PROJ_DIR, 
        FIGS_DIR, 
        'sydnor_age_fx.png'
        )
    )

age_nimg.to_filename(
    join(
        PROJ_DIR,
        DATA_DIR,
        'sydnor_age10_fx.nii'
        )
    )

sa_axis_gii = nib.load(sa_axis_path)
sa_axis_arr = sa_axis_gii.get_fdata()

sa_arr = np.zeros_like(glasser_arr)
for i in range(0,age_fx.shape[1]):
    temp = np.where(glasser_arr == i+1, sa_axis_arr[0,i], 0)
    sa_arr += temp

sa_nimg = nib.Nifti1Image(sa_arr, glasser_nii.affine)
#sns.histplot(sa_arr.flatten()[sa_arr.flatten() != 0])

plotting.plot_img_on_surf(
    sa_nimg, 
    #threshold=0.0001, 
    cmap=cmap, 
    #vmax=0.02,
    symmetric_cmap=False,
    output_file=join(
        PROJ_DIR, 
        FIGS_DIR, 
        'sydnor_sa_axis.png'
        )
)

fsaverage = datasets.fetch_surf_fsaverage()


sa_left = surface.vol_to_surf(sa_nimg, fsaverage.pial_left)

sa_nimg.to_filename(
    join(
        PROJ_DIR,
        DATA_DIR,
        'sensorimotor_association_axis.nii'
        )
    )

#g = plotting.plot_stat_map(age_nimg)
#g.add_contours(desikan)
#g.savefig()

dk_nii = nib.load(desikan)
dk_arr = dk_nii.get_fdata()

resampled_age = resample_to_img(age_nimg, dk_nii, )
resampled_sa = resample_to_img(sa_nimg, dk_nii, )

#g = plotting.plot_stat_map(resampled_sa)
#g.add_contours(desikan)

resampled_age_arr = resampled_age.get_fdata()
resampled_saa_arr = resampled_sa.get_fdata()


rnd_vars = nifti_mapping.filter(like="dmri_rsirndgm_cdk", axis=0)

rnd_df = pd.DataFrame(index=rnd_vars.index)
for i in rnd_vars.index:
    #regn = i.split('_')[-1]
    val = nifti_mapping.loc[i]['atlas_value']
    rnd_df.at[i, 'SA_avg'] = np.nanmean(resampled_saa_arr[np.where(dk_arr == val)])
    rnd_df.at[i, 'age_avg'] = np.nanmean(resampled_age_arr[np.where(dk_arr == val)])
    
rnd_df = rnd_df.sort_values('SA_avg')
rnd_df['SA_rank'] = list(range(1,69))

rnd_df['hemi'] = [i.split('_')[-1][-2:] for i in rnd_df.index]

left_sorted = rnd_df[rnd_df['hemi'] == 'lh'].sort_values('SA_avg')
left_sorted['SA_rank'] = list(range(1,35))

right_sorted = rnd_df[rnd_df['hemi'] == 'rh'].sort_values('SA_avg')
right_sorted['SA_rank'] = list(range(1,35))
rnd_df2 = pd.concat([left_sorted, right_sorted])

rnd_df['SA_rank'] = rnd_df['SA_rank'] - rnd_df['SA_rank'].mean()
rnd_df2['SA_rank'] = rnd_df2['SA_rank'] - rnd_df2['SA_rank'].mean()

sa_rank = series_2_nifti(rnd_df['SA_rank'], '../output', save=False)
sa_rank2 = series_2_nifti(rnd_df2['SA_rank'], '../output', save=False)

plotting.plot_img_on_surf(sa_rank, cmap=cmap, threshold=0.01, )

plotting.plot_img_on_surf(sa_rank2, cmap=cmap, threshold=0.01)

rni_vars = nifti_mapping.filter(like="dmri_rsirnigm_cdk", axis=0)

rni_df = pd.DataFrame(index=rni_vars.index)
for i in rni_vars.index:
    #regn = i.split('_')[-1]
    val = nifti_mapping.loc[i]['atlas_value']
    rni_df.at[i, 'SA_avg'] = np.nanmean(resampled_saa_arr[np.where(dk_arr == val)])
    rni_df.at[i, 'age_avg'] = np.nanmean(resampled_age_arr[np.where(dk_arr == val)])
    
rni_df = rni_df.sort_values('SA_avg')
rni_df['SA_rank'] = list(range(1,69))


var_vars = nifti_mapping.filter(like="rsfmri_var_cdk", axis=0)

var_df = pd.DataFrame(index=var_vars.index)
for i in var_vars.index:
    #regn = i.split('_')[-1]
    val = nifti_mapping.loc[i]['atlas_value']
    var_df.at[i, 'SA_avg'] = np.nanmean(resampled_saa_arr[np.where(dk_arr == val)])
    var_df.at[i, 'age_avg'] = np.nanmean(resampled_age_arr[np.where(dk_arr == val)])
    
var_df = var_df.sort_values('SA_avg')
var_df['SA_rank'] = list(range(1,69))

thk_vars = nifti_mapping.filter(like="smri_thick_cdk", axis=0)

thk_df = pd.DataFrame(index=thk_vars.index)
for i in thk_vars.index:
    #regn = i.split('_')[-1]
    val = nifti_mapping.loc[i]['atlas_value']
    thk_df.at[i, 'SA_avg'] = np.nanmean(resampled_saa_arr[np.where(dk_arr == val)])
    thk_df.at[i, 'age_avg'] = np.nanmean(resampled_age_arr[np.where(dk_arr == val)])
    
thk_df = thk_df.sort_values('SA_avg')
thk_df['SA_rank'] = list(range(1,69))

thk_df.to_pickle(join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.pkl'))
var_df.to_pickle(join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.pkl'))
rni_df.to_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.pkl'))
rnd_df.to_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.pkl'))