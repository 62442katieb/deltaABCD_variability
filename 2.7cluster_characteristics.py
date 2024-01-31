#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join
from nilearn import plotting

from scipy.stats import spearmanr, pointbiserialr, chi2_contingency
from utils import jili_sidak_mc, assign_region_names, series_2_nifti
#from sklearn.linear_model import LinearRegression


sns.set(style='white', context='talk', font_scale=1.1)
plt.rcParams["font.family"] = "Avenir Next Condensed"



PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

# RUN THIS WITH ALL OF THE DEMOGRAPHIC DATA 
# GO TEAM


imputed_cdk = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_knn-cdk.pkl"))
#imputed_dcg = pd.read_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data.csv")), 
#                          index_col='subjectkey', 
#                          header=0)

numerical = ["interview_age","rsfmri_var_meanmotion",
        "rsfmri_var_subthreshnvols",
        "rsfmri_var_subtcignvols",
        "rsfmri_var_ntpoints","nihtbx_picvocab_uncorrected",
        "nihtbx_flanker_uncorrected",
        "nihtbx_list_uncorrected",
        "nihtbx_cardsort_uncorrected",
        "nihtbx_pattern_uncorrected",
        "nihtbx_picture_uncorrected",
        "nihtbx_reading_uncorrected", "pds_p_ss_female_category_2", 
        "pds_p_ss_male_category_2","cbcl_scr_syn_anxdep_r", 
        "cbcl_scr_syn_withdep_r", 
        "cbcl_scr_syn_somatic_r", 
        "cbcl_scr_syn_social_r", 
        "cbcl_scr_syn_thought_r", 
        "cbcl_scr_syn_attention_r", 
        "cbcl_scr_syn_rulebreak_r", 
        "cbcl_scr_syn_aggressive_r", 
        "cbcl_scr_syn_internal_r", 
        "cbcl_scr_syn_external_r", 
        "cbcl_scr_syn_totprob_r", ]

non_numerical = [#"interview_date", 
        "sex", "site_id_l", "demo_prnt_ethn_v2",
        "demo_prnt_marital_v2",
        "demo_prnt_ed_v2",
        "demo_comb_income_v2"]

vars_of_interest = numerical + non_numerical

cols = []
for var in numerical:
    columns = imputed_cdk.filter(regex=f'{var}.*', axis=1).columns
    if len(columns) == 0:
        pass
    else:
        for column in columns:
            cols.append(column)

#print(len(cols))
#print(cols)
num_df = imputed_cdk[cols]

cols = []
for var in non_numerical:
    columns = imputed_cdk.filter(regex=f'{var}.*', axis=1).columns
    if len(columns) == 0:
        pass
    else:
        for column in columns:
            cols.append(column)

#print(len(cols))
#print(cols)
non_df = imputed_cdk[cols]

brain_df = imputed_cdk.filter(regex='.*mri.*change_score', axis=1)

atlas = 'desikankillany'

siemens = join(PROJ_DIR, OUTP_DIR, 'responsibilities-siemens-2023-07-08.csv')
ge = join(PROJ_DIR, OUTP_DIR, 'responsibilities-ge-2023-07-08.csv')
philips = join(PROJ_DIR, OUTP_DIR, 'responsibilities-philips-2023-07-08.csv')
siemens_df = pd.read_csv(siemens, index_col=0, header=0)
ge_df = pd.read_csv(ge, index_col=0, header=0)
philips_df = pd.read_csv(philips, index_col=0, header=0)

scanners = {'SIEMENS': siemens_df,
            'GE MEDICAL SYSTEMS': ge_df, 
            "Philips Medical Systems": philips_df, 
            }

brain_meas = {
    'thick': 'smri_thick.*change_score',
    'rni': 'dmri_rsirni.*change_score',
    'rnd': 'dmri_rsirnd.*change_score',
    'var': 'rsfmri_var.*change_score',
}



big_df = pd.DataFrame()
for scanner in scanners.keys():
    # find the smaller cluster
    print(scanner)
    dat = scanners[scanner]
    ppt_per_clust = dat.sum()
    ppt_per_clust = ppt_per_clust[ppt_per_clust > 0]
    dat = dat[ppt_per_clust.index]
    #print(dat.head())
    print(dat.sum())
    #small = dat.sum().argmin()
    dat.columns = [f'{scanner}_{col}' for col in dat.columns]
    #large = dat.sum().argmax()
    #list_ = ['', '']
    #list_[small] = 'small'
    #list_[large] = 'large'
    #dat.columns = list_
    #print(dat.columns)
    alpha, _ = jili_sidak_mc(num_df, 0.05)

    stats = ['t', 'p']
    clusts = dat.columns
    index = pd.MultiIndex.from_product([stats, clusts, clusts])
    #print(index)
    spearman_df = pd.DataFrame(
    index=num_df.columns,
    columns=pd.MultiIndex.from_product([dat, ['r', 'p']])
    )

    temp = num_df.loc[dat.index]
    for column in num_df.columns:
        for component in dat.columns:
            double_temp = pd.concat([temp[column], dat[component]], axis=1).dropna()
            if component == 2:
                r,p = pointbiserialr(double_temp[component], double_temp[column])
                spearman_df.at[column, (component, 'r')] = r
                spearman_df.at[column, (component, 'p')] = p
            else:
                r,p = spearmanr(double_temp[component], double_temp[column])
                spearman_df.at[column, (component, 'r')] = r
                spearman_df.at[column, (component, 'p')] = p

        #if p < alpha:
            #print(column, np.round(r, 2), np.round(p, 3))
    #spearman_df
    spearman_df.dropna(how='all', axis=1).to_csv(join(PROJ_DIR, OUTP_DIR, f'numerical_corrs_by_cluster_{scanner}.csv'))

    stats = ['x2', 'p']
    index = pd.MultiIndex.from_product([stats, clusts, clusts])
    
    table = pd.DataFrame(columns=dat.columns)
    chisq = pd.DataFrame(columns=['x2', 'p'], index=non_df.columns)
    #print(dat.columns)
    for non in non_df.columns:
        dummies = pd.get_dummies(non_df[non])
        #print(dummies.columns)
        small_table = pd.DataFrame(columns=dummies.columns, index=dat.columns)
        for dumb in dummies.columns:
            if dumb not in [777., 999.]:
                vals = dummies[dumb]
                
                if vals[dat.index].sum() == 0:
                    pass
                else:
                    for cluster in dat.columns:
                        ppts1 = dat[dat[cluster] > .1].index
                        one_vals = vals.loc[ppts1]
                        
                        table.at[f'{non}_{dumb}', cluster] = one_vals.sum()
                        small_table.at[cluster, dumb] = one_vals.sum()
        #print(small_table)
        if small_table.sum().sum() > 0:
            x2,p,_,_ = chi2_contingency(small_table.dropna(how='all', axis=1).dropna(how='all', axis=0))
            #print(non, x2, p)
            chisq.at[non, ('x2')] = x2
            chisq.at[non, ('p')] = p
        else:
            pass
        # do a chi-square test for differences across clusters for each variable.
    table.dropna(how='all', axis=1).to_csv(join(PROJ_DIR, 'output', f'categorical_vals_by_cluster_{scanner}.csv'))
    chisq.dropna(how='all', axis=1).to_csv(join(PROJ_DIR, 'output', f'non_numerical_differences_by_cluster_{scanner}.csv'))

    spearman_brain_df = pd.DataFrame(
        index=brain_df.columns,
        columns=pd.MultiIndex.from_product([dat, ['r', 'p']])
        )
    alpha,_ = jili_sidak_mc(brain_df, 0.05)
    nlog_alpha = -1 * np.log10(alpha)
    temp = brain_df.loc[dat.index]
    for column in brain_df.columns:
        for component in dat.columns:
            double_temp = pd.concat([temp[column], dat[component]], axis=1).dropna()
            #print(double_temp.head())
            if len(dat[component].unique()) == 2:
                r,p = pointbiserialr(double_temp[component], double_temp[column])
                
                spearman_brain_df.at[column, (component, 'r')] = r
                spearman_brain_df.at[column, (component, 'p')] = p
            else:
                #print(column, component, r, p)
                r,p = spearmanr(double_temp[component], double_temp[column])
                spearman_brain_df.at[column, (component, 'r')] = r
                spearman_brain_df.at[column, (component, 'p')] = p
    spearman_brain_df = assign_region_names(spearman_brain_df)            
    spearman_brain_df.to_csv(join(PROJ_DIR, OUTP_DIR, f'brain_corrs_by_cluster_{scanner}.csv'))
    for meas in brain_meas:
        for component in dat.columns:
            brain_vals = spearman_brain_df.filter(regex=brain_meas[meas], axis=0)[component]
            brain_vals = brain_vals['p']
            nlogp_ser = pd.Series(
                index=brain_vals.index, 
                name=f'{component}-{meas}-nlogp',
                dtype=float
                )
            for val in brain_vals.index:
                nlogp = -np.log10(brain_vals[val])
                if spearman_brain_df.loc[val][(component, 'r')] > 0:
                    pass
                else:
                    nlogp *= -1
                nlogp_ser.at[val] = nlogp
            series_2_nifti(nlogp_ser, 'figures/')
            #brain_vals.name = f'{component}-{meas}'
            #print(nlogp_ser.head())
            try:
                plotting.plot_img_on_surf(
                    f'figures/{component}-{meas}-nlogp.nii',
                    cmap='bwr', 
                    threshold=nlog_alpha, 
                    symmetric_cbar=True, 
                    output_file=f'figures/{component}-{meas}-nlogp.png')
            except Exception as e:
                print(e)