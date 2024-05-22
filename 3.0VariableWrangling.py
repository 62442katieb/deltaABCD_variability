#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import pingouin as pg
import abcdWrangler as abcdw

from os.path import join
import warnings

warnings.filterwarnings("ignore")

def residualize(X, y=None, confounds=None, return_regs=False):
    '''
    all inputs need to be arrays, not dataframes
    '''
    x_regs = []
    # residualize the outcome
    if confounds is not None:
        if y is not None:
            temp_y = np.reshape(y, (y.shape[0],))
            y = pg.linear_regression(confounds, temp_y)
            resid_y = y.residuals_

            # residualize features
            resid_X = np.zeros_like(X)
            # print(X.shape, resid_X.shape)
            for i in range(0, X.shape[1]):
                X_temp = X[:, i]
                # print(X_temp.shape)
                X_ = pg.linear_regression(confounds, X_temp)
                x_regs.append(X_)
                # print(X_.residuals_.shape)
                resid_X[:, i] = X_.residuals_.flatten()
            if return_regs:
                return resid_y, y, resid_X, x_regs
            else:
                return resid_y, resid_X
        else:
            # residualize features
            resid_X = np.zeros_like(X)
            # print(X.shape, resid_X.shape)
            for i in range(0, X.shape[1]):
                X_temp = X[:, i]
                # print(X_temp.shape)
                X_ = pg.linear_regression(confounds, X_temp)
                x_regs.append(X_)
                # print(X_.residuals_.shape)
                resid_X[:, i] = X_.residuals_.flatten()
            if return_regs:
                return resid_X, x_regs
            else:
                return resid_X

ABCD_DIR = "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.1/abcd-data-release-5.1"
PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

variables = [
    "smri_thick_cdk_banksstslh",
    "smri_thick_cdk_cdacatelh",
    "smri_thick_cdk_cdmdfrlh",
    "smri_thick_cdk_cuneuslh",
    "smri_thick_cdk_ehinallh",
    "smri_thick_cdk_fusiformlh",
    "smri_thick_cdk_ifpllh",
    "smri_thick_cdk_iftmlh",
    "smri_thick_cdk_ihcatelh",
    "smri_thick_cdk_locclh",
    "smri_thick_cdk_lobfrlh",
    "smri_thick_cdk_linguallh",
    "smri_thick_cdk_mobfrlh",
    "smri_thick_cdk_mdtmlh",
    "smri_thick_cdk_parahpallh",
    "smri_thick_cdk_paracnlh",
    "smri_thick_cdk_parsopclh",
    "smri_thick_cdk_parsobislh",
    "smri_thick_cdk_parstgrislh",
    "smri_thick_cdk_pericclh",
    "smri_thick_cdk_postcnlh",
    "smri_thick_cdk_ptcatelh",
    "smri_thick_cdk_precnlh",
    "smri_thick_cdk_pclh",
    "smri_thick_cdk_rracatelh",
    "smri_thick_cdk_rrmdfrlh",
    "smri_thick_cdk_sufrlh",
    "smri_thick_cdk_supllh",
    "smri_thick_cdk_sutmlh",
    "smri_thick_cdk_smlh",
    "smri_thick_cdk_frpolelh",
    "smri_thick_cdk_tmpolelh",
    "smri_thick_cdk_trvtmlh",
    "smri_thick_cdk_insulalh",
    "smri_thick_cdk_banksstsrh",
    "smri_thick_cdk_cdacaterh",
    "smri_thick_cdk_cdmdfrrh",
    "smri_thick_cdk_cuneusrh",
    "smri_thick_cdk_ehinalrh",
    "smri_thick_cdk_fusiformrh",
    "smri_thick_cdk_ifplrh",
    "smri_thick_cdk_iftmrh",
    "smri_thick_cdk_ihcaterh",
    "smri_thick_cdk_loccrh",
    "smri_thick_cdk_lobfrrh",
    "smri_thick_cdk_lingualrh",
    "smri_thick_cdk_mobfrrh",
    "smri_thick_cdk_mdtmrh",
    "smri_thick_cdk_parahpalrh",
    "smri_thick_cdk_paracnrh",
    "smri_thick_cdk_parsopcrh",
    "smri_thick_cdk_parsobisrh",
    "smri_thick_cdk_parstgrisrh",
    "smri_thick_cdk_periccrh",
    "smri_thick_cdk_postcnrh",
    "smri_thick_cdk_ptcaterh",
    "smri_thick_cdk_precnrh",
    "smri_thick_cdk_pcrh",
    "smri_thick_cdk_rracaterh",
    "smri_thick_cdk_rrmdfrrh",
    "smri_thick_cdk_sufrrh",
    "smri_thick_cdk_suplrh",
    "smri_thick_cdk_sutmrh",
    "smri_thick_cdk_smrh",
    "smri_thick_cdk_frpolerh",
    "smri_thick_cdk_tmpolerh",
    "smri_thick_cdk_trvtmrh",
    "smri_thick_cdk_insularh",
    "smri_vol_scs_wholeb",
    "dmri_rsirnigm_cdk_bstslh", 
    "dmri_rsirnigm_cdk_caclh", 
    "dmri_rsirnigm_cdk_cmflh", 
    "dmri_rsirnigm_cdk_cnlh", 
    "dmri_rsirnigm_cdk_erlh", 
    "dmri_rsirnigm_cdk_fflh", 
    "dmri_rsirnigm_cdk_iplh", 
    "dmri_rsirnigm_cdk_itlh", 
    "dmri_rsirnigm_cdk_iclh", 
    "dmri_rsirnigm_cdk_lolh", 
    "dmri_rsirnigm_cdk_loflh", 
    "dmri_rsirnigm_cdk_lglh", 
    "dmri_rsirnigm_cdk_moflh", 
    "dmri_rsirnigm_cdk_mtlh", 
    "dmri_rsirnigm_cdk_phlh", 
    "dmri_rsirnigm_cdk_pclh", 
    "dmri_rsirnigm_cdk_poplh", 
    "dmri_rsirnigm_cdk_poblh", 
    "dmri_rsirnigm_cdk_ptglh", 
    "dmri_rsirnigm_cdk_pcclh", 
    "dmri_rsirnigm_cdk_pctlh", 
    "dmri_rsirnigm_cdk_pcglh", 
    "dmri_rsirnigm_cdk_prctlh", 
    "dmri_rsirnigm_cdk_prcnlh", 
    "dmri_rsirnigm_cdk_raclh", 
    "dmri_rsirnigm_cdk_rmflh", 
    "dmri_rsirnigm_cdk_sflh", 
    "dmri_rsirnigm_cdk_splh", 
    "dmri_rsirnigm_cdk_stlh", 
    "dmri_rsirnigm_cdk_smlh", 
    "dmri_rsirnigm_cdk_fplh", 
    "dmri_rsirnigm_cdk_tplh", 
    "dmri_rsirnigm_cdk_ttlh", 
    "dmri_rsirnigm_cdk_islh", 
    "dmri_rsirnigm_cdk_bstsrh", 
    "dmri_rsirnigm_cdk_cacrh", 
    "dmri_rsirnigm_cdk_cmfrh", 
    "dmri_rsirnigm_cdk_cnrh", 
    "dmri_rsirnigm_cdk_errh", 
    "dmri_rsirnigm_cdk_ffrh", 
    "dmri_rsirnigm_cdk_iprh", 
    "dmri_rsirnigm_cdk_itrh", 
    "dmri_rsirnigm_cdk_icrh", 
    "dmri_rsirnigm_cdk_lorh", 
    "dmri_rsirnigm_cdk_lofrh", 
    "dmri_rsirnigm_cdk_lgrh", 
    "dmri_rsirnigm_cdk_mofrh", 
    "dmri_rsirnigm_cdk_mtrh", 
    "dmri_rsirnigm_cdk_phrh", 
    "dmri_rsirnigm_cdk_pcrh", 
    "dmri_rsirnigm_cdk_poprh", 
    "dmri_rsirnigm_cdk_pobrh", 
    "dmri_rsirnigm_cdk_ptgrh", 
    "dmri_rsirnigm_cdk_pccrh", 
    "dmri_rsirnigm_cdk_pctrh", 
    "dmri_rsirnigm_cdk_pcgrh", 
    "dmri_rsirnigm_cdk_prctrh", 
    "dmri_rsirnigm_cdk_prcnrh", 
    "dmri_rsirnigm_cdk_racrh", 
    "dmri_rsirnigm_cdk_rmfrh", 
    "dmri_rsirnigm_cdk_sfrh", 
    "dmri_rsirnigm_cdk_sprh", 
    "dmri_rsirnigm_cdk_strh", 
    "dmri_rsirnigm_cdk_smrh", 
    "dmri_rsirnigm_cdk_fprh", 
    "dmri_rsirnigm_cdk_tprh", 
    "dmri_rsirnigm_cdk_ttrh", 
    "dmri_rsirnigm_cdk_isrh",
    "dmri_rsirndgm_cdk_bstslh", 
    "dmri_rsirndgm_cdk_caclh", 
    "dmri_rsirndgm_cdk_cmflh", 
    "dmri_rsirndgm_cdk_cnlh", 
    "dmri_rsirndgm_cdk_erlh", 
    "dmri_rsirndgm_cdk_fflh", 
    "dmri_rsirndgm_cdk_iplh", 
    "dmri_rsirndgm_cdk_itlh", 
    "dmri_rsirndgm_cdk_iclh", 
    "dmri_rsirndgm_cdk_lolh", 
    "dmri_rsirndgm_cdk_loflh", 
    "dmri_rsirndgm_cdk_lglh", 
    "dmri_rsirndgm_cdk_moflh", 
    "dmri_rsirndgm_cdk_mtlh", 
    "dmri_rsirndgm_cdk_phlh", 
    "dmri_rsirndgm_cdk_pclh", 
    "dmri_rsirndgm_cdk_poplh", 
    "dmri_rsirndgm_cdk_poblh", 
    "dmri_rsirndgm_cdk_ptglh", 
    "dmri_rsirndgm_cdk_pcclh", 
    "dmri_rsirndgm_cdk_pctlh", 
    "dmri_rsirndgm_cdk_pcglh", 
    "dmri_rsirndgm_cdk_prctlh", 
    "dmri_rsirndgm_cdk_prcnlh", 
    "dmri_rsirndgm_cdk_raclh", 
    "dmri_rsirndgm_cdk_rmflh", 
    "dmri_rsirndgm_cdk_sflh", 
    "dmri_rsirndgm_cdk_splh", 
    "dmri_rsirndgm_cdk_stlh", 
    "dmri_rsirndgm_cdk_smlh", 
    "dmri_rsirndgm_cdk_fplh", 
    "dmri_rsirndgm_cdk_tplh", 
    "dmri_rsirndgm_cdk_ttlh", 
    "dmri_rsirndgm_cdk_islh", 
    "dmri_rsirndgm_cdk_bstsrh", 
    "dmri_rsirndgm_cdk_cacrh", 
    "dmri_rsirndgm_cdk_cmfrh", 
    "dmri_rsirndgm_cdk_cnrh", 
    "dmri_rsirndgm_cdk_errh", 
    "dmri_rsirndgm_cdk_ffrh", 
    "dmri_rsirndgm_cdk_iprh", 
    "dmri_rsirndgm_cdk_itrh", 
    "dmri_rsirndgm_cdk_icrh", 
    "dmri_rsirndgm_cdk_lorh", 
    "dmri_rsirndgm_cdk_lofrh", 
    "dmri_rsirndgm_cdk_lgrh", 
    "dmri_rsirndgm_cdk_mofrh", 
    "dmri_rsirndgm_cdk_mtrh", 
    "dmri_rsirndgm_cdk_phrh", 
    "dmri_rsirndgm_cdk_pcrh", 
    "dmri_rsirndgm_cdk_poprh", 
    "dmri_rsirndgm_cdk_pobrh", 
    "dmri_rsirndgm_cdk_ptgrh", 
    "dmri_rsirndgm_cdk_pccrh", 
    "dmri_rsirndgm_cdk_pctrh", 
    "dmri_rsirndgm_cdk_pcgrh", 
    "dmri_rsirndgm_cdk_prctrh", 
    "dmri_rsirndgm_cdk_prcnrh", 
    "dmri_rsirndgm_cdk_racrh", 
    "dmri_rsirndgm_cdk_rmfrh", 
    "dmri_rsirndgm_cdk_sfrh", 
    "dmri_rsirndgm_cdk_sprh", 
    "dmri_rsirndgm_cdk_strh", 
    "dmri_rsirndgm_cdk_smrh", 
    "dmri_rsirndgm_cdk_fprh", 
    "dmri_rsirndgm_cdk_tprh", 
    "dmri_rsirndgm_cdk_ttrh", 
    "dmri_rsirndgm_cdk_isrh",
    "rsfmri_var_cdk_banksstslh",
    "rsfmri_var_cdk_cdaclatelh",
    "rsfmri_var_cdk_cdmdflh",
    "rsfmri_var_cdk_cuneuslh",
    "rsfmri_var_cdk_entorhinallh",
    "rsfmri_var_cdk_fflh",
    "rsfmri_var_cdk_ifpalh",
    "rsfmri_var_cdk_iftlh",
    "rsfmri_var_cdk_ihclatelh",
    "rsfmri_var_cdk_loccipitallh",
    "rsfmri_var_cdk_loboflh",
    "rsfmri_var_cdk_linguallh",
    "rsfmri_var_cdk_moboflh",
    "rsfmri_var_cdk_mdtlh",
    "rsfmri_var_cdk_parahpallh",
    "rsfmri_var_cdk_paracentrallh",
    "rsfmri_var_cdk_parsopllh",
    "rsfmri_var_cdk_parsobalislh",
    "rsfmri_var_cdk_parstularislh",
    "rsfmri_var_cdk_pericclh",
    "rsfmri_var_cdk_postcentrallh",
    "rsfmri_var_cdk_psclatelh",
    "rsfmri_var_cdk_precentrallh",
    "rsfmri_var_cdk_precuneuslh",
    "rsfmri_var_cdk_rlaclatelh",
    "rsfmri_var_cdk_rlmdflh",
    "rsfmri_var_cdk_suflh",
    "rsfmri_var_cdk_spetallh",
    "rsfmri_var_cdk_sutlh",
    "rsfmri_var_cdk_smlh",
    "rsfmri_var_cdk_fpolelh",
    "rsfmri_var_cdk_tpolelh",
    "rsfmri_var_cdk_tvtlh",
    "rsfmri_var_cdk_insulalh",
    "rsfmri_var_cdk_banksstsrh",
    "rsfmri_var_cdk_cdaclaterh",
    "rsfmri_var_cdk_cdmdfrh",
    "rsfmri_var_cdk_cuneusrh",
    "rsfmri_var_cdk_entorhinalrh",
    "rsfmri_var_cdk_ffrh",
    "rsfmri_var_cdk_ifparh",
    "rsfmri_var_cdk_iftrh",
    "rsfmri_var_cdk_ihclaterh",
    "rsfmri_var_cdk_loccipitalrh",
    "rsfmri_var_cdk_lobofrh",
    "rsfmri_var_cdk_lingualrh",
    "rsfmri_var_cdk_mobofrh",
    "rsfmri_var_cdk_mdtrh",
    "rsfmri_var_cdk_parahpalrh",
    "rsfmri_var_cdk_paracentralrh",
    "rsfmri_var_cdk_parsoplrh",
    "rsfmri_var_cdk_parsobalisrh",
    "rsfmri_var_cdk_parstularisrh",
    "rsfmri_var_cdk_periccrh",
    "rsfmri_var_cdk_postcentralrh",
    "rsfmri_var_cdk_psclaterh",
    "rsfmri_var_cdk_precentralrh",
    "rsfmri_var_cdk_precuneusrh",
    "rsfmri_var_cdk_rlaclaterh",
    "rsfmri_var_cdk_rlmdfrh",
    "rsfmri_var_cdk_sufrh",
    "rsfmri_var_cdk_spetalrh",
    "rsfmri_var_cdk_sutrh",
    "rsfmri_var_cdk_smrh",
    "rsfmri_var_cdk_fpolerh",
    "rsfmri_var_cdk_tpolerh",
    "rsfmri_var_cdk_tvtrh",
    "rsfmri_var_cdk_insularh",
    "rsfmri_meanmotion",
    "rsfmri_ntpoints",
    "imgincl_rsfmri_include",
    "imgincl_t1w_include",
    "dmri_meanmotion",
    "imgincl_dmri_include",
    "mrif_score",
    "interview_age",
    "pds_p_ss_female_category_2", 
    "pds_p_ss_male_category_2",  
    "mri_info_manufacturer",
    "mri_info_deviceserialnumber",
    "interview_date",
    'ehi_y_ss_scoreb',
    'mrif_score',
    #'demo_sex_v2'
]

df = dat = abcdw.data_grabber(
    ABCD_DIR, 
    variables, 
    eventname=[
        'baseline_year_1_arm_1', 
        '2_year_follow_up_y_arm_1'
    ],
    multiindex=False,
    long=False
)

# data cleaning
# step 1: pre-covid follow-ups only
df['interview_date.baseline_year_1_arm_1'] = pd.to_datetime(df['interview_date.baseline_year_1_arm_1'], format='%m/%d/%Y')
df['interview_date.2_year_follow_up_y_arm_1'] = pd.to_datetime(df['interview_date.2_year_follow_up_y_arm_1'], format='%m/%d/%Y')
pre_covid = df[df["interview_date.2_year_follow_up_y_arm_1"] < '2020-3-1']
siemens = pre_covid[pre_covid['mri_info_manufacturer.baseline_year_1_arm_1'] == 'SIEMENS']

sample_size = pd.DataFrame(
    columns=[
        'keep', 
        'drop'
    ],
    index=[
        'ABCD Study',
        'Pre-COVID',
        'SIEMENS',
        'sMRI QC 1',
        'dMRI QC 1',
        'fMRI QC 1',
        'sMRI QC 2',
        'dMRI QC 2',
        'fMRI QC 2',
    ]
)

ppts = pd.DataFrame(
    index=df.index,
    columns=[
        'ABCD Study',
        'Pre-COVID',
        'SIEMENS',
        'sMRI QC',
        'dMRI QC',
        'fMRI QC',
    ]
)

sample_size.at['ABCD Study', 'keep'] = len(df.index)
sample_size.at['ABCD Study', 'drop'] = 0
sample_size.at['Pre-COVID', 'keep'] = len(pre_covid.index)
sample_size.at['Pre-COVID', 'drop'] = len(df.index) - len(pre_covid.index)
sample_size.at['SIEMENS', 'keep'] = len(siemens.index)
sample_size.at['SIEMENS', 'drop'] = len(pre_covid.index) - len(siemens.index)

# imaging quality control at baselien
smri_include = abcdw.smri_qc(siemens.filter(like='baseline'))
dmri_include = abcdw.dmri_qc(siemens.filter(like='baseline'), motion_thresh=2.)
fmri_include = abcdw.fmri_qc(siemens.filter(like='baseline'), ntpoints=500, motion_thresh=0.5)

sample_size.at['sMRI QC 1', 'keep'] = len(smri_include)
sample_size.at['sMRI QC 1', 'drop'] = len(siemens.index) - len(smri_include)
sample_size.at['dMRI QC 1', 'keep'] = len(dmri_include)
sample_size.at['dMRI QC 1', 'drop'] = len(siemens.index) - len(dmri_include)
sample_size.at['fMRI QC 1', 'keep'] = len(fmri_include)
sample_size.at['fMRI QC 1', 'drop'] = len(siemens.index) - len(fmri_include)

smri_include2 = abcdw.smri_qc(siemens.filter(like='2_year_follow_up_y_arm_1'))
dmri_include2 = abcdw.dmri_qc(siemens.filter(like='2_year_follow_up_y_arm_1'), motion_thresh=2.)
fmri_include2 = abcdw.fmri_qc(siemens.filter(like='2_year_follow_up_y_arm_1'), ntpoints=500, motion_thresh=0.5)

smri_ppts = list(set(smri_include) & set(smri_include2))
dmri_ppts = list(set(dmri_include) & set(dmri_include2))
fmri_ppts = list(set(fmri_include) & set(fmri_include2))

sample_size.at['sMRI QC 2', 'keep'] = len(smri_ppts)
sample_size.at['sMRI QC 2', 'drop'] = len(smri_include) - len(smri_ppts)
sample_size.at['dMRI QC 2', 'keep'] = len(dmri_ppts)
sample_size.at['dMRI QC 2', 'drop'] = len(dmri_include) - len(dmri_ppts)
sample_size.at['fMRI QC 2', 'keep'] = len(fmri_ppts)
sample_size.at['fMRI QC 2', 'drop'] = len(fmri_include) - len(fmri_ppts)

sample_size.to_csv(join(PROJ_DIR, OUTP_DIR, 'sample_size_qc.csv'))

for ppt in df.index:
    if ppt in pre_covid.index:
        ppts.at[ppt,'Pre-COVID'] = 1
    if ppt in siemens.index:
        ppts.at[ppt,'Pre-COVID'] = 1
    if ppt in smri_include:
        ppts.at[ppt,'sMRI QC'] = 1
    if ppt in dmri_include:
        ppts.at[ppt,'dMRI QC'] = 1
    if ppt in fmri_include:
        ppts.at[ppt,'fMRI QC'] = 1

ppts.to_csv(join(PROJ_DIR, OUTP_DIR, 'ppts_qc.csv'))

covariates = {
    'thk': {
        'vlike': "smri_thick_cdk_",
        'covar': []
    },
    'rnd': {
        'vlike': "dmri_rsirndgm_cdk_",
        'covar': ['dmri_meanmotion', 
                  #'dmri_rsi_meanrot', 
                  #'dmri_rsi_meantrans',
                  ]
    },
    'rni': {
        'vlike': "dmri_rsirnigm_cdk_",
        'covar': ['dmri_meanmotion', 
                  #'dmri_rsi_meanrot', 
                  #'dmri_rsi_meantrans',
                  ]
    },
    'var': {
        'vlike': "rsfmri_var_cdk_",
        'covar': ['rsfmri_meanmotion', 
                  #'rsfmri_var_meanrot', 
                  #'rsfmri_var_meantrans'
                  ]
    }
}

imaging_df = pd.concat(
    [
        df.filter(like='smri_').loc[smri_ppts],
        df.filter(like='dmri_').loc[dmri_ppts],
        df.filter(like='rsfmri_').loc[fmri_ppts]
    ],
    axis=1
)

all_ppts = imaging_df.index
keep = ['mri_info_deviceserialnumber.baseline_year_1_arm_1']
keep += list(df.drop(df.filter(like='mri').columns, axis=1).columns)

qcd_df = pd.concat(
    [
        imaging_df, 
        df.loc[all_ppts][keep]
    ],
    axis=1
)

qcd_df.drop('ehi_y_ss_scoreb.2_year_follow_up_y_arm_1', axis=1).to_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd.pkl'))

#siemens['demo_sex_v2'] = pd.get_dummies(df['demo_sex_v2'])['F']
scanners = pd.get_dummies(qcd_df['mri_info_deviceserialnumber.baseline_year_1_arm_1'])

qcd_df = pd.concat([qcd_df, scanners], axis=1)

all_covars = list(scanners.columns) + ['ehi_y_ss_scoreb.baseline_year_1_arm_1']

#halp
tpts = [
    'baseline_year_1_arm_1',
    '2_year_follow_up_y_arm_1'
]

change_scores = pd.DataFrame(dtype=float)
rci = pd.DataFrame(dtype=float)
for measure in list(covariates.keys()):
    print(measure)
    #motion_df = covariates[measure]['cv_df']
    like = covariates[measure]['vlike']
    covar = covariates[measure]['covar']
    
    temp1 = qcd_df.filter(regex=f'{like}.*baseline_year_1_arm_1')
    temp1.columns = [i.split('.')[0] for i in temp1.columns]
    #print(temp1.head())
    cov_base = [f'{i}.baseline_year_1_arm_1' for i in covar]
    confounds = cov_base + all_covars
    temp2 = qcd_df[confounds]
    #temp2.columns = [i.split('.')[0] for i in temp2.columns]
    #temp2.columns = [f'{col}.baseline_year_1_arm_1' for col in temp2.columns]
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    base_resid = residualize(temp.drop(confounds, axis=1).values, confounds=temp[confounds].values)
    base_resid = pd.DataFrame(data=base_resid, index=temp.index, columns=temp.drop(confounds, axis=1).columns)
    
    # now repeat for year-2 follow-up
    temp1 = qcd_df.filter(regex=f'{like}.*2_year_follow_up_y_arm_1')
    temp1.columns = [i.split('.')[0] for i in temp1.columns]
    cov_y2fu = [f'{i}.2_year_follow_up_y_arm_1' for i in covar]
    confounds = cov_y2fu + all_covars
    temp2 = qcd_df[confounds]
    #temp2.columns = [i.split('.')[0] for i in temp2.columns]
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    y2fu_resid = residualize(temp.drop(confounds, axis=1).values, confounds=temp[confounds].values)
    y2fu_resid = pd.DataFrame(data=y2fu_resid, index=temp.index, columns=temp.drop(confounds, axis=1).columns)
    
    #print(base_resid.head(), y2fu_resid.head())
    # calculate change scores
    for col in base_resid.columns:
        age1 = qcd_df['interview_age.baseline_year_1_arm_1'] / 12
        age2 = qcd_df['interview_age.2_year_follow_up_y_arm_1'] / 12
        temp = pd.concat(
            [base_resid[col], y2fu_resid[col], age1, age2], 
            axis=1
        ).dropna()
        temp2 = pd.concat(
            [base_resid[col], y2fu_resid[col]], 
            axis=1
        ).dropna()
        #print(temp.head())
        for i in temp.index:
            # annual percent change
            base = base_resid.loc[i][col]
            y2fu = y2fu_resid.loc[i][col]
            age0y = age1.loc[i]
            age2y = age2.loc[i]
            change_scores.at[i, col] = (((y2fu - base) / np.mean([y2fu, base])) * 100) / (age2y - age0y)
            # and rci
            s0 = temp2.T.iloc[0].std()
            s2 = temp2.T.iloc[1].std()
            r = np.corrcoef(temp2.T.iloc[0],temp2.T.iloc[1])[0,1]
            sem = np.sqrt(((s0 * np.sqrt(1 - r)) ** 2) + ((s2 * np.sqrt(1 - r)) ** 2))
            #abs_sem = np.std(np.abs(temp.values), ddof=1) / np.sqrt(np.size(temp.values))
            
            rci.at[i,col] = (y2fu - base) / sem / (age2y - age0y)
change_scores = pd.concat(
    [
        change_scores, 
        qcd_df['interview_age.baseline_year_1_arm_1'],
        qcd_df['interview_age.2_year_follow_up_y_arm_1']
    ],
    axis=1
)
change_scores.to_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_change_scores.pkl'))

rci = pd.concat(
    [
        rci, 
        qcd_df['interview_age.baseline_year_1_arm_1'],
        qcd_df['interview_age.2_year_follow_up_y_arm_1']
    ],
    axis=1
)
rci.to_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_rci.pkl'))