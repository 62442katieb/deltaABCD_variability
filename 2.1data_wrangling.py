# %%
import sys
import enlighten
import pandas as pd
from os.path import join, exists

PROJ_DIR = '/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm'
# assembling a dictionary of data structure shortnames and columns of interest
variables = {
    "abcd_imgincl01": [
        "imgincl_dmri_include",
        "imgincl_rsfmri_include",
        "imgincl_t1w_include",],
    "abcd_smrip10201": [
        "interview_age",
        "smri_vol_cdk_total",
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
    ],
    "abcd_drsip101": [
        "dmri_rsi_meanmotion",
        'dmri_rsi_meantrans',
        'dmri_rsi_meanrot',
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
        #"dmri_rsirnigm_cdk_mean",
    ],
    "abcd_drsip201": [
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
    ],
    "abcd_mrirstv02": [
        "rsfmri_var_meanmotion",
        "rsfmri_var_subthreshnvols",
        "rsfmri_var_subtcignvols",
        "rsfmri_var_ntpoints",
        'rsfmri_var_meantrans',
        'rsfmri_var_meanrot',
        'rsfmri_var_maxtrans',
        'rsfmri_var_maxrot',
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
    ],
    "abcd_tbss01": [
        "nihtbx_picvocab_uncorrected",
        "nihtbx_flanker_uncorrected",
        "nihtbx_list_uncorrected",
        "nihtbx_cardsort_uncorrected",
        "nihtbx_pattern_uncorrected",
        "nihtbx_picture_uncorrected",
        "nihtbx_reading_uncorrected",
    ],
    "abcd_mri01": ["sex",
        "mri_info_manufacturer",
        "interview_date"
    ],
    "abcd_mrfindings02": ["mrif_score"],
    "acspsw03": [
        "rel_family_id", 
        "rel_group_id", 
        "rel_ingroup_order", 
        "rel_relationship",
        "race_ethnicity"
    ],
    "abcd_lt01": ["site_id_l"],
    "pdem02": ["demo_prnt_ethn_v2",
        "demo_prnt_marital_v2",
        "demo_prnt_ed_v2",
        "demo_comb_income_v2",
    ],
    "abcd_ssphp01": ["pds_p_ss_female_category_2", 
        "pds_p_ss_male_category_2"],
    "abcd_cbcls01": ["cbcl_scr_syn_anxdep_r", 
        "cbcl_scr_syn_withdep_r", 
        "cbcl_scr_syn_somatic_r", 
        "cbcl_scr_syn_social_r", 
        "cbcl_scr_syn_thought_r", 
        "cbcl_scr_syn_attention_r", 
        "cbcl_scr_syn_rulebreak_r", 
        "cbcl_scr_syn_aggressive_r", 
        "cbcl_scr_syn_internal_r", 
        "cbcl_scr_syn_external_r", 
        "cbcl_scr_syn_totprob_r"]
}

# %%
#print(variables.keys())

# %%
# read in csvs of interest one a time so you don't crash your computer
# grab the vars you want, then clear the rest and read in the next
# make one "missing" column for each modality if, like RSI, a subj is missing
# on all vals if missing on one. double check this.
# also include qa column per modality and make missingness chart before/after data censoring

data_dir = (
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/"
)

changes = ['abcd_smrip10201', 'abcd_smrip20201', 'abcd_smrip30201', 
           'abcd_mrisdp10201', 'abcd_mrisdp20201', 'abcd_dti_p101', 
           'abcd_drsip101', 'abcd_drsip201', 'abcd_mrirsfd01', 
           'abcd_mrirstv02', 'abcd_betnet02', 'mrirscor02', 'abcd_tbss01']

# %%
# column names to retain in the mega dataset
qc_vars = [
    "imgincl_dmri_include",
    "imgincl_rsfmri_include",
    "rsfmri_var_meanmotion",
    "rsfmri_var_subthreshnvols",
    "rsfmri_var_subtcignvols",
    "rsfmri_var_ntpoints",
    "imgincl_t1w_include",
    "dmri_rsi_meanmotion",
    "interview_age",
]

# read in data dictionary
data_df = pd.read_csv(join(data_dir, 'generate_dataset/data_element_names.csv'), index_col=0, header=0)

# %%
# initialize the progress bars
manager = enlighten.get_manager()
tocks = manager.counter(total=len(variables.keys()), desc='Data Structures', unit='data structures')

# keep track of variables that don't make it into the big df
missing = {}

# build the mega_df now
df = pd.DataFrame()
data_dict = pd.DataFrame(columns=['data_structure', 'variable_description'])
for key in variables.keys():
    missing[key] = []
    old_columns = len(df.columns)
    data_dict
    if key in changes:
        path = join(data_dir, 'change_scores', f'{key}_changescores_bl_tp2.csv')
        if exists(path):
            ticks = manager.counter(total=len(variables[key]), desc=key, unit='variables')
            temp_df = pd.read_csv(path, index_col="subjectkey", header=0)
            for column in variables[key]:
                try:
                    data_dict.at[column, 'data_structure'] = key
                    data_dict.at[column, 'variable_description'] = data_df.loc[column, 'description']
                    # grab baselines and 2yfu values
                    df[f'{column}.baseline_year_1_arm_1'] = temp_df[f'{column}.baseline_year_1_arm_1'].copy()
                    df[f'{column}.2_year_follow_up_y_arm_1'] = temp_df[f'{column}.2_year_follow_up_y_arm_1'].copy()
                    if column not in qc_vars:
                        # and the change score, if it's not a qc variable
                        df[f'{column}.change_score'] = temp_df[f'{column}.change_score'].copy()
                    else:
                        pass
                except:
                    # keep track of missing variables
                    missing[key].append(column)
                ticks.update()
            ticks.refresh()
        else:
            print(f'There\'s something wrong with your path for {key}:\n{path}')
            break
    else:
        path = join(data_dir, 'csv', f'{key}.csv')
        if exists(path):
            ticks = manager.counter(total=len(variables[key]), desc=key, unit='variables')
            temp_df = pd.read_csv(path, index_col="subjectkey", header=0, skiprows=[1])

            # original ABCD data structures are in long form, with eventname as a column
            # but I want the data in wide form, only one row per participant
            # and separate columns for values collected at different timepoints/events
            base_df = temp_df[temp_df["eventname"] == "baseline_year_1_arm_1"].copy()
            y2fu_df = temp_df[temp_df["eventname"] == "2_year_follow_up_y_arm_1"].copy()
            temp_df = None
            for column in variables[key]:
                data_dict.at[column, 'data_structure'] = key
                data_dict.at[column, 'variable_description'] = data_df.loc[column, 'description']
                df[f'{column}.baseline_year_1_arm_1'] = base_df[column].copy()
                df[f'{column}.2_year_follow_up_y_arm_1'] = y2fu_df[column].copy()
                ticks.update()
            ticks.refresh()
    new_columns = len(df.columns) - old_columns
    print(f"\t{new_columns} variables added!")
    if len(missing[key]) >= 1:
        print(f"The following {len(missing[key])}variables could not be added:\n{missing[key]}")
    else:
        print(f"All variables were successfully added from {key}.")
    temp_df = None
    base_df = None
    y2fu_df = None
    tocks.update()


# how big is this thing?
print(f"Full dataframe is {sys.getsizeof(df) / 1000000000}GB.")

# write out resulting dataframe
#df.to_csv(join(PROJ_DIR, 'data', 'data.csv'))
df.to_pickle(join(PROJ_DIR, 'data', 'data.pkl'))

data_dict.to_csv(join(PROJ_DIR, 'data', 'data_dictionary.csv'))
