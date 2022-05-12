# %%
import sys
import enlighten
import pandas as pd
from os.path import join, exists, isdir
from os import mkdir

# %%
# assembling a dictionary of data structure shortnames and columns of interest
variables = {
    "abcd_smrip10201": [
        "interview_age",
        "smri_vol_cdk_total",
        "smri_thick_cdk_mean",
        "smri_area_cdk_total",
    ],
    "abcd_dmdtifp101": ["dmdtifp1_38",
                        "dmdtifp1_80",
                        "dmdtifp1_122", 
                        "dmdtifp1_164",
                        "dmdtifp1_206",],
    "abcd_drsip101": ["dmri_rsirni_fib_allfib", ],
    "abcd_drsip201": ["dmri_rsirnd_fib_allfib",],
    "abcd_drsip301": ["dmri_rsirnt_fib_allfib",],
    "abcd_dti_p101": ["dmri_dti_meanmotion", 
                      "dmri_dtifa_fiberat_allfibers", 
                      "dmri_dtimd_fiberat_allfibers", 
                      "dmri_dtild_fiberat_allfibers", 
                      "dmri_dtitd_fiberat_allfibers", 
                      "dmri_dtivol_fiberat_allfibers", ],
    "abcd_mrisdp10201": ["mrisdp_151", "mrisdp_302", "mrisdp_453", "mrisdp_604"],
    "abcd_mri01": ["sex",
        "mri_info_manufacturer",
        "interview_date"
    ],
    "abcd_imgincl01": ["imgincl_t1w_include", "imgincl_dmri_include"],
    "abcd_mrfindings02": ["mrif_score"]
}

# %%
print(variables.keys())

hemispheric = {"abcd_dmdtifp101": ["dmdtifp1_39", "dmdtifp1_40", "dmdtifp1_41", "dmdtifp1_42",
                                   "dmdtifp1_81", "dmdtifp1_82", "dmdtifp1_83", "dmdtifp1_84",],
               "abcd_smrip10201": ["interview_age", "smri_vol_cdk_totallh", "smri_vol_cdk_totalrh",
                                   "smri_thick_cdk_meanlh", "smri_thick_cdk_meanrh", 
                                   "smri_area_cdk_totallh", "smri_area_cdk_totalrh"],
                "abcd_drsip101": ["dmri_rsirni_fib_afbncrh", "dmri_rsirni_fib_afbnclh", "dmri_rsirni_fib_allfibrh", "dmri_rsirni_fib_allfiblh"],
                "abcd_drsip201": ["dmri_rsirnd_fib_afbncrh", "dmri_rsirnd_fib_afbnclh", "dmri_rsirnd_fib_allfibrh", "dmri_rsirnd_fib_allfiblh"],
                "abcd_drsip301": ["dmri_rsirnt_fib_afbncrh", "dmri_rsirnt_fib_afbnclh", "dmri_rsirnt_fib_allfibrh", "dmri_rsirnt_fib_allfiblh"],
                "abcd_dti_p101": ["dmri_dti_meanmotion", "dmri_dtifa_fiberat_allfccrh", "dmri_dtifa_fiberat_allfcclh", "dmri_dtifa_fiberat_allfibrh", "dmri_dtifa_fiberat_allfiblh",
                                  "dmri_dtimd_fiberat_allfccrh", "dmri_dtimd_fiberat_allfcclh", "dmri_dtimd_fiberat_allfibrh", "dmri_dtimd_fiberat_allfiblh",
                                  "dmri_dtild_fiberat_allfccrh", "dmri_dtild_fiberat_allfocclh", "dmri_dtild_fiberat_allfibrh", "dmri_dtild_fiberat_allfiblh",
                                  "dmri_dtitd_fiberat_allfccrh", "dmri_dtitd_fiberat_allfcclh", "dmri_dtitd_fiberat_allfibrh", "dmri_dtitd_fiberat_allfiblh",
                                  "dmri_dtivol_fiberat_allfccrh", "dmri_dtivol_fiberat_allfcclh", "dmri_dtivol_fiberat_allfibrh", "dmri_dtivol_fiberat_allfiblh",],
                "abcd_mrisdp10201": ["mrisdp_149", "mrisdp_150", "mrisdp_300", "mrisdp_301", "mrisdp_451", "mrisdp_452", "mrisdp_602", "mrisdp_603", ],
                "abcd_mri01": ["sex",
                               "mri_info_manufacturer",
                               "interview_date"
                               ],
                "abcd_imgincl01": ["imgincl_t1w_include", "imgincl_dmri_include"],
                "abcd_mrfindings02": ["mrif_score"]}


# %%
# read in csvs of interest one a time so you don't crash your computer
# grab the vars you want, then clear the rest and read in the next
# make one "missing" column for each modality if, like RSI, a subj is missing
# on all vals if missing on one. double check this.
# also include qa column per modality and make missingness chart before/after data censoring

data_dir = (
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/"
)

OUT_DIR = "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/WholeBrainGMWM/"

if not isdir(OUT_DIR):
    mkdir(OUT_DIR)

# %%
# column names to retain in the mega dataset
qc_vars = [
    "imgincl_dmri_include",
    "imgincl_t1w_include",
    "imgincl_t2w_include",
    "interview_age",
]

long = True
# %%
# initialize the progress bars
manager = enlighten.get_manager()
tocks = manager.counter(total=len(variables.keys()), desc='Data Structures', unit='data structures')

# keep track of variables that don't make it into the big df
missing = {}
data_df = pd.read_csv(join(data_dir, 'generate_dataset/data_element_names.csv'), index_col=0)

# build the mega_df now
df = pd.DataFrame()
data_dict = pd.DataFrame(columns=['data_structure', 'variable_description'])
for structure in variables.keys():
    missing[structure] = []
    old_columns = len(df.columns)
    path = join(data_dir, 'csv', f'{structure}.csv')
    if exists(path):
        ticks = manager.counter(total=len(variables[structure]), desc=structure, unit='variables')
        

        # original ABCD data structures are in long form, with eventname as a column
        # but I want the data in wide form, only one row per participant
        # and separate columns for values collected at different timepoints/events
        if long == True:
            index = ["subjectkey","eventname" ]
            cols = variables[structure]
            temp_df = pd.read_csv(path, 
                                  index_col=index, 
                                  header=0, 
                                  skiprows=[1], 
                                  usecols= index + cols)
            df = pd.concat([df, temp_df], axis=1)
            for variable in variables[structure]:
                try:
                    data_dict.at[variable, 'data_structure'] = structure
                    data_dict.at[variable, 'variable_description'] = data_df.loc[variable, 'description']
                except Exception as e:
                    print(e)
            ticks.update()
            
        else:
            temp_df = pd.read_csv(path, index_col="subjectkey", header=0, skiprows=[1])
            base_df = temp_df[temp_df["eventname"] == "baseline_year_1_arm_1"].copy()
            y2fu_df = temp_df[temp_df["eventname"] == "2_year_follow_up_y_arm_1"].copy()
            temp_df = None
            for variable in variables[structure]:
                try:
                    df[f'{variable}.baseline_year_1_arm_1'] = base_df[variable].copy()
                    df[f'{variable}.2_year_follow_up_y_arm_1'] = y2fu_df[variable].copy()
                    data_dict.at[variable, 'data_structure'] = structure
                    data_dict.at[variable, 'variable_description'] = data_df.loc[variable, 'description']
                except Exception as e:
                    print(e)
                ticks.update()
        ticks.refresh()
    new_columns = len(df.columns) - old_columns
    print(f"\t{new_columns} variables added!")
    if len(missing[structure]) >= 1:
        print(f"The following {len(missing[structure])}variables could not be added:\n{missing[structure]}")
    else:
        print(f"All variables were successfully added from {structure}.")
    temp_df = None
    base_df = None
    y2fu_df = None
    tocks.update()


# how big is this thing?
print(f"Full dataframe is {sys.getsizeof(df) / 1000000000}GB.")

df = df.dropna(how="all", axis=0)
# write out resulting dataframe
df.to_csv(join(OUT_DIR, 'whole_brain-data.csv')
)

data_dict.to_csv(join(OUT_DIR, 'whole_brain-data_dictionary.csv'))
print('data and data dictionary have been saved to', OUT_DIR)

df = pd.DataFrame()
data_dict = pd.DataFrame(columns=['data_structure', 'variable_description'])
for structure in hemispheric.keys():
    missing[structure] = []
    old_columns = len(df.columns)
    path = join(data_dir, 'csv', f'{structure}.csv')
    if exists(path):
        ticks = manager.counter(total=len(hemispheric[structure]), desc=structure, unit='variables')
        

        # original ABCD data structures are in long form, with eventname as a column
        # but I want the data in wide form, only one row per participant
        # and separate columns for values collected at different timepoints/events
        if long == True:
            index = ["subjectkey","eventname" ]
            cols = hemispheric[structure]
            temp_df = pd.read_csv(path, 
                                  index_col=index, 
                                  header=0, 
                                  skiprows=[1], 
                                  usecols= index + cols)
            df = pd.concat([df, temp_df], axis=1)
            for variable in hemispheric[structure]:
                try:
                    data_dict.at[variable, 'data_structure'] = structure
                    data_dict.at[variable, 'variable_description'] = data_df.loc[variable, 'description']
                except Exception as e:
                    print(e)
            ticks.update()
            
        else:
            temp_df = pd.read_csv(path, index_col="subjectkey", header=0, skiprows=[1])
            base_df = temp_df[temp_df["eventname"] == "baseline_year_1_arm_1"].copy()
            y2fu_df = temp_df[temp_df["eventname"] == "2_year_follow_up_y_arm_1"].copy()
            temp_df = None
            for variable in variables[structure]:
                try:
                    df[f'{variable}.baseline_year_1_arm_1'] = base_df[variable].copy()
                    df[f'{variable}.2_year_follow_up_y_arm_1'] = y2fu_df[variable].copy()
                    data_dict.at[variable, 'data_structure'] = structure
                    data_dict.at[variable, 'variable_description'] = data_df.loc[variable, 'description']
                except Exception as e:
                    print(e)
                ticks.update()
        ticks.refresh()
    new_columns = len(df.columns) - old_columns
    print(f"\t{new_columns} variables added!")
    if len(missing[structure]) >= 1:
        print(f"The following {len(missing[structure])}variables could not be added:\n{missing[structure]}")
    else:
        print(f"All variables were successfully added from {structure}.")
    temp_df = None
    base_df = None
    y2fu_df = None
    tocks.update()


df = df.dropna(how="all", axis=0)
# write out resulting dataframe
df.to_csv(join(OUT_DIR, 'hemispheres-data.csv')
)

data_dict.to_csv(join(OUT_DIR, 'hemispheres-data_dictionary.csv'))
print('data and data dictionary have been saved to', OUT_DIR)