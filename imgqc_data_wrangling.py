# %%
import sys
import enlighten
import pandas as pd
from os.path import join, exists, isdir
from os import mkdir

# %%
# assembling a dictionary of data structure shortnames and columns of interest
variables = {
    "abcd_imgincl01": ["interview_date",
                       "interview_age",
                       "sex",
                       "eventname",
                       "imgincl_t1w_include",
                       "imgincl_t2w_include",
                       "imgincl_dmri_include",
                       "imgincl_rsfmri_include"],
    "abcd_mrfindings02": ["mrif_score"],
    #"abcd_dti_p101": ["dmri_dti_meanmotion",],
    "abcd_drsip101": ["dmri_rsi_meanmotion"], # should be the same as dti_meanmotion bc it's the same scan
    "abcd_mri01": ["mri_info_manufacturer"],
    "abcd_mrirstv02": ["rsfmri_var_meanmotion",
                       "rsfmri_var_subthreshnvols",
                       "rsfmri_var_subtcignvols",
                       "rsfmri_var_ntpoints"],
}

# prints out the data structures that it'll pull data from
print(variables.keys())

# input directory (where ABCD data is stored)
data_dir = (
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/"
)

# output directory (where everything will be saved)
OUT_DIR = "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/ImagingQC/"

# makes output directory if it doesn't already exist
if not isdir(OUT_DIR):
    mkdir(OUT_DIR)

# should the data be saved in long format (i.e., the way it is on NDA)
# if false, saves data in wide format: one row per NDAR, one column per var (per eventname)
long = True
# %%
# initialize the progress bars
manager = enlighten.get_manager()
tocks = manager.counter(total=len(variables.keys()), desc='Data Structures', unit='data structures')

# keep track of variables that don't make it into the big df
missing = {}

# read in the big ABCD data dictionary
data_df = pd.read_csv(join(data_dir, 'generate_dataset/data_element_names.csv'), index_col=0)

# build the dataframe now
df = pd.DataFrame()

# and build a data dictionary for the included variables
data_dict = pd.DataFrame(columns=['data_structure', 'variable_description'])
for structure in variables.keys():
    missing[structure] = []
    old_columns = len(df.columns)
    path = join(data_dir, 'csv', f'{structure}.csv')
    if exists(path):

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
            #ticks.update()
            
        else:
            # original ABCD data structures are in long form, with eventname as a column
            # but I want the data in wide form, only one row per participant
            # and separate columns for values collected at different timepoints/events
            ticks = manager.counter(total=len(variables[structure]), desc=structure, unit='variables')
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
        #ticks.refresh()
    new_columns = len(df.columns) - old_columns
    print(f"\t{new_columns} variables added!")
    if len(missing[structure]) >= 1:
        print(f"The following {len(missing[structure])}variables could not be added:\n{missing[structure]}")
    else:
        print(f"All variables were successfully added from {structure}.")
    # clears the loaded data structures to save memory
    temp_df = None
    base_df = None
    y2fu_df = None
    
    # updates the progress bar
    tocks.update()


# how big is this thing?
print(f"Full dataframe is {sys.getsizeof(df) / 1000000000}GB.")

df = df.dropna(how="all", axis=0)
# write out resulting dataframe
df.to_csv(join(OUT_DIR, 'qc-data.csv')
)

data_dict.to_csv(join(OUT_DIR, 'qc-data_dictionary.csv'))
print('data and data dictionary have been saved to', OUT_DIR)
