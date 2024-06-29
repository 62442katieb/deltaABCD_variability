

import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from os.path import join



PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


demo_df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_covar.pkl'))

demo_df = demo_df.replace(
    {
        'Pre Puberty': 1,
        'Early Puberty': 2,
        'Mid Puberty': 3,
        'Late/Post Puberty': 4
    }
)
demo_df = demo_df.rename(
    {
        'PDS.baseline_year_1_arm_1': 'baseline_Puberty'
    },
    axis=1
)

thk = pd.read_pickle(join(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_thk_corrs-rci.pkl'))
rni = pd.read_pickle(join(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_rni_corrs-rci.pkl'))

rnd = pd.read_pickle(join(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_rnd_corrs-rci.pkl'))

var = pd.read_pickle(join(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_var_corrs-rci.pkl'))


thk_df = pd.concat(
    [
        demo_df, 
        thk], 
    axis=1
).dropna()
rni_df = pd.concat(
    [
        demo_df, 
        rni
    ], 
    axis=1
).dropna()

rnd_df = pd.concat(
    [
        demo_df, 
        rnd
    ], 
    axis=1
).dropna()

var_df = pd.concat(
    [
        demo_df, 
        var
    ], 
    axis=1
).dropna()

thk_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'thk_plus_demos.csv'))
rni_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'rni_plus_demos.csv'))
rnd_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'rnd_plus_demos.csv'))
var_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'var_plus_demos.csv'))


thk = pd.read_pickle(join(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_thk_corrs.pkl'))
rni = pd.read_pickle(join(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_rni_corrs.pkl'))

rnd = pd.read_pickle(join(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_rnd_corrs.pkl'))

var = pd.read_pickle(join(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_var_corrs.pkl'))


thk_df = pd.concat(
    [
        demo_df, 
        thk], 
    axis=1
).dropna()
rni_df = pd.concat(
    [
        demo_df, 
        rni], 
    axis=1
).dropna()

rnd_df = pd.concat(
    [
        demo_df, 
        rnd], 
    axis=1
).dropna()

var_df = pd.concat(
    [
        demo_df, 
        var], 
    axis=1
).dropna()


thk_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'thk_plus_demos-apd.csv'))
rni_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'rni_plus_demos-apd.csv'))
rnd_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'rnd_plus_demos-apd.csv'))
var_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'var_plus_demos-apd.csv'))


