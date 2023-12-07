import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt

import pyreadr

from os.path import join
from scipy.stats import spearmanr, fligner, variation
from sklearn.ensemble import IsolationForest


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

# set the plotting settings so our graphs are pretty
sns.set(context='talk', style='white', palette='Set2')

thk_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.pkl'))
var_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.pkl'))
rni_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.pkl'))
rnd_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.pkl'))

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'residualized_change_scores.pkl'))
# ONLY SIEMENS #
# need to load in 3.0 vars that include head motion and brain volume

smri_raw = df.filter(regex='smri_thick_cdk.*')
smri_raw *= -1

rni_raw = df.filter(regex='dmri_rsirnigm_cdk.*')

rnd_raw = df.filter(regex='dmri_rsirndgm_cdk.*')
rnd_raw *= -1

var_raw = df.filter(regex='rsfmri_var_cdk.*')

# redo SA rank by hemisphere
thk_df['hemi'] = [i.split('_')[-1][-2:] for i in thk_df.index]
rni_df['hemi'] = [i.split('_')[-1][-2:] for i in rni_df.index]
rnd_df['hemi'] = [i.split('_')[-1][-2:] for i in rnd_df.index]
var_df['hemi'] = [i.split('_')[-1][-2:] for i in var_df.index]

left_sorted = thk_df[thk_df['hemi'] == 'lh'].sort_values('SA_avg')
right_sorted = thk_df[thk_df['hemi'] == 'rh'].sort_values('SA_avg')
thk_df = pd.concat([left_sorted, right_sorted])
#print(thk_df['SA_avg'])

left_sorted = rni_df[rni_df['hemi'] == 'lh'].sort_values('SA_avg')
right_sorted = rni_df[rni_df['hemi'] == 'rh'].sort_values('SA_avg')
rni_df = pd.concat([left_sorted, right_sorted])

left_sorted = rnd_df[rnd_df['hemi'] == 'lh'].sort_values('SA_avg')
right_sorted = rnd_df[rnd_df['hemi'] == 'rh'].sort_values('SA_avg')
rnd_df = pd.concat([left_sorted, right_sorted])
#print(rnd_df['SA_avg'])


left_sorted = var_df[var_df['hemi'] == 'lh'].sort_values('SA_avg')
right_sorted = var_df[var_df['hemi'] == 'rh'].sort_values('SA_avg')
var_df = pd.concat([left_sorted, right_sorted])

smri_long = smri_raw.melt()
smri_long.index = smri_long['variable']

rni_long = rni_raw.melt()
rni_long.index = rni_long['variable']

rnd_long = rnd_raw.melt()
rnd_long.index = rnd_long['variable']

var_long = var_raw.melt()
var_long.index = var_long['variable']

big_df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd.pkl'))
big_df = big_df.drop(big_df.filter(like='mri'), axis=1)

f_ppts = list(set(big_df[big_df['sex.baseline_year_1_arm_1'] == 'F'].index) & set(df.index))
m_ppts = list(set(big_df[big_df['sex.baseline_year_1_arm_1'] == 'M'].index) & set(df.index))

for i in smri_long.index.unique():
    smri_long.at[i,'SA_avg'] = thk_df.loc[i]['SA_avg']
    smri_long.at[i,'hemisphere'] = thk_df.loc[i]['hemi']
    
for i in rni_long.index.unique():
    rni_long.at[i,'SA_avg'] = rni_df.loc[i]['SA_avg']
    rni_long.at[i,'hemisphere'] = rni_df.loc[i]['hemi']
    
for i in rnd_long.index.unique():
    rnd_long.at[i,'SA_avg'] = rnd_df.loc[i]['SA_avg']
    rnd_long.at[i,'hemisphere'] = rnd_df.loc[i]['hemi']
    
for i in var_long.index.unique():
    var_long.at[i,'SA_avg'] = var_df.loc[i]['SA_avg']
    var_long.at[i,'hemisphere'] = var_df.loc[i]['hemi']

the_data = {
    'thk': {
        'name': 'cortical thickness',
        'vlike': 'smri_thick_cdk',
        'long_df': smri_long,
        'wide_df': smri_raw,
        'sa_df': thk_df,
        'order': 3,
    },
    'rnd': {
        'name': 'directional intracellular diffusion',
        'vlike': 'dmri_rsirndgm_cdk',
        'long_df': rnd_long,
        'wide_df': rnd_raw,
        'sa_df': rnd_df,
        'order': 3,
    },
    'rni': {
        'name': 'isotropic intracellular diffusion',
        'vlike': 'dmri_rsirnigm_cdk',
        'long_df': rni_long,
        'wide_df': rni_raw,
        'sa_df': rni_df,
        'order': 3,
    },
    'var': {
        'name': 'BOLD variance',
        'vlike': 'rsfmri_var_cdk',
        'long_df': var_long,
        'wide_df': var_raw,
        'sa_df': var_df,
        'order': 3,
    }
}

fk_stats = pd.DataFrame(
    index=the_data.keys()
)

##########################################################################
### Exploratory analyses 1 & 2: quantile-wise alignment & hsk therein ####
##########################################################################

ile = 10
    
range_ = range(1,ile + 1)
stats = ['r', 'p(r)']
cols = pd.MultiIndex.from_product([list(range_), stats])
index = pd.MultiIndex.from_product([list(the_data.keys()), list(df.index)])

corr_df = pd.DataFrame(
    index=index,
    columns=cols
)
quantiles = {}

for meas in the_data.keys():
    k = 0.
    dat = the_data[meas]['wide_df']
    sa = the_data[meas]['sa_df']

    quantile = {
    }

    for i in range_:
        l = np.quantile(sa['SA_avg'], i/ile)
        ids = list(sa[sa['SA_avg'].between(k,l, inclusive='right')].index)
        #print(i, '\n', [m.split('_')[-1] for m in ids], '\n\n')
        quantile[i] = ids
        fk_stats.at[meas, f'var_q{i}'] = dat[quantile[i]].melt()['value'].var()
        fk_stats.at[meas, f'cv_q{i}'] = variation(dat[quantile[i]].dropna().melt()['value'])
        for j in dat.index:
            r,p = spearmanr(dat.loc[j][ids], sa.loc[ids]['SA_avg'])
            corr_df.at[(meas, j),('r', i)] = r
            corr_df.at[(meas, j),('p(r)', i)] = p
        k = l
    quantiles[meas] = quantile
    #fk, p = fligner(
    #    dat[quantile[1]].dropna().melt()['value'], 
    #    dat[quantile[2]].dropna().melt()['value'], 
    #    dat[quantile[3]].dropna().melt()['value'], 
    #    dat[quantile[4]].dropna().melt()['value']
    #)
    #fk_stats.at[meas,'fk'] = fk
    #fk_stats.at[meas,'p(fk)'] = p


fig,ax = plt.subplots(nrows=4, ncols=2, sharex='col', figsize=(10,20), layout='constrained')
for meas in the_data.keys():
    i = list(the_data.keys()).index(meas)
    temp = the_data[meas]
    temp_df = corr_df.loc[meas]
    temp_long = temp_df['r'].melt()
    g = sns.kdeplot(data=temp_long, x='value', hue='variable', ax=ax[i,0])
    corrs = temp_df.corr(numeric_only=True).replace({1.0:-0.0})
    drop = corrs.filter(like='p').columns
    corrs = corrs.drop(drop, axis=1).drop(drop,axis=0)
    h = sns.heatmap(corrs, cmap='seismic', center=0, square=True, annot=False, ax=ax[i,1])
    g.set_title(temp['name'])
    h.set_xlabel('')
    h.set_ylabel('')
plt.yticks(rotation=0)
fig.savefig(join(PROJ_DIR, FIGS_DIR, f'SA_axis-quantiles-raw_avg.png'), dpi=400, bbox_inches='tight')

##########################################################################
### Not an exploratory analysis, just plotting average change over SA ####
##########################################################################

for meas in the_data.keys():
    temp = the_data[meas]
    sa_df = temp['sa_df']
    wide = temp['wide_df']
    long = temp['long_df']
    order = temp['order']
    help_ = pd.concat(
        [
            sa_df[['SA_avg', 'hemi']],
            wide.mean()
        ],
        axis=1
    )
    
    help_.columns = ['SA_avg', 'hemisphere', 'delta_thickness']

    g = sns.lmplot(
        data=help_,
        x='SA_avg',
        y='delta_thickness', 
        height=6,
        lowess=True,
        #order=2,
        hue='hemisphere'
    )
    g.ax.set_ylabel(temp['name'])
    g.ax.set_xlabel('Sensorimotor-association axis rank')
    g.savefig(join(PROJ_DIR, FIGS_DIR, f'SA_axis-{meas}_lowess-raw_avg.png'), dpi=400, bbox_inches='tight')


    z = np.polyfit(
        x=long.dropna()['SA_avg'], 
        y=long.dropna()['value'], 
        deg=order
        )
    p = np.poly1d(z)

    fig,ax = plt.subplots(figsize=(8,8))
    h = sns.pointplot(
        data=long.dropna(), 
        y='value',
        x='SA_avg',
        #y='delta_thickness',
        hue='hemisphere',
        #alpha=0.01,
        #ax=g.axes[0][0],
        errorbar=('ci',90),
        linestyles=' ',
        errwidth=0.5,
        ax=ax
    )
    #plt.sca(ax)
    #plt.plot(
    #    long.dropna()['SA_avg'].sort_values(), 
    #    p(long.dropna()['SA_avg'].sort_values()), 
    #    'k--', 
    #    label='polynomial',
    #)

    handles, labels = ax.get_legend_handles_labels()
    h.legend(
        bbox_to_anchor=(1,0.5), 
        handles=handles, 
        labels=['Left', 'Right', #f'{order}-order fit'
                ], 
        title='Hemisphere'
    )
    upper_max = int(long['SA_avg'].max() // 10)
    #upper_max = 35
    h.set_ylim(-50,50)
    #h.set_xticks(range(0, upper_max, upper_max // 5))
    #h.set_xticklabels(list(range(0,upper_max,upper_max // 5)))
    h.set_xlabel('Sensorimotor-association axis rank')
    h.set_ylabel(f"Change in {temp['name']} (%)")
    #ax.axhline(0, color='#333333', linestyle='dotted', alpha=0.2)
    sns.despine()
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'SA_axis-{meas}-raw_avg.png'), dpi=400, bbox_inches='tight')

smri_long.to_csv(join(PROJ_DIR, OUTP_DIR, 'SA_axis-thick_long-raw_avgs.csv'))
rnd_long.to_csv(join(PROJ_DIR, OUTP_DIR, 'SA_axis-rnd_long-raw_avgs.csv'))
rni_long.to_csv(join(PROJ_DIR, OUTP_DIR, 'SA_axis-rni_long-raw_avgs.csv'))
var_long.to_csv(join(PROJ_DIR, OUTP_DIR, 'SA_axis-var_long-raw_avgs.csv'))

corr_df.to_csv(join(PROJ_DIR, OUTP_DIR, 'SA_axis-quartile_correlations-raw_avg.csv'))
fk_stats.to_csv(join(PROJ_DIR, OUTP_DIR, 'SA_axis-quartile_heteroscedasticity-raw_avg.csv'))

rci = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_rci.pkl'))


change_over_time = pd.DataFrame(
    dtype=float,
    index=rci.index,
    columns=rci.columns
)

ppts = list(set(rci.index) & set(big_df.index))
for i in ppts:
    time_diff = (big_df.loc[i]['interview_age.2_year_follow_up_y_arm_1'] - big_df.loc[i]['interview_age.baseline_year_1_arm_1']) / 12
    for j in rci.columns:
       change_over_time.at[i,j] =  rci.loc[i][j] / time_diff 


rci_over_time_sa = pd.DataFrame(
    dtype=float,
    index=rci.index,
    columns=pd.MultiIndex.from_product([list(the_data.keys()), ['r', 'p']])
)

for meas in the_data.keys():
    temp = change_over_time.filter(like=the_data[meas]['vlike'], axis=1)
    sa = the_data[meas]['sa_df']['SA_avg']
    for i in temp.index:
        temp2 = pd.concat([temp.loc[i], sa], axis=1)
        cols = temp2.columns
        corr,p = spearmanr(temp2[cols[0]], temp2[cols[1]])
        rci_over_time_sa.at[i, (meas, 'r')] = corr
        rci_over_time_sa.at[i, (meas, 'p')] = p

rci_over_time_sa.to_pickle(join(PROJ_DIR, OUTP_DIR, 'corr_rci_over_time_sa.pkl'))


age_fxs = pd.DataFrame(
    dtype=float,
    columns=['all', 'M', 'F']
)
sa_breakdown = ['all'] + [f'q{i}' for i in range_]
sexes = ['all', 'F', 'M']

index = pd.MultiIndex.from_product([sa_breakdown, sexes])
age_x_sa_q = pd.DataFrame(
    dtype=float,
    index=index,
    columns=list(the_data.keys())
)

rci_sa_corrs = pd.DataFrame(
    dtype=float,
    index=rci.index,
    columns=list(the_data.keys())
)

for measure in the_data.keys():
    temp_dict = the_data[measure]
    temp_quantiles = quantiles[measure]
    temp_df = change_over_time.filter(like=temp_dict['vlike'], axis=1)

    sa = temp_dict['sa_df']['SA_avg']
    
    mean_age = rci.filter(like='interview_age', axis=1).T.mean()
    mean_age.name = 'age'
    for i in temp_df.index:
        temp = pd.concat([temp_df.loc[i], sa], axis=1)
        
        rci_sa_corrs.at[i,measure] = temp.corr(method='spearman').loc[i]['SA_avg']
    for col in temp_df.columns:
        temp = pd.concat([mean_age, temp_df[col]], axis=1)

        age_fxs.at[col,'all'] = temp.corr(method='spearman', numeric_only=True).loc[col]['age']
        age_fxs.at[col,'F'] = temp.loc[f_ppts].corr(method='spearman', numeric_only=True).loc[col]['age']
        age_fxs.at[col,'M'] = temp.loc[m_ppts].corr(method='spearman', numeric_only=True).loc[col]['age']

    age_sa = pd.concat(
        [
            age_fxs.filter(like=temp_dict['vlike'], axis=0),
            sa
        ],
        axis=1
    )
    long_age = age_sa.melt(id_vars='SA_avg', var_name='ppts', value_name='age_corr')
    g = sns.lmplot(long_age, x='SA_avg', y='age_corr', hue='ppts', order=2, aspect=1.2)
    g.facet_axis(0,0).set_ylabel(f'RCI/time x age correlation: {temp_dict["name"]}')
    g.savefig(
        join(
            PROJ_DIR, FIGS_DIR, f'rci_over_time_x_age-{measure}-correlation-SA_avg-cubic.png'
            ),
            dpi=400, bbox_inches='tight'
        )
    age_x_sa_q.at[('all', 'all'), measure] = age_sa.corr(method='spearman').loc['all']['SA_avg']
    age_x_sa_q.at[('all', 'F'), measure] = age_sa.corr(method='spearman').loc['F']['SA_avg']
    age_x_sa_q.at[('all', 'M'), measure] = age_sa.corr(method='spearman').loc['M']['SA_avg']
    for quantile in temp_quantiles.keys():
        regions = temp_quantiles[quantile]
        q_df = age_sa.loc[regions]
        age_x_sa_q.at[(f'q{quantile}', 'all'), measure] = q_df.corr(method='spearman').loc['all']['SA_avg']
        age_x_sa_q.at[(f'q{quantile}', 'F'), measure] = q_df.corr(method='spearman').loc['F']['SA_avg']
        age_x_sa_q.at[(f'q{quantile}', 'M'), measure] = q_df.corr(method='spearman').loc['M']['SA_avg']
age_x_sa_q.to_csv(join(PROJ_DIR, OUTP_DIR, 'rci_over_time_x_age-correlations-SA_avg.csv'))
rci_sa_corrs.to_pickle(join(PROJ_DIR, OUTP_DIR, 'rci_over_time_x_age-correlations.pkl'))
age_fxs.to_pickle(join(PROJ_DIR, OUTP_DIR, 'rci_over_time_x_age-correlations.pkl'))

# age effects for apd
apd_age_fxs = pd.DataFrame(
    dtype=float,
    columns=['all', 'M', 'F']
)
for measure in the_data.keys():
    temp_dict = the_data[measure]
    temp_quantiles = quantiles[measure]
    temp_df = temp_dict['wide_df']
    sa = temp_dict['sa_df']['SA_avg']
    
    mean_age = rci.filter(like='interview_age', axis=1).T.mean()
    mean_age.name = 'age'
    for i in temp_df.index:
        temp2 = pd.concat([temp_df.loc[i], sa], axis=1)
        
        rci_sa_corrs.at[i,measure] = temp2.corr(method='spearman').loc[i]['SA_avg']
    for col in temp_df.columns:
        temp2 = pd.concat([mean_age, temp_df[col]], axis=1)

        apd_age_fxs.at[col,'all'] = temp2.corr(method='spearman', numeric_only=True).loc[col]['age']
        apd_age_fxs.at[col,'F'] = temp2.loc[f_ppts].corr(method='spearman', numeric_only=True).loc[col]['age']
        apd_age_fxs.at[col,'M'] = temp2.loc[m_ppts].corr(method='spearman', numeric_only=True).loc[col]['age']

    age_sa = pd.concat(
        [
            apd_age_fxs.filter(like=temp_dict['vlike'], axis=0),
            sa
        ],
        axis=1
    )
    long_age = age_sa.melt(id_vars='SA_avg', var_name='ppts', value_name='age_corr')
    g = sns.lmplot(long_age, x='SA_avg', y='age_corr', hue='ppts', order=3, aspect=1.2)
    g.facet_axis(0,0).set_ylabel(f'APD x age correlation: {temp_dict["name"]}')
    g.savefig(
        join(
            PROJ_DIR, FIGS_DIR, f'apd_x_age-{measure}-correlation-SA_avg-cubic.png'
            ),
            dpi=400, bbox_inches='tight'
        )
    
    h = sns.lmplot(long_age, x='SA_avg', y='age_corr', hue='ppts', lowess=True, aspect=1.2)
    h.facet_axis(0,0).set_ylabel(f'APD x age correlation: {temp_dict["name"]}')
    h.savefig(
        join(
            PROJ_DIR, FIGS_DIR, f'apd_x_age-{measure}-correlation-SA_avg-lowess.png'
            ),
            dpi=400, bbox_inches='tight'
        )
    age_x_sa_q.at[('all','all'), measure] = age_sa.corr(method='spearman').loc['all']['SA_avg']
    age_x_sa_q.at[('all', 'F'), measure] = age_sa.corr(method='spearman').loc['F']['SA_avg']
    age_x_sa_q.at[('all', 'M'), measure] = age_sa.corr(method='spearman').loc['M']['SA_avg']
    for quantile in temp_quantiles.keys():
        regions = temp_quantiles[quantile]
        q_df = age_sa.loc[regions]
        age_x_sa_q.at[(f'q{quantile}', 'all'), measure] = q_df.corr(method='spearman').loc['all']['SA_avg']
        age_x_sa_q.at[(f'q{quantile}', 'F'), measure] = q_df.corr(method='spearman').loc['F']['SA_avg']
        age_x_sa_q.at[(f'q{quantile}', 'M'), measure] = q_df.corr(method='spearman').loc['M']['SA_avg']
age_x_sa_q.to_csv(join(PROJ_DIR, OUTP_DIR, 'apd_x_age-correlations-SA_avg.csv'))
age_fxs.to_pickle(join(PROJ_DIR, OUTP_DIR, 'apd_x_age-correlations.pkl'))


all_sa = pd.concat([thk_df, rni_df, rnd_df, var_df], axis=0)['SA_avg']
sex = pd.get_dummies(big_df['sex.baseline_year_1_arm_1'].dropna())['F']
age_fx_sa = pd.concat([age_fxs, all_sa / 1000, sex], axis=1)
age_fx_sa['SA_avg^2'] = age_fx_sa['SA_avg'] ** 2

smri = age_fx_sa.filter(like='smri', axis=0)
smri_reg = pg.linear_regression(X=smri[['SA_avg', 'SA_avg^2']], y=smri['all'], remove_na=True)
smri_reg.index = [f'thk_{i}' for i in smri_reg.index]

rsfmri = age_fx_sa.filter(like='rsfmri', axis=0)
rsfmri_reg = pg.linear_regression(X=rsfmri[['SA_avg', 'SA_avg^2']], y=rsfmri['all'], remove_na=True)
rsfmri_reg.index = [f'var_{i}' for i in rsfmri_reg.index]

rnd = age_fx_sa.filter(like='dmri_rsirnd', axis=0)
rnd_reg = pg.linear_regression(X=rnd[['SA_avg', 'SA_avg^2']], y=rnd['all'], remove_na=True)
rnd_reg.index = [f'rnd_{i}' for i in rnd_reg.index]

rni = age_fx_sa.filter(like='dmri_rsirni', axis=0)
rni_reg = pg.linear_regression(X=rni[['SA_avg', 'SA_avg^2']], y=rni['all'], remove_na=True)
rni_reg.index = [f'rni_{i}' for i in rni_reg.index]

regressions = pd.concat([smri_reg, rsfmri_reg, rni_reg, rnd_reg], axis=0)
regressions.to_csv(join(PROJ_DIR, OUTP_DIR, 'rci_over_time_x_age-sa_axis-regressions.csv'))

# now do it again for apd
age_fx_sa = pd.concat([apd_age_fxs, all_sa / 1000, sex], axis=1)
age_fx_sa['SA_avg^2'] = age_fx_sa['SA_avg'] ** 2

smri = age_fx_sa.filter(like='smri', axis=0)
smri_reg = pg.linear_regression(X=smri[['SA_avg', 'SA_avg^2']], y=smri['all'], remove_na=True)
smri_reg.index = [f'thk_{i}' for i in smri_reg.index]

rsfmri = age_fx_sa.filter(like='rsfmri', axis=0)
rsfmri_reg = pg.linear_regression(X=rsfmri[['SA_avg', 'SA_avg^2']], y=rsfmri['all'], remove_na=True)
rsfmri_reg.index = [f'var_{i}' for i in rsfmri_reg.index]

rnd = age_fx_sa.filter(like='dmri_rsirnd', axis=0)
rnd_reg = pg.linear_regression(X=rnd[['SA_avg', 'SA_avg^2']], y=rnd['all'], remove_na=True)
rnd_reg.index = [f'rnd_{i}' for i in rnd_reg.index]

rni = age_fx_sa.filter(like='dmri_rsirni', axis=0)
rni_reg = pg.linear_regression(X=rni[['SA_avg', 'SA_avg^2']], y=rni['all'], remove_na=True)
rni_reg.index = [f'rni_{i}' for i in rni_reg.index]

regressions = pd.concat([smri_reg, rsfmri_reg, rni_reg, rnd_reg], axis=0)
regressions.to_csv(join(PROJ_DIR, OUTP_DIR, 'apd_x_age-sa_axis-regressions.csv'))