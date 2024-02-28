#!/usr/bin/env python
# coding: utf-8
import enlighten
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt

from os.path import join
from scipy.stats import chi2_contingency, fisher_exact

def jili_sidak_mc(data, alpha):
    '''
    Accepts a dataframe (data, samples x features) and a type-i error rate (alpha, float), 
    then adjusts for the number of effective comparisons between variables
    in the dataframe based on the eigenvalues of their pairwise correlations.
    '''
    import math
    import numpy as np

    mc_corrmat = data.corr()
    mc_corrmat.fillna(0, inplace=True)
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print('\nFor {0} vars, number of effective comparisons: {1}\n'.format(mc_corrmat.shape[0], M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

# set the plotting settings so our graphs are pretty
sns.set(context='talk', style='white', palette='Set2')

thk_apd_corrs = pd.read_pickle(
    join(PROJ_DIR, OUTP_DIR, 'sa_thk_corrs.pkl'),
    #index_col=0, header=0
)
var_apd_corrs = pd.read_pickle(
    join(PROJ_DIR, OUTP_DIR, 'sa_var_corrs.pkl'),
    #index_col=0, header=0
)
rni_apd_corrs = pd.read_pickle(
    join(PROJ_DIR, OUTP_DIR, 'sa_rni_corrs.pkl'),
    #index_col=0, header=0
)
rnd_apd_corrs = pd.read_pickle(
    join(PROJ_DIR, OUTP_DIR, 'sa_rnd_corrs.pkl'),
    #index_col=0, header=0
)

cbcl_df = pd.read_pickle(join('/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_vbgmm/', DATA_DIR, "data.pkl"))
cbcl_df = cbcl_df.filter(like='cbcl', axis=1)

r_df = pd.concat(
    [
        thk_apd_corrs['r'].rename('thickness').drop_duplicates(), 
        var_apd_corrs['r'].rename('functional variance'),
        rni_apd_corrs['r'].rename('isotropic diffusion'), 
        rnd_apd_corrs['r'].rename('directional diffusion'), 
        
    ], axis=1
).sort_index()
r_df.columns = ['Cortical thickness', 'Functional variance', 'Isotropic diffusion', 'Directional diffusion']

p_df = pd.concat(
    [
        thk_apd_corrs['p'].rename('thickness').drop_duplicates(),
        var_apd_corrs['p'].rename('functional variance'),
        rni_apd_corrs['p'].rename('isotropic diffusion'), 
        rnd_apd_corrs['p'].rename('directional diffusion'), 
        
    ], axis=1
).sort_index()
p_df.columns = ['Cortical thickness', 'Functional variance', 'Isotropic diffusion', 'Directional diffusion']

#print(spearmanr(r_df.dropna()['functional variance'], r_df.dropna()['thickness']))

fig,ax = plt.subplots(figsize=(10,3))
sns.heatmap(r_df.dropna().sort_values('Cortical thickness').T, ax=ax, cmap='RdBu', center=0)
ax.set_xticklabels('')
#plt.show()
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'SA_corrs_across_measures.png'), dpi=400, bbox_inches='tight')

thk_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.pkl'))
var_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.pkl'))
rni_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.pkl'))
rnd_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.pkl'))
all_sa = pd.concat([thk_df, rni_df, rnd_df, var_df], axis=0)['SA_avg']
thk_df = None
var_df = None
rni_df = None
rnd_df = None

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'residualized_change_scores.pkl'))
rci = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_rci.pkl'))

cbcl_df = cbcl_df.loc[df.index]

# we're changing the CBCL into problems or no problems
cbcl_vars = [
    "cbcl_scr_syn_anxdep_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_withdep_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_somatic_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_social_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_thought_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_attention_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_rulebreak_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_aggressive_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_internal_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_external_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_totprob_r.2_year_follow_up_y_arm_1", 
    "cbcl_scr_syn_anxdep_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_withdep_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_somatic_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_social_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_thought_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_attention_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_rulebreak_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_aggressive_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_internal_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_external_r.baseline_year_1_arm_1", 
    "cbcl_scr_syn_totprob_r.baseline_year_1_arm_1",
]
for var in cbcl_vars:
    temp = cbcl_df[var] > 0
    cbcl_df[f'bin-{var}'] = temp.replace({True: 1, False: 0})
    #print(cbcl_df[f'bin-{var}'].unique())

non_brain_df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, 'data_qcd.pkl'))
non_brain_df = non_brain_df.drop(non_brain_df.filter(like='mri').columns, axis=1)

non_brain_df[['pds_p_ss_female_category_2.baseline_year_1_arm_1',
    'pds_p_ss_male_category_2.baseline_year_1_arm_1']]
non_brain_df['pds_p_ss_category_2.baseline_year_1_arm_1'] = non_brain_df['pds_p_ss_female_category_2.baseline_year_1_arm_1'].fillna(0) + non_brain_df['pds_p_ss_male_category_2.baseline_year_1_arm_1'].fillna(0)
non_brain_df['pds_p_ss_category_2.baseline_year_1_arm_1'].replace({0:np.nan}, inplace=True)

non_brain_df[['pds_p_ss_female_category_2.2_year_follow_up_y_arm_1',
    'pds_p_ss_male_category_2.2_year_follow_up_y_arm_1']]
non_brain_df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'] = non_brain_df['pds_p_ss_female_category_2.2_year_follow_up_y_arm_1'].fillna(0) + non_brain_df['pds_p_ss_male_category_2.2_year_follow_up_y_arm_1'].fillna(0)
non_brain_df['pds_p_ss_category_2.2_year_follow_up_y_arm_1'].replace({0:np.nan}, inplace=True)

non_brain_df = pd.concat([non_brain_df, cbcl_df], axis=1)

rci_over_time_sa = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'corr_rci_over_time_sa.pkl'))

# demographcs
demo_vars = [
    'pds_p_ss_category_2.baseline_year_1_arm_1',
    'race_ethnicity.baseline_year_1_arm_1', 
    'sex.baseline_year_1_arm_1',
    'demo_prnt_ed_v2.baseline_year_1_arm_1',
    'pds_p_ss_category_2.2_year_follow_up_y_arm_1',
    'demo_comb_income_v2.baseline_year_1_arm_1',
]
demos = pd.Series(dtype=float)
ppts = list(set(rci_over_time_sa.index) & set(non_brain_df.index))
for demo_var in demo_vars:
    temp = pd.get_dummies(non_brain_df.loc[ppts][demo_var], prefix=f'{demo_var.split("_")[0]}.{demo_var.split(".")[-1]}')
    for col in temp.columns:
        demos.at[col] = temp[col].sum()
demos.to_csv(join(PROJ_DIR, OUTP_DIR, 'demographics.csv'))

rci_r = rci_over_time_sa.swaplevel(axis=1)['r']

# ok, first thing's first: we need to do some chi-square tests
# comparing those who do/don't align with the SA axis
chi2_vars = [
    'pds_p_ss_female_category_2.baseline_year_1_arm_1',
    'pds_p_ss_male_category_2.baseline_year_1_arm_1',
    'pds_p_ss_category_2.baseline_year_1_arm_1',
    'race_ethnicity.baseline_year_1_arm_1', 
    'sex.baseline_year_1_arm_1',
    'demo_prnt_ed_v2.baseline_year_1_arm_1',
    'pds_p_ss_male_category_2.2_year_follow_up_y_arm_1',
    'pds_p_ss_female_category_2.2_year_follow_up_y_arm_1',
    'pds_p_ss_category_2.2_year_follow_up_y_arm_1',
    'demo_comb_income_v2.baseline_year_1_arm_1',
    "bin-cbcl_scr_syn_anxdep_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_withdep_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_somatic_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_social_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_thought_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_attention_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_rulebreak_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_aggressive_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_internal_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_external_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_totprob_r.baseline_year_1_arm_1", 
    "bin-cbcl_scr_syn_anxdep_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_withdep_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_somatic_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_social_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_thought_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_attention_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_rulebreak_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_aggressive_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_internal_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_external_r.2_year_follow_up_y_arm_1", 
    "bin-cbcl_scr_syn_totprob_r.2_year_follow_up_y_arm_1", 
]

var_names = [
    'Female puberty @ age 9-10',
    'Male puberty @ age 9-10',
    'Puberty @ age 9-10',
    'Race/ethnicity', 
    'Sex',
    'Caregiver education',
    'Male puberty @ age 11-13',
    'Female puberty @ age 11-13',
    'Puberty @ age 11-13',
    'Household income',
    "Anxious/depressed", 
    "Withdrawn/depressed", 
    "Somatic complaints", 
    "Social problems", 
    "Thought problems", 
    "Attention problems", 
    "Rule breaking", 
    "Aggression", 
    "Internalizing", 
    "Externalizing", 
    "Total Problems"
]

alpha, _ = jili_sidak_mc(non_brain_df[chi2_vars], 0.05)

print('cbcl_alpha: ', jili_sidak_mc(non_brain_df.filter(like='bin-cbcl'), 0.05))

# run chi-square tests for each brain measure and categorical variable
cols = pd.MultiIndex.from_product([r_df.columns, ['x2','p']])
chi2_res = pd.DataFrame(index=chi2_vars, columns=cols)
for meas in r_df.columns:
    pos_r = r_df[p_df[meas] < 0.05][meas] > 0
    pos_r = pos_r[pos_r == True]

    neg_r = r_df[p_df[meas] < 0.05][meas] < 0
    neg_r = neg_r[neg_r == True]

    no_r = r_df[p_df[meas] > 0.05][meas] != 0
    
    neg_r = non_brain_df.loc[neg_r.index]
    pos_r = non_brain_df.loc[pos_r.index]
    no_r = non_brain_df.loc[no_r.index]
    
    for var in chi2_vars:
        contingency = pd.DataFrame()
        i = chi2_vars.index(var)
        for level in non_brain_df[var].dropna().unique():
            if type(level) == np.float64 and level > 100:
                pass
            else:
                contingency.at['pos', level] = len(pos_r[pos_r[var] == level].index)
                contingency.at['neg', level] = len(neg_r[neg_r[var] == level].index)
                contingency.at['non', level] = len(no_r[no_r[var] == level].index)
        
        contingency = contingency[contingency.columns[contingency.sum() > 0]]
        ans = chi2_contingency(contingency)
        chi2 = ans[0]
        p = ans[1]
        chi2_res.at[var, (meas, 'x2')] = chi2
        chi2_res.at[var, (meas, 'p')] = p
        fig,ax = plt.subplots(figsize=(5,4), layout='constrained')
        temp_pos = pos_r[var].dropna()
        temp_pos.name = 'positive'
        temp_neg = neg_r[var].dropna()
        temp_neg.name = 'negative'
        temp_non = no_r[var].dropna()
        temp_non.name = 'none'

        tempy_temp = pd.concat([temp_pos, temp_neg, temp_non], axis=1).melt()
        sns.histplot(
            tempy_temp, 
            x='value',
            ax=ax, 
            discrete=True, 
            stat='count', 
            legend=True, 
            hue='variable',
            #color='#f73952', 
            alpha=0.5
        )
        #sns.histplot(
        #    temp_neg, 
        #    ax=ax, 
        #    discrete=True, 
        #    stat='count', 
        #    legend=True, 
        #    color='#3982f7',
        #    alpha=0.5
        #)
        #ax.set_xlabel(var_names[i])
        if type(temp_pos.iloc[0]) == str:
            pass
        elif var == 'demo_comb_income_v2.baseline_year_1_arm_1':
            ax.set_xlim(left=0, right=10.5)
        else:
            ax.set_xlim(left=0, right=min(max(temp_neg.max(), temp_pos.max()), 21))
        if p < alpha:
            fig.savefig(f'../figures/apd-{meas}_{var}_+v-_{np.round(p,4)}.png', dpi=600, bbox_inches='tight')
        plt.close()
chi2_res.to_csv(join(PROJ_DIR, OUTP_DIR, 'chi2-results-apd-sa.csv'))

cols = pd.MultiIndex.from_product([rci_r.columns, ['x2','p']])
chi2_res = pd.DataFrame(index=chi2_vars, columns=cols)
for meas in rci_over_time_sa.columns.levels[0]:
    temp = rci_over_time_sa[meas]
    r_df = temp['r']
    p_df = temp['p']
    #print(p_df.columns)
    pos_r = r_df[p_df < 0.05] > 0
    pos_r = pos_r[pos_r == True]

    neg_r = r_df[p_df < 0.05] < 0
    neg_r = neg_r[neg_r == True]
    
    no_r = r_df[p_df > 0.05] != 0
    
    neg_r = non_brain_df.loc[neg_r.index]
    pos_r = non_brain_df.loc[pos_r.index]
    no_r = non_brain_df.loc[no_r.index]
    
    for var in chi2_vars:
        contingency = pd.DataFrame()
        i = chi2_vars.index(var)
        for level in non_brain_df[var].dropna().unique():
            if type(level) == np.float64 and level > 100:
                pass
            else:
                contingency.at['pos', level] = len(pos_r[pos_r[var] == level].index)
                contingency.at['neg', level] = len(neg_r[neg_r[var] == level].index)
                contingency.at['non', level] = len(no_r[no_r[var] == level].index)
        contingency = contingency[contingency.columns[contingency.sum() > 0]]
        ans = chi2_contingency(contingency)
        chi2 = ans[0]
        p = ans[1]
        chi2_res.at[var, (meas, 'x2')] = chi2
        chi2_res.at[var, (meas, 'p')] = p
        fig,ax = plt.subplots(figsize=(5,4), layout='constrained')
        temp_pos = pos_r[var].dropna()
        temp_pos.name = 'positive'
        temp_neg = neg_r[var].dropna()
        temp_neg.name = 'negative'
        temp_non = no_r[var].dropna()
        temp_non.name = 'none'
        tempy_temp = pd.concat([temp_pos, temp_neg, temp_non], axis=1).melt()
        sns.histplot(
            tempy_temp, 
            x='value',
            ax=ax, 
            discrete=True, 
            stat='count', 
            legend=True, 
            hue='variable',
            #color='#f73952', 
            alpha=0.5
        )
        #ax.set_xlabel(var_names[i])
        #if type(temp_pos.iloc[0]) == str:
        #    pass
        #elif var == 'demo_comb_income_v2.baseline_year_1_arm_1':
        #    ax.set_xlim(left=0, right=10.5)
        #else:
        #ax.set_xlim(left=0, right=min(max(temp_neg.max(), temp_pos.max()), 21))
        if p < alpha:
            fig.savefig(f'figures/rci-{meas}_{var}_+v-_{np.round(p,4)}.png', dpi=600, bbox_inches='tight')
        plt.close()
chi2_res.to_csv(join(PROJ_DIR, OUTP_DIR, 'chi2-results-rci_over_age-sa.csv'))
# and some semipartial correlations, just for fun

rci_sa_corrs = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'rci_x_SA-correlations.pkl'))

puberty_age = [
    'interview_age.baseline_year_1_arm_1',
    'interview_age.2_year_follow_up_y_arm_1',
    'pds_p_ss_category_2.2_year_follow_up_y_arm_1',
    'pds_p_ss_category_2.baseline_year_1_arm_1',
    'sex.baseline_year_1_arm_1'
]
temp = pd.concat([rci_sa_corrs, 
               non_brain_df[puberty_age],
               pd.get_dummies(non_brain_df['sex.baseline_year_1_arm_1'])
              ], 
              axis=1)
#print(temp.columns)
puberty_semipartial = pd.concat([
    pg.partial_corr(
    temp, 
    x='pds_p_ss_category_2.baseline_year_1_arm_1',
    y='thk',
    covar=['interview_age.baseline_year_1_arm_1','F'],
    method='spearman'
    ).rename({'spearman':'thk-base'}, axis=0),
    pg.partial_corr(
    temp, 
    x='pds_p_ss_category_2.baseline_year_1_arm_1',
    y='var',
    covar=['interview_age.baseline_year_1_arm_1','F'],
    method='spearman'
    ).rename({'spearman':'var-base'}, axis=0),
    pg.partial_corr(
    temp, 
    x='pds_p_ss_category_2.baseline_year_1_arm_1',
    y='rni',
    covar=['interview_age.baseline_year_1_arm_1','F'],
    method='spearman'
    ).rename({'spearman':'rni-base'}, axis=0),
    pg.partial_corr(
    temp, 
    x='pds_p_ss_category_2.baseline_year_1_arm_1',
    y='rnd',
    covar=['interview_age.baseline_year_1_arm_1','F'],
    method='spearman'
    ).rename({'spearman':'rnd-base'}, axis=0),
    ##############
    pg.partial_corr(
    temp, 
    x='pds_p_ss_category_2.2_year_follow_up_y_arm_1',
    y='thk',
    covar=['interview_age.2_year_follow_up_y_arm_1','F'],
    method='spearman'
    ).rename({'spearman':'thk-2yfu'}, axis=0),
    pg.partial_corr(
    temp, 
    x='pds_p_ss_category_2.2_year_follow_up_y_arm_1',
    y='var',
    covar=['interview_age.2_year_follow_up_y_arm_1','F'],
    method='spearman'
    ).rename({'spearman':'var-2yfu'}, axis=0),
    pg.partial_corr(
    temp, 
    x='pds_p_ss_category_2.2_year_follow_up_y_arm_1',
    y='rni',
    covar=['interview_age.2_year_follow_up_y_arm_1','F'],
    method='spearman'
    ).rename({'spearman':'rni-2yfu'}, axis=0),
    pg.partial_corr(
    temp, 
    x='pds_p_ss_category_2.2_year_follow_up_y_arm_1',
    y='rnd',
    covar=['interview_age.2_year_follow_up_y_arm_1','F'],
    method='spearman'
    ).rename({'spearman':'rnd-2yfu'}, axis=0),
])
puberty_semipartial.to_csv(join(PROJ_DIR, OUTP_DIR, 'sa_axis-puberty-semipartial_r.csv'))