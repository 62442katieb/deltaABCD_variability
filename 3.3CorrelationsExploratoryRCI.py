import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
import abcdWrangler as abcdw

from os.path import join
from scipy.stats import spearmanr, fligner, variation
from nilearn import plotting

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

# set the plotting settings so our graphs are pretty
sns.set(context='talk', style='white', palette='Set2')
quartile_palette = sns.color_palette(['#A2B017', '#D87554', '#9518AA', '#0D0993'])

thk_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.pkl'))
var_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.pkl'))
rni_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.pkl'))
rnd_df = pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.pkl'))

change_scores = {
    'apd': pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_change_scores.pkl')),
    'rci': pd.read_pickle(join(PROJ_DIR, OUTP_DIR, 'residualized_rci.pkl')),
} 


big_df = pd.read_csv(join(PROJ_DIR, DATA_DIR, 'data_covar.csv'))


for score in change_scores.keys():
    df = change_scores[score]
    ppts = list(set(df.index) & set(big_df.index))
    # ONLY SIEMENS #
    smri = df.filter(regex='smri_thick_cdk.*')
    rni = df.filter(regex='dmri_rsirnigm_cdk.*')
    rnd = df.filter(regex='dmri_rsirndgm_cdk.*')
    var = df.filter(regex='rsfmri_var_cdk.*')

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

    smri_long = smri.melt()
    smri_long.index = smri_long['variable']

    rni_long = rni.melt()
    rni_long.index = rni_long['variable']

    rnd_long = rnd.melt()
    rnd_long.index = rnd_long['variable']

    var_long = var.melt()
    var_long.index = var_long['variable']

    f_ppts = list(set(big_df[big_df['demo_sex_v2_bl'] == 'Female'].index) & set(df.index))
    m_ppts = list(set(big_df[big_df['demo_sex_v2_bl'] == 'Male'].index) & set(df.index))


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
            'name': 'Cortical thickness',
            'vlike': 'smri_thick_cdk',
            'long_df': smri_long,
            'wide_df': smri,
            'sa_df': thk_df,
            'order': 2,
        },
        'rnd': {
            'name': 'Directional diffusion',
            'vlike': 'dmri_rsirndgm_cdk',
            'long_df': rnd_long,
            'wide_df': rnd,
            'sa_df': rnd_df,
            'order': 0,
        },
        'rni': {
            'name': 'Isotropic diffusion',
            'vlike': 'dmri_rsirnigm_cdk',
            'long_df': rni_long,
            'wide_df': rni,
            'sa_df': rni_df,
            'order': 1,
        },
        'var': {
            'name': 'BOLD variance',
            'vlike': 'rsfmri_var_cdk',
            'long_df': var_long,
            'wide_df': var,
            'sa_df': var_df,
            'order': 2,
        }
    }

    fk_stats = pd.DataFrame(
        index=the_data.keys()
    )

    ##########################################################################
    ### Exploratory analyses 1 & 2: quantile-wise alignment & hsk therein ####
    ##########################################################################

    ile = 4
        
    range_ = range(1,ile + 1)
    stats = ['r', 'p(r)']
    cols = pd.MultiIndex.from_product([stats, list(range_)])
    index = pd.MultiIndex.from_product([list(the_data.keys()), list(df.index)])

    corr_df = pd.DataFrame(
        index=index,
        columns=cols,
        dtype=float
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
        fk, p = fligner(
            dat[quantile[1]].dropna().melt()['value'], 
            dat[quantile[2]].dropna().melt()['value'], 
            dat[quantile[3]].dropna().melt()['value'], 
            dat[quantile[4]].dropna().melt()['value']
        )
        fk_stats.at[meas,'fk'] = fk
        fk_stats.at[meas,'p(fk)'] = p


    fig,ax = plt.subplots(nrows=4, ncols=2, figsize=(10,20), layout='constrained')
    for meas in the_data.keys():
        i = list(the_data.keys()).index(meas)
        temp = the_data[meas]
        temp_df = corr_df.loc[meas]
        temp_long = temp_df['r'].melt()
        g = sns.kdeplot(
            data=temp_long, 
            x='value', 
            hue='variable', 
            ax=ax[i,0],
            palette=quartile_palette
        )
        legend = g.get_legend()
        if i > 0:
            legend.remove()
        else:
            legend.set_bbox_to_anchor((1,1))
            legend.set_title('Quartile')
        corrs = temp_df['r'].corr(numeric_only=True).replace({1.0:-0.0})
        #drop = corrs.filter(like='p').columns
        #corrs = corrs.drop(drop, axis=1).drop(drop,axis=0)
        h = sns.heatmap(
            corrs, 
            cmap='seismic', 
            center=0, 
            square=True, 
            annot=True, 
            fmt=".2f",
            annot_kws={"fontsize":16,},
            ax=ax[i,1], 
            vmax=1, vmin=-1
        )
        g.set_title(temp['name'])
        g.set_xlabel('Spearman correlation')
        h.set_xlabel('Quantile')
        h.set_ylabel('')
        h.set_xticklabels(['S++', 'S+', 'A+', 'A++'])
        h.set_yticklabels(['S++', 'S+', 'A+', 'A++'])
        ax[i,0].axvline(temp_long['value'].mean(), color='#333333', linestyle='dotted', alpha=0.2)
    plt.yticks(rotation=0)
    fig.savefig(join(PROJ_DIR, FIGS_DIR, f'SA_axis-quantiles-{score}_avg.png'), dpi=400, bbox_inches='tight')
    plt.close()

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
        g.savefig(join(PROJ_DIR, FIGS_DIR, f'SA_axis-{meas}_lowess-{score}_avg.png'), dpi=400, bbox_inches='tight')


        z = np.polyfit(
            x=long.dropna()['SA_avg'], 
            y=long.dropna()['value'], 
            deg=order
            )
        p = np.poly1d(z)

        fig,ax = plt.subplots(figsize=(8,8))
        h = sns.lineplot(
            data=long.dropna(), 
            x='SA_avg',
            y='value',
            #y='delta_thickness',
            hue='hemisphere',
            marker='X',
            markersize=10,
            #alpha=0.01,
            #ax=g.axes[0][0],
            errorbar=('ci',90),
            linestyle='none',
            err_style='bars',
            #errwidth=0.5,
            ax=ax,
            
        )
        plt.plot(
            long.dropna()['SA_avg'].sort_values(), 
            p(long.dropna()['SA_avg'].sort_values()), 
            'k--', 
            label='polynomial',
        )

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
        #h.set_ylim(-50,50)
        #h.set_xticks(range(0, upper_max, upper_max // 5))
        #h.set_xticklabels(list(range(0,upper_max,upper_max // 5)))
        h.set_xlabel('Sensorimotor-association axis rank')
        h.set_ylabel(f"Reliable change in {temp['name']}")
        #ax.axhline(0, color='#333333', linestyle='dotted', alpha=0.2)
        sns.despine()
        fig.savefig(join(PROJ_DIR, FIGS_DIR, f'SA_axis-{meas}-{score}_avg.png'), dpi=400, bbox_inches='tight')
        plt.close()

    smri_long.to_csv(join(PROJ_DIR, OUTP_DIR, f'SA_axis-thick_long-{score}_avgs.csv'))
    rnd_long.to_csv(join(PROJ_DIR, OUTP_DIR, f'SA_axis-rnd_long-{score}_avgs.csv'))
    rni_long.to_csv(join(PROJ_DIR, OUTP_DIR, f'SA_axis-rni_long-{score}_avgs.csv'))
    var_long.to_csv(join(PROJ_DIR, OUTP_DIR, f'SA_axis-var_long-{score}_avgs.csv'))

    corr_df.to_csv(join(PROJ_DIR, OUTP_DIR, f'SA_axis-quartile_correlations-{score}_avg.csv'))
    fk_stats.to_csv(join(PROJ_DIR, OUTP_DIR, f'SA_axis-quartile_heteroscedasticity-{score}_avg.csv'))

    ##########################################################################
    ################# Exploratory analysis 4: age effects ####################
    ##########################################################################

    sa_breakdown = ['all'] + [f'q{i}' for i in range_]
    sexes = ['all', 'F', 'M']


    rci_over_time_sa = pd.DataFrame(
        dtype=float,
        index=df.index,
        columns=pd.MultiIndex.from_product([list(the_data.keys()), ['r', 'p']])
    )

    for meas in the_data.keys():
        temp = df.filter(like=the_data[meas]['vlike'], axis=1)
        sa = the_data[meas]['sa_df']['SA_avg']
        for i in temp.index:
            temp2 = pd.concat([temp.loc[i], sa], axis=1)
            cols = temp2.columns
            corr,p = spearmanr(temp2[cols[0]], temp2[cols[1]])
            rci_over_time_sa.at[i, (meas, 'r')] = corr
            rci_over_time_sa.at[i, (meas, 'p')] = p

    rci_over_time_sa.to_pickle(join(PROJ_DIR, OUTP_DIR, f'corr_{score}_over_time_sa.pkl'))

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

    for measure in the_data.keys():
        temp_dict = the_data[measure]
        temp_quantiles = quantiles[measure]
        temp_df = df.filter(like=temp_dict['vlike'], axis=1)

        sa = temp_dict['sa_df']['SA_avg']
        
        mean_age = df.filter(like='interview_age', axis=1).T.mean()
        mean_age.name = 'age'
        order = temp_dict['order']
        
        for col in temp_df.columns:
            temp = pd.concat([mean_age, temp_df[col]], axis=1)

            age_fxs.at[col,'all'] = temp.corr(method='spearman', numeric_only=True).loc[col]['age']
            age_fxs.at[col,'F'] = temp.loc[f_ppts].corr(method='spearman', numeric_only=True).loc[col]['age']
            age_fxs.at[col,'M'] = temp.loc[m_ppts].corr(method='spearman', numeric_only=True).loc[col]['age']
        age_fx_nii = abcdw.series_2_nifti(age_fxs['all'].filter(like=temp_dict['vlike']), out_dir=None)
        age_fx_plot = plotting.plot_img_on_surf(
            age_fx_nii, 
            symmetric=True, 
            cmap='RdBu_r'
        )
        age_fx_plot[0].savefig(
            join(PROJ_DIR, FIGS_DIR, f'{score}-{measure}_x_change.png'),
            bbox_inches='tight',
            dpi=400
        )
        age_sa = pd.concat(
            [
                age_fxs.filter(like=temp_dict['vlike'], axis=0),
                sa
            ],
            axis=1
        )
        sns.set(context='talk', style='white', palette='Greys_r')
        long_age = age_sa.melt(id_vars='SA_avg', var_name='ppts', value_name='age_corr')
        g = sns.lmplot(age_sa, x='SA_avg', y='all',  #hue='ppts', 
                    order=order, aspect=1.2)
        g.facet_axis(0,0).set_ylabel(f'Age-change correlation: {temp_dict["name"]}')
        g.savefig(
            join(
                PROJ_DIR, FIGS_DIR, f'{score}_x_age-{measure}-correlation-SA_avg-polynomial-all.png'
                ),
                dpi=400, bbox_inches='tight'
            )
        
        h = sns.lmplot(long_age, x='SA_avg', y='age_corr', hue='ppts', lowess=True, aspect=1.2)
        h.facet_axis(0,0).set_ylabel(f'Age-change correlation: {temp_dict["name"]}')
        h.savefig(
            join(
                PROJ_DIR, FIGS_DIR, f'{score}_x_age-{measure}-correlation-SA_avg-lowess.png'
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
    age_x_sa_q.to_csv(join(PROJ_DIR, OUTP_DIR, f'{score}_x_age-correlations-SA_avg.csv'))
    age_fxs.to_pickle(join(PROJ_DIR, OUTP_DIR, f'{score}_x_age-correlations.pkl'))

    all_sa = pd.concat([thk_df, rni_df, rnd_df, var_df], axis=0)['SA_avg']
    sex = pd.get_dummies(big_df['demo_sex_v2_bl'].dropna())['Female']

    # now do it again for rci
    age_fx_sa = pd.concat([age_fxs, all_sa / 1000, sex], axis=1)
    age_fx_sa['SA_avg^2'] = age_fx_sa['SA_avg'] ** 2

    print(score, 'partial corrs')
    smri = age_fx_sa.filter(like='smri', axis=0)
    partial1 = pg.partial_corr(smri, x='all', y='SA_avg', covar='SA_avg^2', method='spearman')
    partial2 = pg.partial_corr(smri, x='all', y='SA_avg^2', covar='SA_avg', method='spearman')
    print(f"thk:\nlinear:\tr_s = {np.round(partial1['r'],2)}\tp = {np.round(partial1['p-val'],4)}")
    print(f'quadtc:\tr_s = {np.round(partial2["r"],2)}\tp = {np.round(partial2["p-val"],4)}')
    smri_reg = pg.linear_regression(X=smri[['SA_avg', 'SA_avg^2']], y=smri['all'], remove_na=True)
    smri_reg.index = [f'thk_{i}' for i in smri_reg.index]
    smri_reg.at['thk_1','standard_b'] = smri_reg.loc['thk_1']['coef'] * np.std(smri['SA_avg']) / np.std(smri['all'])
    smri_reg.at['thk_2','standard_b'] = smri_reg.loc['thk_2']['coef'] * np.std(smri['SA_avg^2']) / np.std(smri['all'])

    smri_reg.at['thk_1','standard_2.5'] = smri_reg.loc['thk_1']['CI[2.5%]'] * np.std(smri['SA_avg']) / np.std(smri['all'])
    smri_reg.at['thk_1','standard_97.5'] = smri_reg.loc['thk_1']['CI[97.5%]'] * np.std(smri['SA_avg']) / np.std(smri['all'])

    smri_reg.at['thk_2','standard_2.5'] = smri_reg.loc['thk_2']['CI[2.5%]'] * np.std(smri['SA_avg^2']) / np.std(smri['all'])
    smri_reg.at['thk_2','standard_97.5'] = smri_reg.loc['thk_2']['CI[97.5%]'] * np.std(smri['SA_avg^2']) / np.std(smri['all'])

    rsfmri = age_fx_sa.filter(like='rsfmri', axis=0)
    partial1 = pg.partial_corr(rsfmri, x='all', y='SA_avg', covar='SA_avg^2', method='spearman')
    partial2 = pg.partial_corr(rsfmri, x='all', y='SA_avg^2', covar='SA_avg', method='spearman')
    print(f"var:\nlinear:\tr_s = {np.round(partial1['r'],2)}\tp = {np.round(partial1['p-val'],4)}")
    print(f"quadtc:\tr_s = {np.round(partial2['r'],2)}\tp = {np.round(partial2['p-val'],4)}")
    rsfmri_reg = pg.linear_regression(X=rsfmri[['SA_avg', 'SA_avg^2']], y=rsfmri['all'], remove_na=True)
    rsfmri_reg.index = [f'var_{i}' for i in rsfmri_reg.index]
    rsfmri_reg.at['var_1','standard_b'] = rsfmri_reg.loc['var_1']['coef'] * np.std(rsfmri['SA_avg']) / np.std(rsfmri['all'])
    rsfmri_reg.at['var_2','standard_b'] = rsfmri_reg.loc['var_2']['coef'] * np.std(rsfmri['SA_avg^2']) / np.std(rsfmri['all'])

    rsfmri_reg.at['var_1','standard_2.5'] = rsfmri_reg.loc['var_1']['CI[2.5%]'] * np.std(rsfmri['SA_avg']) / np.std(rsfmri['all'])
    rsfmri_reg.at['var_1','standard_97.5'] = rsfmri_reg.loc['var_1']['CI[97.5%]'] * np.std(rsfmri['SA_avg']) / np.std(rsfmri['all'])
    rsfmri_reg.at['var_2','standard_2.5'] = rsfmri_reg.loc['var_2']['CI[2.5%]'] * np.std(rsfmri['SA_avg^2']) / np.std(rsfmri['all'])
    rsfmri_reg.at['var_2','standard_97.5'] = rsfmri_reg.loc['var_2']['CI[97.5%]'] * np.std(rsfmri['SA_avg^2']) / np.std(rsfmri['all'])

    rnd = age_fx_sa.filter(like='dmri_rsirnd', axis=0)
    partial1 = pg.partial_corr(rnd, x='all', y='SA_avg', covar='SA_avg^2', method='spearman')
    partial2 = pg.partial_corr(rnd, x='all', y='SA_avg^2', covar='SA_avg', method='spearman')
    print(f"rnd:\nlinear:\tr_s = {np.round(partial1['r'],2)}\tp = {np.round(partial1['p-val'],4)}")
    print(f"quadtc:\tr_s = {np.round(partial2['r'],2)}\tp = {np.round(partial2['p-val'],4)}")
    rnd_reg = pg.linear_regression(X=rnd[['SA_avg', 'SA_avg^2']], y=rnd['all'], remove_na=True)
    rnd_reg.index = [f'rnd_{i}' for i in rnd_reg.index]
    rnd_reg.at['rnd_1','standard_b'] = rnd_reg.loc['rnd_1']['coef'] * np.std(rnd['SA_avg']) / np.std(rnd['all'])
    rnd_reg.at['rnd_2','standard_b'] = rnd_reg.loc['rnd_2']['coef'] * np.std(rnd['SA_avg^2']) / np.std(rnd['all'])

    rnd_reg.at['rnd_1','standard_2.5'] = rnd_reg.loc['rnd_1']['CI[2.5%]'] * np.std(rnd['SA_avg']) / np.std(rnd['all'])
    rnd_reg.at['rnd_1','standard_97.5'] = rnd_reg.loc['rnd_1']['CI[97.5%]'] * np.std(rnd['SA_avg']) / np.std(rnd['all'])

    rnd_reg.at['rnd_2','standard_2.5'] = rnd_reg.loc['rnd_2']['CI[2.5%]'] * np.std(rnd['SA_avg^2']) / np.std(rnd['all'])
    rnd_reg.at['rnd_2','standard_97.5'] = rnd_reg.loc['rnd_2']['CI[97.5%]'] * np.std(rnd['SA_avg^2']) / np.std(rnd['all'])

    rni = age_fx_sa.filter(like='dmri_rsirni', axis=0)
    partial1 = pg.partial_corr(rni, x='all', y='SA_avg', covar='SA_avg^2', method='spearman')
    partial2 = pg.partial_corr(rni, x='all', y='SA_avg^2', covar='SA_avg', method='spearman')
    print(f"rni:\nlinear:\tr_s = {np.round(partial1['r'],2)}\tp = {np.round(partial1['p-val'],4)}")
    print(f"quadtc:\tr_s = {np.round(partial2['r'],2)}\tp = {np.round(partial2['p-val'],4)}")
    rni_reg = pg.linear_regression(X=rni[['SA_avg', 'SA_avg^2']], y=rni['all'], remove_na=True)
    rni_reg.index = [f'rni_{i}' for i in rni_reg.index]
    rni_reg.at['rni_1','standard_b'] = rni_reg.loc['rni_1']['coef'] * np.std(rni['SA_avg']) / np.std(rni['all'])
    rni_reg.at['rni_2','standard_b'] = rni_reg.loc['rni_2']['coef'] * np.std(rni['SA_avg^2']) / np.std(rni['all'])

    rni_reg.at['rni_1','standard_2.5'] = rni_reg.loc['rni_1']['CI[2.5%]'] * np.std(rni['SA_avg']) / np.std(rni['all'])
    rni_reg.at['rni_1','standard_97.5'] = rni_reg.loc['rni_1']['CI[97.5%]'] * np.std(rni['SA_avg']) / np.std(rni['all'])
    rni_reg.at['rni_2','standard_2.5'] = rni_reg.loc['rni_2']['CI[2.5%]'] * np.std(rni['SA_avg^2']) / np.std(rni['all'])
    rni_reg.at['rni_2','standard_97.5'] = rni_reg.loc['rni_2']['CI[97.5%]'] * np.std(rni['SA_avg^2']) / np.std(rni['all'])

    regressions = pd.concat([smri_reg, rsfmri_reg, rni_reg, rnd_reg], axis=0)
    regressions.to_csv(join(PROJ_DIR, OUTP_DIR, f'{score}_x_age-sa_axis-regressions.csv'))
