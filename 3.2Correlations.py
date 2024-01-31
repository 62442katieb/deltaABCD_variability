#!/usr/bin/env python
# coding: utf-8

# In[10]:



pip install pyreadr


# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pyreadr


# In[2]:


from os.path import join
from scipy.stats import spearmanr


# In[3]:


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis//"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


# In[4]:


thk_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.csv'),
    index_col=0, header=0
)
var_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.csv'),
    index_col=0, header=0
)
rni_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.csv'),
    index_col=0, header=0
)
rnd_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.csv'),
    index_col=0, header=0
)


# In[5]:


# no need to do this for each of the SA/age dfs
# just showing you the structure of the data
rnd_df.head()


# In[12]:


# read in each .Rda file and run correlations
result = pyreadr.read_r(join(PROJ_DIR, OUTP_DIR, 'residualized_rnd.Rda'))
residualized_rnd = result['Group1_residuals']


# In[13]:


# this cell does the correlations

# first, make empty dataframes that we'll fill in the for loop
# for s-a axis loading corrs/alignment
sa_rnd_corrs = pd.DataFrame()
# and for age-10 map corrs/alignment
age_rnd_corrs = pd.DataFrame()


# In[14]:


# now for each person (i),
for i in residualized_rnd.index:
    # we'll grab all their residual RND change scores
    temp1 = residualized_rnd.loc[i]
    # and rename the mini-dataframe, so that we know these are per-participant values
    temp1.name = 'ppt'
    # fix the index so that it matches the SA-axis rank
    temp1.index = [var.split('.')[0] for var in temp1.index]
    # just grab the S-A axis rank column from rnd_df
    temp2 = rnd_df['SA_rank']
    # rename it so that we know these are per-region s-a axis values
    temp2.name = 'sa_axis'
    # put those two mini-dfs together to make life easier
    # this aligns them based on the index, which they share
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    # correlate! using the new names we gave the two columns
    r,p = spearmanr(temp['ppt'], temp['sa_axis'])
    # save the r values to the 'r' column and the p-values to the 'p' column
    sa_rnd_corrs.at[i,'p'] = p
    sa_rnd_corrs.at[i,'r'] = r


# In[15]:


# and now do it all over for the age map
temp2 = rnd_df['age_avg']
temp2.name = 'age_effect'
temp = pd.concat([temp1, temp2], axis=1).dropna()
r,p = spearmanr(temp['ppt'], temp['age_effect'])


# In[16]:


age_rnd_corrs.at[i,'p'] = p
age_rnd_corrs.at[i,'r'] = r


# In[17]:


sa_rnd_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'sa_rnd_corrs.csv'))
sa_rnd_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'age_rnd_corrs.csv'))


# In[18]:


# set the plotting settings so our graphs are pretty
sns.set(context='talk', style='white', palette='husl')


# In[22]:


# we're going to plot all the correlations
# and in a different color, the significant correlations @ p < 0.01
fig,ax = plt.subplots()
sns.kdeplot(sa_rnd_corrs['r'], fill=True, ax=ax)
sns.kdeplot(sa_rnd_corrs[sa_rnd_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'rnd_x_sa-axis.png'), bbox_inches='tight')


# In[42]:


# same for age effect
fig,ax = plt.subplots()
sns.kdeplot(age_rnd_corrs['r'], fill=True, ax=ax, warn_singular=False)
sns.kdeplot(age_rnd_corrs[age_rnd_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'rnd_x_age.png'), bbox_inches='tight')


# In[38]:


age_rnd_corrs.head()


# In[39]:


sa_rnd_corrs.head()


# In[41]:


# RNIGM
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pyreadr

from os.path import join
from scipy.stats import spearmanr


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis//"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


thk_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.csv'),
    index_col=0, header=0
)
var_df = pd.read_csv( 
    join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.csv'),
    index_col=0, header=0
)
rni_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.csv'),
    index_col=0, header=0
)
rnd_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.csv'),
    index_col=0, header=0
)


# no need to do this for each of the SA/age dfs
# just showing you the structure of the data
# rni_df.head()


# read in each .Rda file and run correlations
result = pyreadr.read_r(join(PROJ_DIR, OUTP_DIR, 'residualized_rnigm.Rda'))
residualized_rni = result['Group1_residuals']


# this cell does the correlations

# first, make empty dataframes that we'll fill in the for loop
# for s-a axis loading corrs/alignment
sa_rni_corrs = pd.DataFrame()
# and for age-10 map corrs/alignment
age_rni_corrs = pd.DataFrame()

# now for each person (i),
for i in residualized_rni.index:
    # we'll grab all their residual RNI change scores
    temp1 = residualized_rni.loc[i]
    # and rename the mini-dataframe, so that we know these are per-participant values
    temp1.name = 'ppt'
    # fix the index so that it matches the SA-axis rank
    temp1.index = [var.split('.')[0] for var in temp1.index]
    # just grab the S-A axis rank column from rni_df
    temp2 = rni_df['SA_rank']
    # rename it so that we know these are per-region s-a axis values
    temp2.name = 'sa_axis'
    # put those two mini-dfs together to make life easier
    # this aligns them based on the index, which they share
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    # correlate! using the new names we gave the two columns
    r,p = spearmanr(temp['ppt'], temp['sa_axis'])
    # save the r values to the 'r' column and the p-values to the 'p' column
    sa_rni_corrs.at[i,'p'] = p
    sa_rni_corrs.at[i,'r'] = r

    # and now do it all over for the age map
    temp2 = rni_df['age_avg']
    temp2.name = 'age_effect'
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    r,p = spearmanr(temp['ppt'], temp['age_effect'])

    age_rni_corrs.at[i,'p'] = p
    age_rni_corrs.at[i,'r'] = r

sa_rni_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'sa_rni_corrs.csv'))
sa_rni_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'age_rni_corrs.csv'))


# set the plotting settings so our graphs are pretty
sns.set(context='talk', style='white', palette='husl')

# we're going to plot all the correlations
# and in a different color, the significant correlations @ p < 0.01
fig,ax = plt.subplots()
sns.kdeplot(sa_rni_corrs['r'], fill=True, ax=ax, warn_singular=False)
sns.kdeplot(sa_rni_corrs[sa_rni_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'rni_x_sa-axis.png'), bbox_inches='tight')



# same for age effect
fig,ax = plt.subplots()
sns.kdeplot(age_rni_corrs['r'], fill=True, ax=ax, warn_singular=False)
sns.kdeplot(age_rni_corrs[age_rni_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'rni_x_age.png'), bbox_inches='tight')


# In[40]:


# no need to do this for each of the SA/age dfs
# just showing you the structure of the data
rni_df.head()


# In[43]:


age_rni_corrs.head()


# In[37]:


sa_rni_corrs.head()


# In[45]:


# RSFMRI
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pyreadr

from os.path import join
from scipy.stats import spearmanr


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis//"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


thk_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.csv'),
    index_col=0, header=0
)
var_df = pd.read_csv( 
    join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.csv'),
    index_col=0, header=0
)
rni_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.csv'),
    index_col=0, header=0
)
rnd_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.csv'),
    index_col=0, header=0
)


# no need to do this for each of the SA/age dfs
# just showing you the structure of the data
# var_df.head()


# read in each .Rda file and run correlations
result = pyreadr.read_r(join(PROJ_DIR, OUTP_DIR, 'residualized_rsfmri.Rda'))
residualized_var = result['Group1_residuals']


# this cell does the correlations

# first, make empty dataframes that we'll fill in the for loop
# for s-a axis loading corrs/alignment
sa_var_corrs = pd.DataFrame()
# and for age-10 map corrs/alignment
age_var_corrs = pd.DataFrame()

# now for each person (i),
for i in residualized_var.index:
    # we'll grab all their residual RSFMRI change scores
    temp1 = residualized_var.loc[i]
    # and rename the mini-dataframe, so that we know these are per-participant values
    temp1.name = 'ppt'
    # fix the index so that it matches the SA-axis rank
    temp1.index = [var.split('.')[0] for var in temp1.index]
    # just grab the S-A axis rank column from var_df
    temp2 = var_df['SA_rank']
    # rename it so that we know these are per-region s-a axis values
    temp2.name = 'sa_axis'
    # put those two mini-dfs together to make life easier
    # this aligns them based on the index, which they share
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    # correlate! using the new names we gave the two columns
    r,p = spearmanr(temp['ppt'], temp['sa_axis'])
    # save the r values to the 'r' column and the p-values to the 'p' column
    sa_var_corrs.at[i,'p'] = p
    sa_var_corrs.at[i,'r'] = r

    # and now do it all over for the age map
    temp2 = var_df['age_avg']
    temp2.name = 'age_effect'
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    r,p = spearmanr(temp['ppt'], temp['age_effect'])

    age_var_corrs.at[i,'p'] = p
    age_var_corrs.at[i,'r'] = r

sa_var_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'sa_var_corrs.csv'))
sa_var_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'age_var_corrs.csv'))


# set the plotting settings so our graphs are pretty
sns.set(context='talk', style='white', palette='husl')

# we're going to plot all the correlations
# and in a different color, the significant correlations @ p < 0.01
fig,ax = plt.subplots()
sns.kdeplot(sa_var_corrs['r'], fill=True, ax=ax, warn_singular=False)
sns.kdeplot(sa_var_corrs[sa_var_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'rsfmri_x_sa-axis.png'), bbox_inches='tight')



# same for age effect
fig,ax = plt.subplots()
sns.kdeplot(age_var_corrs['r'], fill=True, ax=ax, warn_singular=False)
sns.kdeplot(age_var_corrs[age_var_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'rsfmri_x_age.png'), bbox_inches='tight')


# In[46]:


# THICK
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pyreadr

from os.path import join
from scipy.stats import spearmanr


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis//"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


thk_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'smri_thick_age-SA.csv'),
    index_col=0, header=0
)
var_df = pd.read_csv( 
    join(PROJ_DIR, OUTP_DIR, 'rsfmri_var_age-SA.csv'),
    index_col=0, header=0
)
rni_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'dmri_rsirnigm_age-SA.csv'),
    index_col=0, header=0
)
rnd_df = pd.read_csv(
    join(PROJ_DIR, OUTP_DIR, 'dmri_rsirndgm_age-SA.csv'),
    index_col=0, header=0
)


# no need to do this for each of the SA/age dfs
# just showing you the structure of the data
# thk_df.head()


# read in each .Rda file and run correlations
result = pyreadr.read_r(join(PROJ_DIR, OUTP_DIR, 'residualized_thick.Rda'))
residualized_thk = result['Group1_residuals']


# this cell does the correlations

# first, make empty dataframes that we'll fill in the for loop
# for s-a axis loading corrs/alignment
sa_thk_corrs = pd.DataFrame()
# and for age-10 map corrs/alignment
age_thk_corrs = pd.DataFrame()

# now for each person (i),
for i in residualized_thk.index:
    # we'll grab all their residual thickness change scores
    temp1 = residualized_thk.loc[i]
    # and rename the mini-dataframe, so that we know these are per-participant values
    temp1.name = 'ppt'
    # fix the index so that it matches the SA-axis rank
    temp1.index = [var.split('.')[0] for var in temp1.index]
    # just grab the S-A axis rank column from thk_df
    temp2 = thk_df['SA_rank']
    # rename it so that we know these are per-region s-a axis values
    temp2.name = 'sa_axis'
    # put those two mini-dfs together to make life easier
    # this aligns them based on the index, which they share
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    # correlate! using the new names we gave the two columns
    r,p = spearmanr(temp['ppt'], temp['sa_axis'])
    # save the r values to the 'r' column and the p-values to the 'p' column
    sa_thk_corrs.at[i,'p'] = p
    sa_thk_corrs.at[i,'r'] = r

    # and now do it all over for the age map
    temp2 = thk_df['age_avg']
    temp2.name = 'age_effect'
    temp = pd.concat([temp1, temp2], axis=1).dropna()
    r,p = spearmanr(temp['ppt'], temp['age_effect'])

    age_thk_corrs.at[i,'p'] = p
    age_thk_corrs.at[i,'r'] = r

sa_thk_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'sa_thk_corrs.csv'))
sa_thk_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, 'age_thk_corrs.csv'))


# set the plotting settings so our graphs are pretty
sns.set(context='talk', style='white', palette='husl')

# we're going to plot all the correlations
# and in a different color, the significant correlations @ p < 0.01
fig,ax = plt.subplots()
sns.kdeplot(sa_thk_corrs['r'], fill=True, ax=ax, warn_singular=False)
sns.kdeplot(sa_thk_corrs[sa_thk_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'thk_x_sa-axis.png'), bbox_inches='tight')



# same for age effect
fig,ax = plt.subplots()
sns.kdeplot(age_thk_corrs['r'], fill=True, ax=ax, warn_singular=False)
sns.kdeplot(age_thk_corrs[age_thk_corrs['p'] < 0.01]['r'], fill=True, ax=ax)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'thk_x_age.png'), bbox_inches='tight')


# In[ ]:




