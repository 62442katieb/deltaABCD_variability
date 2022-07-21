#!/usr/bin/env python
# coding: utf-8

# In[2]:


import enlighten
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from os.path import join, exists
from sklearn.model_selection import KFold
from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LinearRegression
from impyute.util import find_null
from impyute.util import checks
from impyute.util import preprocess
#from impyute.imputation.cs import mice, fast_knn


# In[3]:


def mice(data, **kwargs):
    """Multivariate Imputation by Chained Equations

    Reference:
        Buuren, S. V., & Groothuis-Oudshoorn, K. (2011). Mice: Multivariate
        Imputation by Chained Equations in R. Journal of Statistical Software,
        45(3). doi:10.18637/jss.v045.i03

    Implementation follows the main idea from the paper above. Differs in
    decision of which variable to regress on (here, I choose it at random).
    Also differs in stopping criterion (here the model stops after change in
    prediction from previous prediction is less than 10%).
    
    THIS IS FROM IMPYUTE, BUT I WANT TO OPTIMIZE AND/OR PARALLELIZE IT
    CURRENTLY TAKES SEVERAL DAYS TO RUN WITH ABCD IMAGING DATA

    Parameters
    ----------
    data: numpy.ndarray
        Data to impute.

    Returns
    -------
    numpy.ndarray
        Imputed data.

    """
    null_xy = find_null(data)

    # Add a column of zeros to the index values
    null_xyv = np.append(null_xy, np.zeros((np.shape(null_xy)[0], 1)), axis=1)

    null_xyv = [[int(x), int(y), v] for x, y, v in null_xyv]
    temp = []
    cols_missing = set([y for _, y, _ in null_xyv])

    # Step 1: Simple Imputation, these are just placeholders
    for x_i, y_i, value in null_xyv:
        # Column containing nan value without the nan value
        col = data[:, [y_i]][~np.isnan(data[:, [y_i]])]

        new_value = np.mean(col)
        data[x_i][y_i] = new_value
        temp.append([x_i, y_i, new_value])
    null_xyv = temp

    # Step 5: Repeat step 2 - 4 until convergence (the 100 is arbitrary)

    converged = [False] * len(null_xyv)
    
    while all(converged):
        print(converged)
        # Step 2: Placeholders are set back to missing for one variable/column
        dependent_col = int(np.random.choice(list(cols_missing)))
        missing_xs = [int(x) for x, y, value in null_xyv if y == dependent_col]

        # Step 3: Perform linear regression using the other variables
        x_train, y_train = [], []
        for x_i in (x_i for x_i in range(len(data)) if x_i not in missing_xs):
            x_train.append(np.delete(data[x_i], dependent_col))
            y_train.append(data[x_i][dependent_col])
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Step 4: Missing values for the missing variable/column are replaced
        # with predictions from our new linear regression model
        temp = []
        # For null indices with the dependent column that was randomly chosen
        for i, x_i, y_i, value in enumerate(null_xyv):
            if y_i == dependent_col:
                # Row 'x' without the nan value
                new_value = model.predict(np.delete(data[x_i], dependent_col))
                data[x_i][y_i] = new_value.reshape(1, -1)
                temp.append([x_i, y_i, new_value])
                delta = (new_value-value)/value
                if delta < 0.1:
                    converged[i] = True
        null_xyv = temp
    return data


# In[4]:


sns.set(style='whitegrid', context='talk')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'


# In[5]:


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


# In[6]:


df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data.csv"), index_col=0, header=0)
df.drop(list(df.filter(regex='lesion.*').columns), axis=1, inplace=True)
no_2yfu = df[df["interview_date.2_year_follow_up_y_arm_1"].isna() == True].index
df = df.drop(no_2yfu, axis=0)


# In[7]:


deltasmri_complete = pd.concat([df.filter(regex='smri.*change_score'), 
                                df.filter(regex='mrisdp.*change_score')], axis=1).dropna()
deltarsfmri_complete = df.filter(regex='rsfmri.*change_score').dropna(how='any')
deltarsi_complete = df.filter(regex='dmri_rsi.*change_score').dropna()
deltadti_complete = df.filter(regex='dmri_dti.*change_score').dropna()


# In[8]:


# subset by atlas for smri and rsfmri (variance) data
smri_atlas = {'cdk': [], 
              'mrisdp': [],
              #'cf12': []
             }
rsfmri_atlas = {'cdk': [],
                'cortgordon': []
               }

for atlas in smri_atlas.keys():
    smri_atlas[atlas] = list(deltasmri_complete.filter(regex=f'{atlas}.*').columns)

for atlas in rsfmri_atlas.keys():
    rsfmri_atlas[atlas] = list(deltarsfmri_complete.filter(regex=f'{atlas}.*').columns)


# In[9]:


smri_scs = list(deltasmri_complete.filter(regex='.*vol_scs_.*').columns)
rsfmri_scs = list(deltarsfmri_complete.filter(regex='.*_scs_.*').columns)


# In[13]:


# build data subsets for clustering
subcort = smri_scs + rsfmri_scs
cdk_columns = smri_atlas['cdk'] + rsfmri_atlas['cdk'] + list(deltarsi_complete.columns) + list(deltadti_complete.columns) + subcort
cdk_data = df.filter(cdk_columns)

dcg_columns = smri_atlas['mrisdp'] + rsfmri_atlas['cortgordon'] + list(deltarsi_complete.columns) + list(deltadti_complete.columns) + subcort
dcg_data = df.filter(dcg_columns)


# In[16]:


#get_ipython().run_line_magic('timeit', '')
# let's impute some missing values, heyyyyy
cdk_data_complete = mice(cdk_data.values)
dcg_data_complete = mice(dcg_data.values)


# In[17]:


imputed_cdk = pd.DataFrame(data=cdk_data_complete, 
                           columns=cdk_data.columns, 
                           index=cdk_data.index)

imputed_dcg = pd.DataFrame(data=dcg_data_complete, 
                           columns=dcg_data.columns, 
                           index=dcg_data.index)


# In[19]:


imputed_cdk.describe()


# In[20]:


imputed_dcg.describe()


# In[ ]:


imputed_cdk.to_csv(join(join(PROJ_DIR, DATA_DIR, "desikankillany_MICEimputed_data.csv")))
imputed_dcg.to_csv(join(join(PROJ_DIR, DATA_DIR, "destrieux+gordon_MICEimputed_data.csv")))


# In[ ]:




