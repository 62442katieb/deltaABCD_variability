#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
import matplotlib.pyplot as plt
import nibabel.freesurfer.mghformat as mgh

from glob import glob
from os.path import join, exists
from nilearn import plotting, datasets, image


# In[2]:


sns.set(style='whitegrid', context='talk')
plt.rcParams["font.family"] = "monospace"
plt.rcParams['font.monospace'] = 'Courier New'


# In[3]:


ABCD_DIR = "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0"
PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"
RESR_DIR = "resources/"


# In[4]:


data_dictionary = pd.read_csv(join(ABCD_DIR, 'generate_dataset/data_element_names.csv'), 
                              header=0, 
                              index_col=0)


# In[5]:


data = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data_qcd.csv"), index_col=0, header=0)


# In[6]:


smri_vars = pd.concat([data.filter(regex='smri.*change_score'), 
                                data.filter(regex='mrisdp.*change_score')], axis=1).dropna(how='all').columns
rsfmri_vars = data.filter(regex='rsfmri.*change_score').dropna(how='all').columns
rsi_vars = data.filter(regex='dmri_rsi.*change_score').dropna(how='all').columns
dti_vars = data.filter(regex='dmri_dti.*change_score').dropna(how='all').columns


# In[7]:


smri_vars = [v.split('.')[0] for v in smri_vars]
rsfmri_vars = [v.split('.')[0] for v in rsfmri_vars]
rsi_vars = [v.split('.')[0] for v in rsi_vars]
dti_vars = [v.split('.')[0] for v in dti_vars]


# In[8]:


imaging_vars = list(smri_vars) + list(rsfmri_vars) + list(rsi_vars) + list(dti_vars)


# In[9]:


mapping = pd.DataFrame(columns=['modality', 
                                'abcd_structure',
                                'abcd_description', 
                                'atlas_description', 
                                'atlas', 
                                'atlas_value',
                                'atlas_fname'])


# In[10]:


destrieux = datasets.fetch_atlas_surf_destrieux()
destrieux_vol = datasets.fetch_atlas_destrieux_2009()
desikan = datasets.fetch_neurovault_ids(image_ids=(23262, ))
gordon_og = '/Users/katherine.b/Dropbox/Mac/Downloads/Parcels 2/Parcels_MNI_222.nii'
gordon_xl = '/Users/katherine.b/Dropbox/Mac/Downloads/Parcels 2/Parcels.xlsx'
subcort = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr50-2mm')


# In[11]:


scs_nifti = subcort.maps


# In[12]:


nib.save(scs_nifti, join(PROJ_DIR, RESR_DIR, 'harvox-subcortical-maxprob-thr50-2mm'))


# In[13]:


for i in imaging_vars:
    if i in smri_vars:
        mapping.at[i, 'modality'] = 'smri'
    elif i in rsfmri_vars:
        mapping.at[i, 'modality'] = 'fmri' 
    elif i in rsi_vars:
        mapping.at[i, 'modality'] = 'rsi'
    elif i in dti_vars:
        mapping.at[i, 'modality'] = 'dti'
    mapping.at[i, 'abcd_description'] = data_dictionary.loc[i]['description']
    mapping.at[i, 'abcd_structure'] = data_dictionary.loc[i]['structure']
    if '_scs_' in i:
        mapping.at[i, 'atlas'] = 'Subcortical'
        mapping.at[i, 'atlas_fname'] = join(PROJ_DIR, RESR_DIR, 'harvox-subcortical-maxprob-thr50-2mm.nii')
    elif 'gordon' in i:
        mapping.at[i, 'atlas'] = 'Gordon'
        #mapping.at[i, 'atlas_fname'] = 
    elif '_cdk_' in i:
        mapping.at[i, 'atlas'] = 'Desikan'
        mapping.at[i, 'atlas_fname'] = desikan.images[0]
    elif 'mrisdp' in i:
        mapping.at[i, 'atlas'] = 'Destrieux'
        mapping.at[i, 'atlas_fname'] = destrieux_vol.maps
    elif '_cf12_' in i:
        # remove
        mapping.at[i, 'atlas'] = 'Fuzzy 12'
    elif 'fib' in i:
        mapping.at[i, 'atlas'] = 'Fiber Atlas'


# In[14]:


l_and_r_destrieux = destrieux.labels * 2
for i in range(0, len(l_and_r_destrieux)):
    if i == 0:
        pass
    elif i < 42:
        mapping.at[f'mrisdp_{i}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{i}','atlas_value'] = i
        mapping.at[f'mrisdp_{302 + i}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{302 + i}','atlas_value'] = i
        mapping.at[f'mrisdp_{906 + i}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{906 + i}','atlas_value'] = i
    elif i == 42:
        pass
    elif 42 < i < 76:
        mapping.at[f'mrisdp_{i - 1}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{i - 1}','atlas_value'] = i
        mapping.at[f'mrisdp_{302 + i - 1}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{302 + i - 1}','atlas_value'] = i
        mapping.at[f'mrisdp_{906 + i - 1}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{906 + i - 1}','atlas_value'] = i
    elif i == 76:
        pass
    elif 76 < i < 116:
        mapping.at[f'mrisdp_{i - 2}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{i - 2}','atlas_value'] = i
        mapping.at[f'mrisdp_{302 + i - 2}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{302 + i - 2}','atlas_value'] = i
        mapping.at[f'mrisdp_{906 + i - 2}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{906 + i - 2}','atlas_value'] = i
    elif i == 116:
        pass
    elif i > 116:
        mapping.at[f'mrisdp_{i - 3}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{i - 3}','atlas_value'] = i
        mapping.at[f'mrisdp_{302 + i - 3}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{302 + i - 3}','atlas_value'] = i
        mapping.at[f'mrisdp_{906 + i - 3}','atlas_description'] = l_and_r_destrieux[i]
        mapping.at[f'mrisdp_{906 + i - 3}','atlas_value'] = i


# In[15]:


desikan_labels = pd.read_csv(join(PROJ_DIR, DATA_DIR, 'desikan_labels.txt'), 
                             sep='\t', 
                             header=0,
                             usecols=['value']
                            )


# In[16]:


cdk_area = mapping.filter(regex='smri_area.*_cdk_.*', axis=0).index
cdk_thick = mapping.filter(regex='smri_thick.*_cdk_.*', axis=0).index
cdk_t1wcnt = mapping.filter(regex='smri_t1wcnt.*_cdk_.*', axis=0).index
cdk_tvar = mapping.filter(regex='rsfmri_var_cdk_.*', axis=0).index
cdk_rnd = mapping.filter(regex='dmri_rsirndgm_cdk.*', axis=0).index
cdk_rni = mapping.filter(regex='dmri_rsirnigm_cdk.*', axis=0).index


# In[17]:


for i in range(0, len(desikan_labels.values)):
    mapping.at[cdk_area[i],'atlas_description'] = desikan_labels.iloc[i]['value']
    mapping.at[cdk_area[i],'atlas_value'] = desikan_labels.index[i]
    mapping.at[cdk_thick[i],'atlas_description'] = desikan_labels.iloc[i]['value']
    mapping.at[cdk_thick[i],'atlas_value'] = desikan_labels.index[i]
    mapping.at[cdk_t1wcnt[i],'atlas_description'] = desikan_labels.iloc[i]['value']
    mapping.at[cdk_t1wcnt[i],'atlas_value'] = desikan_labels.index[i]
    mapping.at[cdk_tvar[i],'atlas_description'] = desikan_labels.iloc[i]['value']
    mapping.at[cdk_tvar[i],'atlas_value'] = desikan_labels.index[i]
    mapping.at[cdk_rnd[i],'atlas_description'] = desikan_labels.iloc[i]['value']
    mapping.at[cdk_rnd[i],'atlas_value'] = desikan_labels.index[i]
    mapping.at[cdk_rni[i],'atlas_description'] = desikan_labels.iloc[i]['value']
    mapping.at[cdk_rni[i],'atlas_value'] = desikan_labels.index[i]


# In[18]:


scs_1 = ['x',
         'x',
         'x',
         'x',
        "tplh",
        "caudatelh",
        "putamenlh",
        "pallidumlh",
        "bstem",
        "hpuslh",
        "amygdalalh",
        "aal",
        'x',
        'x',
         'x',
        "tprh",
        "caudaterh",
        "putamenrh",
        "pallidumrh",
        "hpusrh",
        "amygdalarh",
        "aar"]


# In[19]:


subcort_map = {}
for i in range(0, len(scs_1)):
    if scs_1[i] == 'x':
        pass
    else:
        subcort_map[scs_1[i]] = [subcort.labels[i], i]


# In[20]:


gordon_nii = nib.load(gordon_og)
gordon_arr = gordon_nii.get_fdata()
gordon_mapping = pd.read_excel(gordon_xl, index_col=0)
networks = list(gordon_mapping['Community'].unique())
gordon2abcd = {'Default': 'dt', 
               'SMhand': 'smh', 
               'SMmouth': 'smm', 
               'Visual': 'vs', 
               'FrontoParietal': 'fo', 
               'Auditory': 'ad', 
               'None': 'n', 
               'CinguloParietal': 'ca', 
               'RetrosplenialTemporal': 'rspltp', 
               'CinguloOperc': 'cgc', 
               'VentralAttn': 'vta', 
               'Salience': 'sa', 
               'DorsalAttn': 'dla'}
for i in range(len(networks)):
    indices = gordon_mapping[gordon_mapping['Community'] == networks[i]].index
    for j in indices:
        gordon_mapping.loc[j,'ntwk_label'] = i + 1
        gordon_mapping.loc[j,'abcd_label'] = gordon2abcd[networks[i]]
        
new_gordon_arr = np.zeros_like(gordon_arr)
for i in gordon_mapping.index:
    new_gordon_arr[np.where(gordon_arr == i)] = gordon_mapping.loc[i,'ntwk_label']
new_gordon_nii = nib.Nifti2Image(new_gordon_arr, gordon_nii.affine)
out_path = join(PROJ_DIR, RESR_DIR, 'gordon_networks_222')

nib.save(new_gordon_nii, 
         f'{out_path}.nii')
rsfcntwks = mapping.filter(regex='.*_c_', axis=0).index
mapping.loc[rsfcntwks, 'atlas'] = 'Gordon Networks'
mapping.loc[rsfcntwks, 'atlas_fname'] = f'{out_path}.nii'
for i in rsfcntwks:
    abcd_ntwk = i.split('_')[3]
    gordon_ntwk = list(gordon2abcd.keys())[list(gordon2abcd.values()).index(abcd_ntwk)]
    mapping.loc[i, 'atlas_description'] = gordon_ntwk
    value = gordon_mapping[gordon_mapping['abcd_label'] == abcd_ntwk]['ntwk_label'].unique()[0]
    mapping.loc[i, 'atlas_value'] = value


# In[21]:


all_meas = [i.split('_')[-1] for i in list(mapping.index)]


# In[22]:


scs_fc = {'aalh': ['Left Accumbens', 11], 
          'aarh': ['Right Accumbens', 21], 
          'aglh': ['Left Amygdala', 10], 
          'agrh': ['Right Amygdala', 20], 
          'bs': ['Brain-Stem', 8], 
          'cdelh': ['Left Caudate', 5], 
          'cderh': ['Right Caudate', 16], 
          'crcxlh': ['Left Cerebellum', np.nan],
          'crcxrh': ['Right Cerebellum', np.nan], 
          'hplh': ['Left Hippocampus', 9], 
          'hprh': ['Right Hippocampus', 19], 
          'pllh': ['Left Pallidum', 7], 
          'plrh': ['Right Pallidum', 18], 
          'ptlh': ['Left Putamen', 6], 
          'ptrh': ['Right Putamen', 17], 
          'thplh': ['Left Thalamus', 4],
          'thprh': ['Right Thalamus', 15]}


# In[23]:


rsi_scs_vars = mapping.filter(regex=f'.*_rsi.*_scs_.*', axis=0).index
rsi_scs = [i.split('_')[-1] for i in rsi_scs_vars]


# In[24]:


rsi_scs_map = {'aalh': ['Left Accumbens', 11], 
               'aarh': ['Right Accumbens', 21],
               'ablh': ['Left Accumbens', 11], 
               'abrh': ['Right Accumbens', 21], 
               'aglh': ['Left Amygdala', 10], 
               'agrh': ['Right Amygdala', 20], 
               'bs': ['Brain-Stem', 8], 
               'cbclh': ['Left Cerebellar Cortex', np.nan], 
               'cbcrh': ['Right Cerebellar Cortex', np.nan], 
               'cbwmlh': ['Left Cerebellar White Matter', np.nan],
               'cbwmrh': ['Right Cerebellar White Matter', np.nan], 
               'cdlh': ['Left Caudate', 5], 
               'cdrh': ['Right Caudate', 16], 
               'cdelh': ['Left Caudate', 5], 
               'cderh': ['Right Caudate', 16], 
               'cwmlh': ['Left Cerebral White Matter', np.nan], 
               'cwmrh': ['Right Cerebral White Matter', np.nan], 
               'hclh': ['Left Hippocampus', 9], 
               'hcrh': ['Right Hippocampus', 19],
               'hplh': ['Left Hippocampus', 9], 
               'hprh': ['Right Hippocampus', 19],
               'ilvlh': ['Left Inferior Lateral Ventrical', np.nan], 
               'ilvrh': ['Right Inferior Lateral Ventrical', np.nan], 
               'lvrh': ['Right Lateral Ventrical', np.nan], 
               'pllh': ['Left Pallidum', 7], 
               'plrh': ['Right Pallidum', 18], 
               'ptlh': ['Left Putamen', 6], 
               'ptrh': ['Right Putamen', 17], 
               'tplh': ['Left Thalamus', 4],
               'tprh': ['Right Thalamus', 15],
               'thplh': ['Left Thalamus', 4],
               'thprh': ['Right Thalamus', 15],
               'vdclh': ['Left Ventricle', np.nan], 
               'vdcrh': ['Right Ventricle', np.nan]}


# In[25]:


# assign values to subcortical variables
scs_vars = list(mapping.filter(regex='.*_scs_.*', axis=0).index)
for key in scs_vars:
    #print(key)
    scs_key = key.split('_')[-1]
    if '_cor_' in key:
        if not scs_key in rsi_scs_map:
            print(f'{scs_key} not in rsi_scs_map')
            pass
        else:
            print(scs_key, rsi_scs_map[scs_key])
            mapping.at[key,'atlas_description'] = rsi_scs_map[scs_key][0]
            mapping.at[key,'atlas_value'] = rsi_scs_map[scs_key][1]
            #print(key1, scs_fc[key1][1])
    elif 'rsi' in key:
        if not scs_key in scs_fc:
            #print(f'{scs_key} not in scs_fc')
            pass
        else:
            #print(scs_key, scs_fc[scs_key])
            mapping.at[key,'atlas_description'] = scs_fc[scs_key][0]
            mapping.at[key,'atlas_value'] = scs_fc[scs_key][1]
            #print(key1, scs_fc[key1][1])
    else:
        if not scs_key in subcort_map:
            #print(f'{scs_key} not in subcort_map')
            pass
        else:
            #print(var)
            mapping.at[key,'atlas_description'] = subcort_map[scs_key][0]
            mapping.at[key,'atlas_value'] = subcort_map[scs_key][1]


# In[26]:


len(mapping.filter(regex=f'dmri_rsi.*_scs_.*', axis=0).index) - len(mapping.filter(regex=f'dmri_rsi.*_scs_.*', axis=0).index.unique())


# In[27]:


# gordon parcellation
gordon_labels = pd.read_excel(join(PROJ_DIR, 'resources', 'gordon', 'Parcels.xlsx'),
                              header=0, 
                              index_col=0)


# In[28]:


cortgordon = mapping.filter(regex='.*_cortgordon', axis=0).index
for var in cortgordon:
    value = int(var.split('_')[-1][2:-2])
    mapping.at[var, 'atlas_value'] = value
    mapping.at[var, 'atlas_description'] = gordon_labels.loc[value]['Community']


# region_names = {'aal': ('Accumbens', 'L'), 
#                 'aalh': ('Accumbens', 'L'), 
#                 'aar': ('Accumbens', 'R'), 
#                 'aarh': ('Accumbens', 'R'), 
#                 'ablh': ('Accumbens', 'L'), 
#                 'abrh' ('Accumbens', 'R'),
#                 'ad': ('Auditory Network', 'B'), 
#                 'aglh': ('Amygdala', 'L'), 
#                 'agrh': ('Amygdala', 'R'), 
#                 'amygdalalh': ('Amugdala', 'L'), 
#                 'amygdalarh': ('Amygdala', 'R'), 
#                 'aomtmlh': (),
#                 'aomtmrh': (), 
#                 'atrlh', 
#                 'atrrh', 
#                 'banksstslh': ('Banks of Superior Temporal Sulcus', 'L'), 
#                 'banksstsrh': ('Banks of Superior Temporal Sulcus', 'R'),
#                 'brainstem': ('Brainstem', 'B'), 
#                 'bs': ('Brainstem', 'B'), 
#                 'bstem': ('Brainstem', 'B'), 
#                 'ca': ('Cinguloparietal Network', 'B'), 
#                 'caudatelh': ('Caudate', 'L'), 
#                 'caudaterh': ('Caudate', 'R'),
#                 'cbclh': ('Cerebellar Cortex', 'L'), 
#                 'cbcrh': ('Cerebellar Cortex', 'R'), 
#                 'cbwmlh': ('Cerebellar White Matter', 'L'), 
#                 'cbwmrh': ('Cerebellar White Matter', 'R'), 
#                 'cc': ('Corpus Callosum', 'B'), 
#                 'cdacatelh': ('Anterior Cingulate, Caudal', 'L'),
#                 'cdacaterh': ('Anterior Cingulate, Caudal', 'R'), 
#                 'cdaclatelh': ('Anterior Cingulate, Caudal', 'L'), 
#                 'cdaclaterh': ('Anterior Cingulate, Caudal', 'R'), 
#                 'cdelh': ('Caudate', 'L'), 
#                 'cderh': ('Caudate', 'R'), 
#                 'cdlh': ('Caudate', 'L'),
#                 'cdmdflh': ('Middle Frontal Gyrus, Caudal', 'L'), 
#                 'cdmdfrh': ('Middle Frontal Gyrus, Caudal', 'R'), 
#                 'cdmdfrlh': ('Middle Frontal Gyrus, Caudal', 'L'), 
#                 'cdmdfrrh': ('Middle Frontal Gyrus, Caudal', 'R'), 
#                 'cdrh': ('Caudate', 'R'), 
#                 'cgc': ('Cingulo-Opercular Network', 'B'),
#                 'cgclh': ('Cingulate Cingulum', 'L'), 
#                 'cgcrh': ('Cingulate Cingulum', 'R'), 
#                 'cghlh': ('Parahippocampal Cingulum', 'L'), 
#                 'cghrh': ('Parahippocampal Cingulum', 'R'),  
#                 'crbcortexlh': ('Cerebellar Cortex', 'L'),
#                 'crbcortexrh': ('Cerebellar Cortex', 'R'), 
#                 'crbwmatterlh': ('Cerebellar White Matter', 'L'), 
#                 'crbwmatterrh': ('Cerebellar White Matter', 'L'), 
#                 'crcxlh': ('Cerebellar Cortex', 'L'), 
#                 'crcxrh': ('Cerebellar Cortex', 'R'),
#                 'cstlh': ('Corticospinal Tract', 'L'), 
#                 'cstrh': ('Corticospinal Tract', 'R'), 
#                 'cuneuslh': ('Cuneus', 'L'), 
#                 'cuneusrh': ('Cuneus', 'R'), 
#                 'cwmlh': ('Cerebral White Matter', 'L'), 
#                 'cwmrh': ('Cerebral White Matter', 'L'), 
#                 'dla': ('Dorsal Attention Network', 'B'),
#                 'dlprefrlh': ('Dorsal Prefrontal Cortex', 'L'), 
#                 'dlprefrrh': ('Dorsal Prefrontal Cortex', 'R'), 
#                 'dmfrlh': ('Dorsomedial Frontal Cortex', 'L'), 
#                 'dmfrrh': ('Dorsomedial Frontal Cortex', 'R'), 
#                 'dt': ('Default Mode Network', 'B'), 
#                 'ehinallh': ('Entorhinal Cortex', 'L'),
#                 'ehinalrh': ('Entorhinal Cortex', 'R'), 
#                 'entorhinallh': ('Entorhinal Cortex', 'L'), 
#                 'entorhinalrh': ('Entorhinal Cortex', 'R'), 
#                 'fflh': ('Fusiform Gyrus', 'L'), 
#                 'ffrh': ('Fusiform Gyrus', 'R'), 
#                 'fmaj': ('Fornix Major', 'B'),
#                 'fmin': ('Fornix Minor', 'B'), 
#                 'fo': ('Frontoparietal Network', 'B'), 
#                 'fpolelh': ('Frontal Pole', 'L'), 
#                 'fpolerh': ('Frontal Pole', 'R'), 
#                 'frpolelh': ('Frontal Pole', 'L'), 
#                 'frpolerh': ('Frontal Pole', 'R'),
#                 'fscslh': ('Superior Corticostriate Tract (Frontal)', 'L'), 
#                 'fscsrh': ('Superior Corticostriate Tract (Frontal)', 'R'), 
#                 'fusiformlh': ('Fusiform Gyrus', 'L'), 
#                 'fusiformrh': ('Fusiform Gyrus', 'R'), 
#                 'fxcutlh': ('Fornix (excluding Fimbria)', 'L'),
#                 'fxcutrh': ('Fornix (excluding Fimbria)', 'R'), 
#                 'fxlh': ('Fornix', 'L'), 
#                 'fxrh': ('Fornix', 'R'), 
#                 'hclh': ('Hippocampus', 'L'),
#                 'hcrh': ('Hippocampus', 'R'), 
#                 'hplh': ('Hippocampus', 'L'), 
#                 'hprh': ('Hippocampus', 'R'), 
#                 'hpuslh': ('Hippocampus', 'L'), 
#                 'hpusrh': ('Hippocampus', 'R'), 
#                 'ifolh': ('Inferior Fronto-occipital Fasciculus', 'L'), 
#                 'iforh': ('Inferior Fronto-occipital Fasciculus', 'R'),
#                 'ifpalh': ('Inferior Parietal', 'L'), 
#                 'ifparh': ('Inferior Parietal', 'R'), 
#                 'ifpllh': ('Inferior Parietal', 'L'), 
#                 'ifplrh': ('Inferior Parietal', 'R'), 
#                 'ifsfclh', 
#                 'ifsfcrh',
#        'iftlh', 'iftmlh', 'iftmrh', 'iftrh', 'ihcatelh', 'ihcaterh',
#        'ihclatelh', 'ihclaterh', 'ilflh', 'ilfrh', 'ilvlh', 'ilvrh',
#        'insulalh', 'insularh', 'intracranialv', 'linguallh', 'lingualrh',
#        'lobfrlh', 'lobfrrh', 'loboflh', 'lobofrh', 'loccipitallh',
#        'loccipitalrh', 'locclh', 'loccrh', 'lvrh', 'mdtlh', 'mdtmlh',
#        'mdtmrh', 'mdtrh', 'mobfrlh', 'mobfrrh', 'moboflh', 'mobofrh', 'n',
#        'obfrlh', 'obfrrh', 'occlh', 'occrh', 'pallidumlh', 'pallidumrh',
#        'paracentrallh', 'paracentralrh', 'paracnlh', 'paracnrh',
#        'parahpallh', 'parahpalrh', 'parsobalislh', 'parsobalisrh',
#        'parsobislh', 'parsobisrh', 'parsopclh', 'parsopcrh', 'parsopllh',
#        'parsoplrh', 'parstgrislh', 'parstgrisrh', 'parstularislh',
#        'parstularisrh', 'pclh', 'pcrh', 'pericclh', 'periccrh', 'pllh',
#        'plrh', 'postcentrallh', 'postcentralrh', 'postcnlh', 'postcnrh',
#        'precentrallh', 'precentralrh', 'precnlh', 'precnrh',
#        'precuneuslh', 'precuneusrh', 'psclatelh', 'psclaterh', 'pscslh',
#        'pscsrh', 'pslflh', 'pslfrh', 'ptcatelh', 'ptcaterh', 'ptlh',
#        'ptoltmlh', 'ptoltmrh', 'ptrh', 'putamenlh', 'putamenrh',
#        'rlaclatelh', 'rlaclaterh', 'rlmdflh', 'rlmdfrh', 'rracatelh',
#        'rracaterh', 'rrmdfrlh', 'rrmdfrrh', 'rspltp', 'sa', 'scslh',
#        'scsrh', 'sifclh', 'sifcrh', 'slflh', 'slfrh', 'smh', 'smlh',
#        'smm', 'smrh', 'spetallh', 'spetalrh', 'suflh', 'sufrh', 'sufrlh',
#        'sufrrh', 'supllh', 'suplrh', 'sutlh', 'sutmlh', 'sutmrh', 'sutrh',
#        'thplh', 'thprh', 'tmpolelh', 'tmpolerh', 'total', 'tplh',
#        'tpolelh', 'tpolerh', 'tprh', 'trvtmlh', 'trvtmrh', 'tslflh',
#        'tslfrh', 'tvtlh', 'tvtrh', 'unclh', 'uncrh', 'vdclh', 'vdcrh',
#        'vedclh', 'vedcrh', 'ventraldclh', 'ventraldcrh', 'vs', 'vta',
#        'vtdclh', 'vtdcrh'}

# In[29]:


# skipping fuzzy12 because the atlases are only available upon email request
# and I doubt I'll use them for plotting anyway
# next up: fiber atlas
FIBER_DIR = '/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Outcomes/ABCD-brain outcomes/AtlasTrack/'
atlastrack_rois = join(FIBER_DIR, 'fibertract_visualization', 'ABCD Atlas Track ROIs')
fiber_key = pd.read_csv(join(FIBER_DIR, 'documentation', 'DTI_Fiber_Legend.csv'), 
                        header=0, 
                        index_col=0)


# In[30]:


fiber_key['FiberName'] = [name.lower() for name in fiber_key['FiberName']]


# In[31]:


for var in mapping.filter(regex='.*_fiber', axis=0).index:
    tract_name = var.split('_')[-1]
    hemisphere = tract_name[-2]
    if not 'h' in var:
        tract = tract_name
        index = fiber_key[fiber_key['FiberName'] == tract].index
    else:
        tract = tract_name[:-2]
        index = fiber_key[fiber_key['FiberName'] == f'{hemisphere}_{tract}'].index
        mapping.at[var, 'atlas_description'] = fiber_key.loc[index]['LongFiberName']
        mapping.at[var, 'atlas_value'] = int(index.values)
        mapping.at[var, 'atlas_fname'] = join(atlastrack_rois, f'fiber_{int(index.values)}_count_countatlas.nii.gz')


# In[32]:


for var in mapping.filter(regex='rsi.*_fib.*', axis=0).index:
    tract_name = var.split('_')[-1]
    hemisphere = tract_name[-2]
    if not 'h' in var:
        tract = tract_name
        index = fiber_key[fiber_key['FiberName'] == tract].index
    else:
        tract = tract_name[:-2]
        index = fiber_key[fiber_key['FiberName'] == f'{hemisphere}_{tract}'].index
        mapping.at[var, 'atlas_description'] = fiber_key.loc[index]['LongFiberName']
        mapping.at[var, 'atlas_value'] = int(index.values)
        mapping.at[var, 'atlas_fname'] = join(atlastrack_rois, f'fiber_{int(index.values)}_count_countatlas.nii.gz')


# In[39]:


mapping.filter(regex='.*_fiber', axis=0)


# In[34]:


vol_mapping = {'smri_vol_cdk_banksstslh.change_score': 1001.0,
    'smri_vol_cdk_cdacatelh.change_score': 1002.0,
    'smri_vol_cdk_cdmdfrlh.change_score': 1003.0,
    'smri_vol_cdk_cuneuslh.change_score': 1005.0,
    'smri_vol_cdk_ehinallh.change_score': 1006.0,
    'smri_vol_cdk_fusiformlh.change_score': 1007.0,
    'smri_vol_cdk_ifpllh.change_score': 1008.0,
    'smri_vol_cdk_iftmlh.change_score': 1009.0,
    'smri_vol_cdk_ihcatelh.change_score': 1010.0,
    'smri_vol_cdk_locclh.change_score': 1011.0,
    'smri_vol_cdk_lobfrlh.change_score': 1012.0,
    'smri_vol_cdk_linguallh.change_score': 1013.0,
    'smri_vol_cdk_mobfrlh.change_score': 1014.0,
    'smri_vol_cdk_mdtmlh.change_score': 1015.0,
    'smri_vol_cdk_parahpallh.change_score': 1016.0,
    'smri_vol_cdk_paracnlh.change_score': 1017.0,
    'smri_vol_cdk_parsopclh.change_score': 1018.0,
    'smri_vol_cdk_parsobislh.change_score': 1019.0,
    'smri_vol_cdk_parstgrislh.change_score': 1020.0,
    'smri_vol_cdk_pericclh.change_score': 1021.0,
    'smri_vol_cdk_postcnlh.change_score': 1022.0,
    'smri_vol_cdk_ptcatelh.change_score': 1023.0,
    'smri_vol_cdk_precnlh.change_score': 1024.0,
    'smri_vol_cdk_pclh.change_score': 1025.0,
    'smri_vol_cdk_rracatelh.change_score': 1026.0,
    'smri_vol_cdk_rrmdfrlh.change_score': 1027.0,
    'smri_vol_cdk_sufrlh.change_score': 1028.0,
    'smri_vol_cdk_supllh.change_score': 1029.0,
    'smri_vol_cdk_sutmlh.change_score': 1030.0,
    'smri_vol_cdk_smlh.change_score': 1031.0,
    'smri_vol_cdk_frpolelh.change_score': 1032.0,
    'smri_vol_cdk_tmpolelh.change_score': 1033.0,
    'smri_vol_cdk_trvtmlh.change_score': 1034.0,
    'smri_vol_cdk_insulalh.change_score': 1035.0,
    'smri_vol_cdk_banksstsrh.change_score': 2001.0,
    'smri_vol_cdk_cdacaterh.change_score': 2002.0,
    'smri_vol_cdk_cdmdfrrh.change_score': 2003.0,
    'smri_vol_cdk_cuneusrh.change_score': 2005.0,
    'smri_vol_cdk_ehinalrh.change_score': 2006.0,
    'smri_vol_cdk_fusiformrh.change_score': 2007.0,
    'smri_vol_cdk_ifplrh.change_score': 2008.0,
    'smri_vol_cdk_iftmrh.change_score': 2009.0,
    'smri_vol_cdk_ihcaterh.change_score': 2010.0,
    'smri_vol_cdk_loccrh.change_score': 2011.0,
    'smri_vol_cdk_lobfrrh.change_score': 2012.0,
    'smri_vol_cdk_lingualrh.change_score': 2013.0,
    'smri_vol_cdk_mobfrrh.change_score': 2014.0,
    'smri_vol_cdk_mdtmrh.change_score': 2015.0,
    'smri_vol_cdk_parahpalrh.change_score': 2016.0,
    'smri_vol_cdk_paracnrh.change_score': 2017.0,
    'smri_vol_cdk_parsopcrh.change_score': 2018.0,
    'smri_vol_cdk_parsobisrh.change_score': 2019.0,
    'smri_vol_cdk_parstgrisrh.change_score': 2020.0,
    'smri_vol_cdk_periccrh.change_score': 2021.0,
    'smri_vol_cdk_postcnrh.change_score': 2022.0,
    'smri_vol_cdk_ptcaterh.change_score': 2023.0,
    'smri_vol_cdk_precnrh.change_score': 2024.0,
    'smri_vol_cdk_pcrh.change_score': 2025.0,
    'smri_vol_cdk_rracaterh.change_score': 2026.0,
    'smri_vol_cdk_rrmdfrrh.change_score': 2027.0,
    'smri_vol_cdk_sufrrh.change_score': 2028.0,
    'smri_vol_cdk_suplrh.change_score': 2029.0,
    'smri_vol_cdk_sutmrh.change_score': 2030.0,
    'smri_vol_cdk_smrh.change_score': 2031.0,
    'smri_vol_cdk_frpolerh.change_score': 2032.0,
    'smri_vol_cdk_tmpolerh.change_score': 2033.0,
    'smri_vol_cdk_trvtmrh.change_score': 2034.0,
    'smri_vol_cdk_insularh.change_score': 2035.0,}


# In[35]:


for var in vol_mapping.keys():
    variable = var.split('.')[0]
    mapping.at[variable,'atlas_value'] = vol_mapping[var]


# In[36]:


mapping.to_csv(join(PROJ_DIR, DATA_DIR, 'variable_to_nifti_mapping.csv'))


# In[37]:


mapping


# In[38]:


mapping.loc['dmri_rsirnd_scs_vdcrh']


# In[ ]:




