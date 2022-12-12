import pandas as pd
import numpy as np
from os.path import join
from nilearn import plotting, surface
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def jili_sidak_mc(data, alpha):
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
    print('Number of effective comparisons: {0}'.format(M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff

def plot_surfaces(nifti, surf, cmap, vmax, threshold):
    '''
    Plots of medial and lateral left and right surface views from nifti volume
    '''
    
    texture_l = surface.vol_to_surf(nifti, surf.pial_left, interpolation='nearest')
    texture_r = surface.vol_to_surf(nifti, surf.pial_right, interpolation='nearest')
    
    fig = plt.figure(figsize=(12,4))
    gs = GridSpec(1, 4)

    ax0 = fig.add_subplot(gs[0], projection='3d')
    ax1 = fig.add_subplot(gs[1], projection='3d')
    ax2 = fig.add_subplot(gs[2], projection='3d')
    ax3 = fig.add_subplot(gs[3], projection='3d')
    plt.tight_layout(w_pad=-1, h_pad=-1)
    figure = plotting.plot_surf_stat_map(surf.pial_left, 
                                         texture_l, 
                                         symmetric_cbar=False, 
                                         threshold=threshold,
                                         cmap=cmap, 
                                         view='lateral', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax0)
    figure = plotting.plot_surf_stat_map(surf.pial_left, 
                                         texture_l, 
                                         symmetric_cbar=False, 
                                         threshold=threshold,     
                                         cmap=cmap, 
                                         view='medial', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax1)
    figure = plotting.plot_surf_stat_map(surf.pial_right, 
                                         texture_r, 
                                         symmetric_cbar=False, 
                                         threshold=threshold,
                                         cmap=cmap, 
                                         view='lateral', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax2)
    figure = plotting.plot_surf_stat_map(surf.pial_right, 
                                         texture_r, 
                                         symmetric_cbar=False, 
                                         threshold=threshold,     
                                         cmap=cmap, 
                                         view='medial', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax3)
    return figure

def assign_region_names(df, missing=False):
    '''
    Input: 
    df = dataframe (variable x columns) with column containing region names in ABCD var ontology, 
    Output: 
    df = same dataframe, but with column mapping region variables to actual region names
    missing = optional, list of ABCD region names not present in region_names dictionary
    '''
    if not 'region' in df.columns:
        for var in df.index:
            #print(var)
            if 'mrisdp' in var:
                var_num = int(var.split('.')[0].split('_')[-1])
                df.at[var, 'modality'] = 'smri'
                df.at[var, 'atlas'] = 'dtx'
                if var_num <= 148:
                    df.at[var, 'measure'] = 'thick'
                elif var_num <= 450 and var_num >= 303:
                    df.at[var, 'measure'] = 'area'
                elif var_num < 604 and var_num >= 450:
                    df.at[var, 'measure'] = 'vol'
                elif var_num <= 1054 and var_num >= 907:
                    df.at[var, 'measure'] = 't1wcnt'
                elif var_num == 604:
                    df.at[var, 'measure'] = 'gmvol'
            elif '_' in var:
                var_list = var.split('.')[0].split('_')
                df.at[var, 'modality'] = var_list[0]
                df.at[var, 'measure'] = var_list[1]
                df.at[var, 'atlas'] = var_list[2]
                df.at[var, 'region'] = '_'.join(var_list[3:])

        df = df[df['measure'] != 't1w']
        df = df[df['measure'] != 't2w']
    region_names = {'aal': ('Accumbens', 'L'), 
                    'aalh': ('Accumbens', 'L'), 
                    'aar': ('Accumbens', 'R'), 
                    'aarh': ('Accumbens', 'R'), 
                    'ablh': ('Accumbens', 'L'), 
                    'abrh': ('Accumbens', 'R'),
                    'ad': ('Auditory Network', 'B'), 
                    'aglh': ('Amygdala', 'L'), 
                    'agrh': ('Amygdala', 'R'), 
                    'amygdalalh': ('Amugdala', 'L'), 
                    'amygdalarh': ('Amygdala', 'R'), 
                    'aomtmlh': (),
                    'aomtmrh': (), 
                    'atrlh': ('Anterior Thalamic Radiation', 'L'), 
                    'atrrh': ('Anterior Thalamic Radiation', 'R'),
                    'bstslh': ('Banks of Superior Temporal Sulcus', 'L'),
                    'bstsrh': ('Banks of Superior Temporal Sulcus', 'R'),
                    'banksstslh': ('Banks of Superior Temporal Sulcus', 'L'), 
                    'banksstsrh': ('Banks of Superior Temporal Sulcus', 'R'),
                    'brainstem': ('Brainstem', 'B'), 
                    'bs': ('Brainstem', 'B'), 
                    'bstem': ('Brainstem', 'B'), 
                    'ca': ('Cinguloparietal Network', 'B'), 
                    'caclh': ('Cingulate Gyrus, Caudal Anterior', 'L'),
                    'cacrh': ('Cingulate Gyrus, Caudal Anterior', 'L'),
                    'caudatelh': ('Caudate', 'L'), 
                    'caudaterh': ('Caudate', 'R'),
                    'cbclh': ('Cerebellar Cortex', 'L'), 
                    'cbcrh': ('Cerebellar Cortex', 'R'), 
                    'cbwmlh': ('Cerebellar White Matter', 'L'), 
                    'cbwmrh': ('Cerebellar White Matter', 'R'), 
                    'cc': ('Corpus Callosum', 'B'), 
                    'cdacatelh': ('Anterior Cingulate, Caudal', 'L'),
                    'cdacaterh': ('Anterior Cingulate, Caudal', 'R'), 
                    'cdaclatelh': ('Anterior Cingulate, Caudal', 'L'), 
                    'cdaclaterh': ('Anterior Cingulate, Caudal', 'R'), 
                    'cdelh': ('Caudate', 'L'), 
                    'cderh': ('Caudate', 'R'), 
                    'cdlh': ('Caudate', 'L'),
                    'cdmdflh': ('Middle Frontal Gyrus, Caudal', 'L'), 
                    'cdmdfrh': ('Middle Frontal Gyrus, Caudal', 'R'), 
                    'cdmdfrlh': ('Middle Frontal Gyrus, Caudal', 'L'), 
                    'cdmdfrrh': ('Middle Frontal Gyrus, Caudal', 'R'), 
                    'cdrh': ('Caudate', 'R'), 
                    'cgc': ('Cingulo-Opercular Network', 'B'),
                    'cgclh': ('Cingulate Cingulum', 'L'), 
                    'cgcrh': ('Cingulate Cingulum', 'R'), 
                    'cghlh': ('Parahippocampal Cingulum', 'L'), 
                    'cghrh': ('Parahippocampal Cingulum', 'R'),
                    'cmflh': ('Middle Frontal Gyrus, Caudal', 'L'),
                    'cmfrm': ('Middle Frontal Gyrus, Caudal', 'R'),
                    'crbcortexlh': ('Cerebellar Cortex', 'L'),
                    'crbcortexrh': ('Cerebellar Cortex', 'R'), 
                    'crbwmatterlh': ('Cerebellar White Matter', 'L'), 
                    'crbwmatterrh': ('Cerebellar White Matter', 'L'), 
                    'crcxlh': ('Cerebellar Cortex', 'L'), 
                    'crcxrh': ('Cerebellar Cortex', 'R'),
                    'cstlh': ('Corticospinal Tract', 'L'), 
                    'cstrh': ('Corticospinal Tract', 'R'), 
                    'cuneuslh': ('Cuneus', 'L'), 
                    'cuneusrh': ('Cuneus', 'R'), 
                    'cwmlh': ('Cerebral White Matter', 'L'), 
                    'cwmrh': ('Cerebral White Matter', 'L'), 
                    'dla': ('Dorsal Attention Network', 'B'),
                    'dlprefrlh': ('Dorsal Prefrontal Cortex', 'L'), 
                    'dlprefrrh': ('Dorsal Prefrontal Cortex', 'R'), 
                    'dmfrlh': ('Dorsomedial Frontal Cortex', 'L'), 
                    'dmfrrh': ('Dorsomedial Frontal Cortex', 'R'), 
                    'dt': ('Default Mode Network', 'B'), 
                    'df': ('Default Mode Network', 'B'), 
                    'ehinallh': ('Entorhinal Cortex', 'L'),
                    'ehinalrh': ('Entorhinal Cortex', 'R'), 
                    'entorhinallh': ('Entorhinal Cortex', 'L'), 
                    'entorhinalrh': ('Entorhinal Cortex', 'R'), 
                    'fflh': ('Fusiform Gyrus', 'L'), 
                    'ffrh': ('Fusiform Gyrus', 'R'), 
                    'fmaj': ('Fornix Major', 'B'),
                    'fmin': ('Fornix Minor', 'B'), 
                    'fo': ('Frontoparietal Network', 'B'), 
                    'fpolelh': ('Frontal Pole', 'L'), 
                    'fpolerh': ('Frontal Pole', 'R'), 
                    'frpolelh': ('Frontal Pole', 'L'), 
                    'frpolerh': ('Frontal Pole', 'R'),
                    'fscslh': ('Superior Corticostriate Tract (Frontal)', 'L'), 
                    'fscsrh': ('Superior Corticostriate Tract (Frontal)', 'R'), 
                    'fusiformlh': ('Fusiform Gyrus', 'L'), 
                    'fusiformrh': ('Fusiform Gyrus', 'R'), 
                    'fxcutlh': ('Fornix (excluding Fimbria)', 'L'),
                    'fxcutrh': ('Fornix (excluding Fimbria)', 'R'), 
                    'fxlh': ('Fornix', 'L'), 
                    'fxrh': ('Fornix', 'R'), 
                    'hclh': ('Hippocampus', 'L'),
                    'hcrh': ('Hippocampus', 'R'), 
                    'hplh': ('Hippocampus', 'L'), 
                    'hprh': ('Hippocampus', 'R'), 
                    'hpuslh': ('Hippocampus', 'L'), 
                    'hpusrh': ('Hippocampus', 'R'), 
                    'ifolh': ('Inferior Fronto-occipital Fasciculus', 'L'), 
                    'iforh': ('Inferior Fronto-occipital Fasciculus', 'R'),
                    'ifpalh': ('Inferior Parietal', 'L'), 
                    'ifparh': ('Inferior Parietal', 'R'), 
                    'ifpllh': ('Inferior Parietal', 'L'), 
                    'ifplrh': ('Inferior Parietal', 'R'), 
                    'ifsfclh': ('Inferior Frontal Superior Frontal', 'L'), 
                    'ifsfcrh': ('Inferior Frontal Superior Frontal', 'R'),
                    'iftlh': ('Inferior Temporal', 'L'), 
                    'iftmlh': ('Inferior Temporal', 'L'), 
                    'iftmrh': ('Inferior Temporal', 'R'), 
                    'iftrh': ('Inferior Temporal', 'R'), 
                    'ihcatelh': ('Cingulate Gyrus, Ithsmus', 'L'), 
                    'ihcaterh': ('Cingulate Gyrus, Ithsmus', 'R'),
                    'ihclatelh': ('Cingulate Gyrus, Ithsmus', 'L'), 
                    'ihclaterh': ('Cingulate Gyrus, Ithsmus', 'R'), 
                    'ilflh': ('Inferior Longitudinal Fasiculus', 'L'), 
                    'ilfrh': ('Inferior Longitudinal Fasiculus', 'R'), 
                    'ilvlh': ('Inferior Lateral Ventricle', 'L'), 
                    'ilvrh': ('Inferior Lateral Ventricle', 'R'),
                    'insulalh': ('Insula', 'L'), 
                    'insularh': ('Insula', 'R'), 
                    'intracranialv': ('Intracranial Volume', 'B'), 
                    'linguallh': ('Lingual Gyrus', 'L'), 
                    'lingualrh': ('Lingual Gyrus', 'R'),
                    'lobfrlh': ('Orbitofrontal Gyrus, Lateral', 'L'), 
                    'lobfrrh': ('Orbitofrontal Gyrus, Lateral', 'R'), 
                    'loboflh': ('Orbitofrontal Gyrus, Lateral', 'L'), 
                    'lobofrh': ('Orbitofrontal Gyrus, Lateral', 'R'), 
                    'loccipitallh': ('Occipital Gyrus, Lateral', 'L'),
                    'loccipitalrh': ('Occipital Gyrus, Lateral', 'R'), 
                    'locclh': ('Occipital Gyrus, Lateral', 'L'), 
                    'loccrh': ('Occipital Gyrus, Lateral', 'R'), 
                    'lvrh': ('Lateral Ventricle', 'R'), 
                    'mdtlh': ('Middle Temporal Gyrus', 'L'), 
                    'mdtmlh': ('Middle Temporal Gyrus', 'L'),
                    'mdtmrh': ('Middle Temporal Gyrus', 'R'), 
                    'mdtrh': ('Middle Temporal Gyrus', 'R'), 
                    'mobfrlh': ('Occipital Gyrus, Medial', 'L'), 
                    'mobfrrh': ('Occipital Gyrus, Medial', 'R'), 
                    'moboflh': ('Occipital Gyrus, Medial', 'L'), 
                    'mobofrh': ('Occipital Gyrus, Medial', 'R'), 
                    'n': ('Extra-Network', 'B'),
                    'pallidumlh': ('Pallidum', 'L'), 
                    'pallidumrh': ('Pallidum', 'R'),
                    'paracentrallh': ('Paracentral Gyrus', 'L'), 
                    'paracentralrh': ('Paracentral Gyrus', 'R'), 
                    'paracnlh': ('Paracentral Gyrus', 'L'), 
                    'paracnrh': ('Paracentral Gyrus', 'R'),
                    'parahpallh': ('Parahippocampal Gyrus', 'L'), 
                    'parahpalrh': ('Parahippocampal Gyrus', 'R'), 
                    'parsobalislh': ('Inferior Frontal Gyrus, Pars Orbitalis', 'L'), 
                    'parsobalisrh': ('Inferior Frontal Gyrus, Pars Orbitalis', 'R'),
                    'parsobislh': ('Inferior Frontal Gyrus, Pars Orbitalis', 'L'), 
                    'parsobisrh': ('Inferior Frontal Gyrus, Pars Orbitalis', 'R'), 
                    'parsopclh': ('Inferior Frontal Gyrus, Pars Opercularis', 'L'), 
                    'parsopcrh': ('Inferior Frontal Gyrus, Pars Opercularis', 'R'), 
                    'parsopllh': ('Inferior Frontal Gyrus, Pars Opercularis', 'L'),
                    'parsoplrh': ('Inferior Frontal Gyrus, Pars Opercularis', 'R'), 
                    'parstgrislh': ('Inferior Frontal Gyrus, Pars Triangularis', 'L'), 
                    'parstgrisrh': ('Inferior Frontal Gyrus, Pars Triangularis', 'R'), 
                    'parstularislh': ('Inferior Frontal Gyrus, Pars Triangularis', 'L'),
                    'parstularisrh': ('Inferior Frontal Gyrus, Pars Triangularis', 'R'), 
                    'pclh': ('Precuneus', 'L'), 
                    'pcrh': ('Precuneus', 'R'), 
                    'pericclh': ('Pericalcarine Cortex', 'L'), 
                    'periccrh': ('Pericalcarine Cortex', 'L'), 
                    'pllh': ('Pallidum', 'L'),
                    'plrh': ('Pallidum', 'L'), 
                    'postcentrallh': ('Postcentral Gyrus', 'L'), 
                    'postcentralrh': ('Postcentral Gyrus', 'R'), 
                    'postcnlh': ('Postcentral Gyrus', 'L'), 
                    'postcnrh': ('Postcentral Gyrus', 'R'),
                    'precentrallh': ('Precentral Gyrus', 'L'), 
                    'precentralrh': ('Precentral Gyrus', 'R'), 
                    'precnlh': ('Precentral Gyrus', 'L'), 
                    'precnrh': ('Precentral Gyrus', 'R'),
                    'precuneuslh': ('Precuneus', 'L'), 
                    'precuneusrh': ('Precuneus', 'L'), 
                    'psclatelh': ('Cingulate Gyrus, Posterior', 'L'), 
                    'psclaterh': ('Cingulate Gyrus, Posterior', 'R'), 
                    'pscslh': ('Superior Corticostriate Tract, Parietal', 'L'),
                    'pscsrh': ('Superior Corticostriate Tract, Parietal', 'R'), 
                    'pslflh': ('Superior Longitudinal Fasiculus, Parietal', 'L'), 
                    'pslfrh': ('Superior Longitudinal Fasiculus, Parietal', 'R'), 
                    'ptcatelh': ('Cingulate Gyrus, Posterior', 'L'), 
                    'ptcaterh': ('Cingulate Gyrus, Posterior', 'R'), 
                    'ptlh': ('Putamen', 'L'),
                    'ptoltmlh': (), 
                    'ptoltmrh': (), 
                    'ptrh': ('Putamen', 'R'), 
                    'putamenlh': ('Putamen', 'L'), 
                    'putamenrh': ('Putamen', 'R'),
                    'rlaclatelh': ('Cingulate Gyrus, Rostral Anterior', 'L'), 
                    'rlaclaterh': ('Cingulate Gyrus, Rostral Anterior', 'R'), 
                    'rlmdflh': ('Middle Frontal Gyrus, Rostral', 'L'), 
                    'rlmdfrh': ('Middle Frontal Gyrus, Rostral', 'R'), 
                    'rracatelh': ('Cingulate Gyrus, Rostral Anterior', 'L'),
                    'rracaterh': ('Cingulate Gyrus, Rostral Anterior', 'R'), 
                    'rrmdfrlh': ('Middle Frontal Gyrus, Rostral', 'L'), 
                    'rrmdfrrh': ('Middle Frontal Gyrus, Rostral', 'R'), 
                    'rspltp': ('Retrosplenial Temporal Network', 'B'), 
                    'sa': ('Salience Network', 'B'), 
                    'scslh': ('Superior Corticostriate Tract', 'L'),
                    'scsrh': ('Superior Corticostriate Tract', 'L'), 
                    'sifclh': ('Striatum, Inferior Frontal', 'L'), 
                    'sifcrh': ('Striatum, Inferior Frontal', 'L'), 
                    'slflh': ('Superior Longitudinal Fasiculus', 'L'), 
                    'slfrh': ('Superior Longitudinal Fasiculus', 'R'), 
                    'smh': ('Sensorimotor Network, Hand', 'B'), 
                    'smlh': ('Supramarginal Gyrus', 'L'),
                    'smm': ('Sensorimotor Network, Mouth', 'B'), 
                    'smrh': ('Supramarginal Gyrus', 'R'), 
                    'spetallh': ('Superior Parietal Lobule', 'L'), 
                    'spetalrh': ('Superior Parietal Lobule', 'R'), 
                    'suflh': ('Superior Frontal Gyrus', 'L'), 
                    'sufrh': ('Superior Frontal Gyrus', 'R'), 
                    'sufrlh': ('Superior Frontal Gyrus', 'L'),
                    'sufrrh': ('Superior Frontal Gyrus', 'R'), 
                    'supllh': ('Superior Parietal Lobule', 'L'), 
                    'suplrh': ('Superior Parietal Lobule', 'R'), 
                    'sutlh': ('Superior Temporal Gyrus', 'L'), 
                    'sutmlh': ('Superior Temporal Gyrus', 'L'), 
                    'sutmrh': ('Superior Temporal Gyrus', 'R'), 
                    'sutrh': ('Superior Temporal Gyrus', 'R'),
                    'thplh': ('Thalamus', 'L'), 
                    'thprh': ('Thalamus', 'R'), 
                    'tmpolelh': ('Temporal Pole', 'L'), 
                    'tmpolerh': ('Temporal Pole', 'R'), 
                    'total': (), 
                    'tplh': ('Thalamus', 'L'),
                    'tpolelh': ('Temporal Pole', 'L'), 
                    'tpolerh': ('Temporal Pole', 'R'), 
                    'tprh': ('Thalamus', 'R'), 
                    'trvtmlh': ('Transverse Temporal Gyrus', 'L'), 
                    'trvtmrh': ('Transverse Temporal Gyrus', 'R'), 
                    'tslflh': ('Superior Longitudinal Fasiculus, Temporal', 'L'),
                    'tslfrh': ('Superior Longitudinal Fasiculus, Temporal', 'R'), 
                    'tvtlh': ('Transverse Temporal Gyrus', 'L'), 
                    'tvtrh': ('Transverse Temporal Gyrus', 'R'), 
                    'unclh': ('Uncinate Fasiculus', 'L'), 
                    'uncrh': ('Uncinate Fasiculus', 'R'), 
                    'vdclh': ('Ventral Diencephalon', 'L'), 
                    'vdcrh': ('Ventral Diencephalon', 'R'),
                    'vedclh': ('Ventral Diencephalon', 'L'), 
                    'vedcrh': ('Ventral Diencephalon', 'R'), 
                    'ventraldclh': ('Ventral Diencephalon', 'L'), 
                    'ventraldcrh': ('Ventral Diencephalon', 'R'), 
                    'vs': ('Visual Network', 'B'), 
                    'vta': ('Ventral Attention Network', 'B'),
                    'vtdclh': ('Ventral Diencephalon', 'L'), 
                    'vtdcrh': ('Ventral Diencephalon', 'R'),
                    'ad_ngd_ad': ('Auditory Network, Within', 'B'),
                    'ca': ('Cinguloparietal Network', 'B'),
                    'cerc': ('Cingulo-opercular Network', 'B'),
                    'cgc': ('Cingulo-opercular Network', 'B'),
                    'cmfrh': ('Middle Frontal Gyrus, Caudal', 'R'),
                    'cnlh': ('Cuneus', 'L'),
                    'cnrh': ('Cuneus', 'R'),
                    'copa': ('Cinguloparietal Network', 'B'),
                    'pcclh': ('Pericalcarine', 'L'),
                    'pccrh': ('Pericalcarine', 'R'),
                    'pcglh': ('Cingulate Gyrus, Posterior', 'L'),
                    'pcgrh': ('Cingulate Gyrus, Posterior', 'R'),
                    'pctlh': ('Postcentral Gyrus', 'L'),
                    'pctrh': ('Postcentral Gyrus', 'R'),
                    'phlh': ('Parahippocampal Gyrus', 'L'),
                    'phrh': ('Parahippocampal Gyrus', 'R'),
                    'poblh': ('Inferior Frontal Gyrus, Pars Orbitalis', 'L'),
                    'pobrh': ('Inferior Frontal Gyrus, Pars Orbitalis', 'R'),
                    'poplh': ('Inferior Frontal Gyrus, Pars Opercularis', 'L'),
                    'poprh': ('Inferior Frontal Gyrus, Pars Opercularis', 'R'),
                    'prcnlh': ('Precuneus', 'L'),
                    'prcnrh': ('Precuneus', 'R'),
                    'prctlh': ('Precentral Gyrus', 'L'),
                    'prctrh': ('Precentral Gyrus', 'R'),
                    'ptglh': ('Inferior Frontal Gyrus, Pars Triangularis', 'L'),
                    'ptgrh': ('Inferior Frontal Gyrus, Pars Triangularis', 'R'),
                    'raclh': ('Cingulate Gyrus, Rostral Anterior', 'L'),
                    'racrh': ('Cingulate Gyrus, Rostral Anterior', 'R'),
                    'rmflh': ('Middle Frontal Gyrus, Rostral', 'L'),
                    'rmfrh': ('Middle Frontal Gyrus, Rostral', 'R'),
                    'sflh': ('Superior Frontal Gyrus', 'L'),
                    'sfrh': ('Superior Frontal Gyrus', 'R'),
                    'splh': ('Superior Parietal Lobule', 'L'),
                    'sprh': ('Superior Parietal Lobule', 'R'),
                    'stlh': ('Superior Temporal Lobule', 'L'),
                    'strh': ('Superior Temporal Lobule', 'R'),
                    'ttlh': ('Transverse Temporal Lobe', 'L'),
                    'ttrh': ('Transverse Temporal Lobe', 'R'),
                    'erlh': ('Entorhinal Cortex', 'L'),
                    'errh': ('Entorhinal Cortex', 'R'),'fplh': ('Frontal Pole', 'L'),
                    'fprh': ('Frontal Pole', 'R'),
                    'au': ('Auditory Network', 'B'),
                    'dsa': ('Dorsal Attention Network', 'B'),
                    'fopa': ('Frontoparietal Network', 'B'),
                    'none': ('Extra-Network', 'B'),
                    'rst': ('Retrosplenial Temporal Network', 'B'),
                    'iclh': ('Cingulate Gyrus, Ithmus', 'L'),
                    'icrh': ('Cingulate Gyrus, Ithmus', 'R'),
                    'iplh': ('Inferior Parietal Lobule', 'L'),
                    'iprh': ('Inferior Parietal Lobule', 'R'),
                    'islh': ('Insula', 'L'),
                    'isrh': ('Insula', 'R'),
                    'itlh': ('Inferior Temporal Gyrus', 'L'),
                    'itrh': ('Inferior Temporal Gyrus', 'R'),
                    'lglh': ('Lingual Gyrus', 'L'),
                    'lgrh': ('Lingual Gyrus', 'R'),
                    'loflh': ('Orbitofrontal Gyrus, Lateral', 'L'),
                    'lofrh': ('Orbitofrontal Gyrus, Lateral', 'R'),
                    'lolh': ('Lateral Occipital Gyrus', 'L'),
                    'lorh': ('Lateral Occipital Gyrus', 'R'),
                    'moflh': ('Orbitofrontal Gyrus, Medial', 'L'),
                    'mofrh': ('Orbitofrontal Gyrus, Medial', 'R'),
                    'mtlh': ('Middle Temporal Gyrus', 'L'),
                    'mtrh': ('Middle Temporal Gyrus', 'R'),}
    missing = []
    for i in df.index: 
        if '_scs_' in df.loc[i]['region']:
            temp = df.loc[i]['region'].split('_scs_')
            region_name = f'{region_names[temp[0]][0]}, {region_names[temp[1]][0]}'
            hemisphere = region_names[temp[1]][1]
            df.at[i, 'long_region'] = region_name
            df.at[i, 'hemisphere'] = hemisphere
        elif '_ngd_' in df.loc[i]['region']:
            temp = df.loc[i]['region'].split('_ngd_')
            region_name = f'{region_names[temp[0]][0]}, {region_names[temp[1]][0]}'
            hemisphere = region_names[temp[1]][1]
            df.at[i, 'long_region'] = region_name
            df.at[i, 'hemisphere'] = hemisphere
        elif df.loc[i]['region'] not in region_names.keys():
            missing.append(df.loc[i]['region'])
        else:
            long_region = region_names[df.loc[i]['region']]
            df.at[i, 'long_region'] = long_region[0]
            df.at[i, 'hemisphere'] = long_region[1]
    if missing == True:
        return df, missing
    else:
        return df