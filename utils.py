import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
from os.path import join
from nilearn import plotting, surface, datasets
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

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
                    'cghlh': ('Parahippocam Cingulum', 'L'), 
                    'cghrh': ('Parahippocam Cingulum', 'R'),
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
                    'ifh': ('Inferior Parietal', 'L'), 
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
                    'lidumlh': ('lidum', 'L'), 
                    'lidumrh': ('lidum', 'R'),
                    'paracentrallh': ('Paracentral Gyrus', 'L'), 
                    'paracentralrh': ('Paracentral Gyrus', 'R'), 
                    'paracnlh': ('Paracentral Gyrus', 'L'), 
                    'paracnrh': ('Paracentral Gyrus', 'R'),
                    'parahlh': ('Parahippocam Gyrus', 'L'), 
                    'parahrh': ('Parahippocam Gyrus', 'R'), 
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
                    'pllh': ('lidum', 'L'),
                    'plrh': ('lidum', 'L'), 
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
                    'phlh': ('Parahippocam Gyrus', 'L'),
                    'phrh': ('Parahippocam Gyrus', 'R'),
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
    if not 'long_region' in df.columns:
        df['measure'] = ''
        df['region'] = ''
        df['modality'] = ''
        df['atlas'] = ''
        df['long_region'] = ''
        df['hemisphere'] = ''
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
                region = '_'.join(var_list[3:])
                df.at[var, 'region'] = region
                if 'scs' in var:
                    if 'rsirni' in var:
                        df.at[var, 'measure'] = 'rsirnigm'
                    elif 'rsirnd' in var:
                        df.at[var, 'measure'] = 'rsirndgm'
                    else:
                        pass
                else:
                    pass
                if '_scs_' in region:
                    temp = region.split('_scs_')
                    region_name = f'{region_names[temp[0]][0]}, {region_names[temp[1]][0]}'
                    hemisphere = region_names[temp[1]][1]
                    df.at[var, 'long_region'] = region_name
                    df.at[var, 'hemisphere'] = hemisphere
                    df.at[var, 'measure'] = 'subcortical-network fc'
                elif '_ngd_' in region:
                    temp = region.split('_ngd_')
                    if temp[0] == temp[1]:
                        df.at[var, 'measure'] = 'within-network fc'
                    else:
                        df.at[var, 'measure'] = 'between-network fc'
                    region_name = f'{region_names[temp[0]][0]}, {region_names[temp[1]][0]}'
                    hemisphere = region_names[temp[1]][1]
                    df.at[var, 'long_region'] = region_name
                    df.at[var, 'hemisphere'] = hemisphere
                elif str(region) not in (region_names.keys()):
                    missing.append(region)
                else:
                    long_region = region_names[region]
                    df.at[var, 'long_region'] = long_region[0]
                    df.at[var, 'hemisphere'] = long_region[1]

        df = df[df['measure'] != 't1w']
        df = df[df['measure'] != 't2w']
    else:
        pass
    
    print(f'missed {len(missing)} regions bc they weren\'t in the dict')
    return df

def plot_brains(series, out_dir):
    nifti_mapping = pd.read_csv('/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/data/variable_to_nifti_mapping.csv', 
                                header=0, 
                                index_col=0)

    series.index = [x.split('.')[0] for x in series.index]
    vmax = series.quantile(0.85)
    #vmin = series.quantile(0.25)

    # list of measures to plot
    measures = {'cortical-thickness': 'smri_thick_cdk_*',
                'cortical-gwcontrast': 'smri_t1wcnt_cdk_*',
                'cortical-area': 'smri_area_cdk_.*',
                'cortical-volume': 'smri_vol_cdk_.*', 
                'subcortical-volume': 'smri_vol_scs_.*', 
                'subcortical-RND': 'dmri_rsirnd_scs_.*',
                'subcortical-RNI': 'dmri_rsirni_scs_.*',
                'cortical-RND': 'dmri_rsirndgm_.*',
                'cortical-RNI': 'dmri_rsirnigm_.*',
                'cortical-BOLD-variance': 'rsfmri_var_cdk_.*',
                'tract-volume': 'dmri_dtivol_fiberat_.*', 
                'tract-FA': 'dmri_dtifa_fiberat_.*', 
                'tract-MD': 'dmri_dtimd_fiberat_.*',
                'tract-LD': 'dmri_dtild_fiberat_.*', 
                'tract-TD': 'dmri_dtitd_fiberat_.*', 
                'tract-RND': 'dmri_rsirnd_fib_.*',
                'tract-RNI': 'dmri_rsirni_fib_.*'}
    fc_cort_var = series.filter(regex='.*fmri.*_c_.*').index
    fc_scor_var = series.filter(regex='.*fmri.*_cor_.*').index
    fmri_var_var = series.filter(regex='.*fmri.*_var_.*').index

    #morph_var = df[df['concept'] == 'macrostructure'].index
    #cell_var = df[df['concept'] == 'microstructure'].index
    func_var = list(fmri_var_var) 
    conn_var = list(fc_cort_var) + list(fc_scor_var)

    conn_measures = {'cortical-network-connectivity': 'rsfmri_c_ngd_.*',
                'subcortical-network-connectivity': 'rsfmri_cor_ngd_.*_scs_.*',}

    fsaverage = datasets.fetch_surf_fsaverage()

    cmap = sns.diverging_palette(22, 256.3, s=70, l=50, center="light", as_cmap=True)
    pal = sns.diverging_palette(22, 256.3, s=70, l=50, center="light", as_cmap=False)
    #sns.plot(morph_ + func_ + cell_)

    pals = {'cortical-thickness': cmap,
            'cortical-gwcontrast': cmap,
                'cortical-area': cmap,
                'cortical-volume': cmap, 
                'subcortical-volume': cmap, 
                'subcortical-RND': cmap,
                'subcortical-RNI': cmap,
                'cortical-RND': cmap,
                'cortical-RNI': cmap,
                'cortical-BOLD-variance': cmap,
                'tract-volume': cmap, 
                'tract-FA': cmap, 
                'tract-MD': cmap,
                'tract-LD': cmap, 
                'tract-TD': cmap,
                'tract-RND': cmap,
                'tract-RNI': cmap,
            'cortical-network-connectivity': cmap,
                'subcortical-network-connectivity': cmap}

    # let's plot APC on brains pls
    for measure in measures.keys():
        #print(measure, measures[measure])
        #print(measure)

        meas_df = series.filter(regex=measures[measure], axis=0)
        meas_vars = meas_df.index

        meas_df.drop_duplicates(inplace=True)

        if meas_df.sum() == 0:
            pass
        else:
            if 'tract' in measure:
                fibers = nifti_mapping.filter(regex=measures[measure], axis=0).index
                var = fibers[0]
                tract_fname = nifti_mapping.loc[var]['atlas_fname']
                tract_nii = nib.load(tract_fname)
                tract_arr = tract_nii.get_fdata()
                #print(np.unique(tract_arr))
                avg = series.loc[f'{var}']
                tract_arr *= avg
                all_tracts_arr = np.zeros(tract_arr.shape)
                all_tracts_arr += tract_arr
                for var in fibers[1:]:    
                    tract_fname = nifti_mapping.loc[var]['atlas_fname']
                    if type(tract_fname) is str:
                        try:
                            tract_nii = nib.load(tract_fname)
                            tract_arr = tract_nii.get_fdata()
                            #print(np.unique(tract_arr))
                            avg = series.loc[f'{var}']
                            tract_arr *= avg
                            all_tracts_arr += tract_arr
                        except Exception as e:
                            pass
                    else:
                        pass
                meas_nimg = nib.Nifti1Image(all_tracts_arr, tract_nii.affine)
                #plt.figure(layout='tight')
                fig,ax = plt.subplots()
                q = plotting.plot_stat_map(meas_nimg, display_mode='z',  threshold=vmax*0.01,
                                            cut_coords=[-20, 0, 18, 40], vmax=vmax*1.1, 
                                            annotate=False, cmap=cmap, colorbar=False,
                                            symmetric_cbar=False, axes=ax
                                    )
                #q.add_edges(meas_nimg)
                min = np.format_float_scientific(np.min(meas_df), precision=3)
                max = np.format_float_scientific(np.max(meas_df), precision=3)
                fig.savefig(f'{out_dir}/{measure}-{series.name}_{min}-{max}.png', dpi=400, bbox_inches='tight')
                plt.close(fig)
            else:
                #print(nifti_mapping.loc[meas_vars]['atlas_fname'])
                atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
                #print(atlas_fname)
                atlas_nii = nib.load(atlas_fname)
                atlas_arr = atlas_nii.get_fdata()
                plotting_arr = np.zeros(atlas_arr.shape)
                for i in meas_df.index:
                    if i in nifti_mapping.index:
                        value = nifti_mapping.loc[i]['atlas_value']
                        
                        #print(i, value)
                        if value is np.nan:
                            pass
                        
                        else:
                            avg = series.at[i]
                            if avg is not float:
                                avg = np.mean(avg)
                            else:
                                pass
                            #print(avg, value, atlas_arr.shape)
                            plotting_arr[np.where(atlas_arr == value)] = avg
                    else:
                        pass
                
                meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
                min = np.format_float_scientific(np.min(meas_df), precision=3)
                max = np.format_float_scientific(np.max(meas_df), precision=3)
                if 'subcortical' in measure:
                    fig,ax = plt.subplots()
                    #plt.figure(layout='tight')
                    q = plotting.plot_stat_map(meas_nimg, display_mode='z', threshold=vmax*0.01,
                                        cut_coords=[-20, -10, 0, 10], vmax=vmax*1.1, 
                                        annotate=False, cmap=cmap, colorbar=False,
                                        symmetric_cbar=False, axes=ax)

                    fig.savefig(f'{out_dir}/{measure}-{series.name}_{min}-{max}.png', dpi=400, bbox_inches='tight')
                    plt.close(fig)
                elif 'cortical' in measure:
                    figure = plot_surfaces(meas_nimg, fsaverage, cmap, vmax, 0.001)
                    figure.savefig(f'{out_dir}/{measure}-{series.name}_{min}-{max}.png', dpi=400, bbox_inches='tight')
                    plt.close(figure)

    # gather variables (network names) for plotting connectivity
    corrs = series.filter(regex='rsfmri_c_ngd.*', axis=0).index
    corrs = [i.split('.')[0] for i in corrs]
    networks = list(np.unique([i.split('_')[-1] for i in corrs]))

    corrs = series.filter(regex='rsfmri_c_ngd.*', axis=0).index
    corrs = [i.split('.')[0] for i in corrs]
    networks = list(np.unique([i.split('_')[-1] for i in corrs]))

    btwn_fc = []
    wthn_fc = []
    for var in fc_cort_var:
        var_list = var.split('_')
        #print(var_list)
        if var_list[3] == var_list[5]:
            #print(var, 'within-network')
            wthn_fc.append(var)
        else:
            btwn_fc.append(var)
            #print(var, 'between-network')

    btwn_fc_src = [i.split('.')[0].split('_')[3] for i in btwn_fc]
    btwn_fc_trgt = [i.split('.')[0].split('_')[-1] for i in btwn_fc]

    #vmax = 3.5

    # okay, now we're plotting between and within network connectivity
    #within-network fc is easy to plot bc there's only one HSK value per network (per fligner_var)
    meas_df = series.loc[wthn_fc]
    if meas_df.sum() == 0:
        pass
    else:
        meas_vars = [i.split('.')[0] for i in meas_df.index]
        atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
        #print(atlas_fname)
        atlas_nii = nib.load(atlas_fname)
        atlas_arr = atlas_nii.get_fdata()
        plotting_arr = np.zeros(atlas_arr.shape)
        for i in meas_df.index:
            j = i.split('.')[0]
            value = nifti_mapping.loc[j]['atlas_value']
            #print(i, value)
            if value is np.nan:
                pass
            elif 'crbwmatterlh' in i:
                pass
            else:
                plotting_arr[np.where(atlas_arr == value)] = series.at[i]

        meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
        figure = plot_surfaces(meas_nimg, fsaverage, cmap, vmax, 0.001)
        figure.savefig(f'{out_dir}/{measure}-{series.name}.png', dpi=400)

    scs_varnames = [i.split('.')[0].split('_')[-1] for i in fc_scor_var]

    # now subcortical-cortical functional connectivity
    sig = []
    meas_df = series.loc[fc_scor_var]

    if meas_df.sum() == 0:
        pass
    else:
        scs_vars = pd.Series(scs_varnames, index=fc_scor_var)
        scs_vars.drop_duplicates(inplace=True)
        avgs = pd.DataFrame()
        for scs in np.unique(scs_varnames):
            scs_temp = scs_vars[scs_vars == scs].index
            temp_df = meas_df[scs_temp]
            # calculate average change of all 
            # significantly heteroscedastic network connections

            for i in temp_df.index:
                sig.append(temp_df.loc[i])
            mean_apc = np.mean(sig)
            #print(mean_hsk)
            # grab name of first conn var for this network for plotting
            avgs.at[temp_df.index[0], 'apc'] = mean_apc
        #print(nsig)
        meas_vars = [i.split('.')[0] for i in avgs.index]
        atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
        #print(atlas_fname)
        atlas_nii = nib.load(atlas_fname)
        atlas_arr = atlas_nii.get_fdata()
        plotting_arr = np.zeros(atlas_arr.shape)
        sig = 0
        for i in avgs.index:
            j = i.split('.')[0]
            value = nifti_mapping.loc[j]['atlas_value']
            #print(i, value)
            if value is np.nan:
                pass
            else:
                plotting_arr[np.where(atlas_arr == value)] = avgs.loc[i]        
        meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
        fig,ax = plt.subplots(#ncols=2, gridspec_kw=grid_kw, figsize=(24,4)
                            )
        #plt.figure(layout='tight')
        q = plotting.plot_stat_map(meas_nimg, display_mode='z',  threshold=.01,
                            cut_coords=[-20, -10, 0, 10], vmax=vmax*1.1, 
                            annotate=False, cmap=cmap, colorbar=False,
                            symmetric_cbar=False, axes=ax)

        #ax[1].set_visible(False)
        min = np.min(avgs)
        max = np.max(avgs)
        fig.savefig(f'{out_dir}/{measure}-{series.name}_{min}-{max}.png', dpi=400, bbox_inches='tight')
        plt.close(fig)

    # between-network FC is tough bc we have to average all of a networks HSK values
    # but only the significantly HSK connections
    sig = []
    meas_df = series.loc[btwn_fc]
    if meas_df.sum() == 0:
        pass
    else:
        #meas_df.loc[btwn_fc, 'from_ntwk'] = btwn_fc_src
        from_ntwks = pd.Series(btwn_fc_src, index=btwn_fc)
        #meas_df.loc[btwn_fc, 'to_ntwk'] = btwn_fc_trgt
        to_ntwks = pd.Series(btwn_fc_trgt, index=btwn_fc)
        avgs = pd.DataFrame()
        for ntwk in np.unique(btwn_fc_src):
            from_ntwk_index = from_ntwks[from_ntwks == ntwk].index
            to_ntwk_index = from_ntwks[to_ntwks == ntwk].index
            temp_df = meas_df.loc[from_ntwk_index]
            temp_df2 = meas_df.loc[to_ntwk_index]
            temp_df = pd.concat([temp_df, temp_df2], axis=0)
            # calculate average heteroscedasticity of all 
            # significantly heteroscedastic network connections
            for i in temp_df.index:
                sig.append(temp_df.loc[i])
            mean_hsk = np.mean(sig)
            # grab name of first conn var for this network for plotting
            avgs.at[temp_df.index[0]] = mean_hsk
        meas_vars = [i.split('.')[0] for i in avgs.index]
        atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
        #print(atlas_fname)
        atlas_nii = nib.load(atlas_fname)
        atlas_arr = atlas_nii.get_fdata()
        plotting_arr = np.zeros(atlas_arr.shape)
        sig = 0
        for i in avgs.index:
            value = nifti_mapping.loc[i]['atlas_value']
            #print(i, value)
            if value is np.nan:
                pass
            elif value.shape == (0,):
                pass
            else:
                if i not in avgs.index:
                    pass
                else:
                    plotting_arr[np.where(atlas_arr == value)] = avgs.loc[i] 
        min = np.format_float_scientific(np.min(avgs), precision=3)
        max = np.format_float_scientific(np.max(avgs), precision=3)    
        meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
        figure = plot_surfaces(meas_nimg, fsaverage, cmap, vmax, 0.001)
        figure.savefig(f'{out_dir}/{measure}-{series.name}_{min}-{max}.png', dpi=400)

    #fig = plt.figure()
    #ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

    #range_ = np.arange(-int(vmax*5), int(vmax*5)) / 5.

    #cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', 
    #                            cmap=cmap, 
    #                            values=range_, 
    #                            )
    #ax.set_xlabel(series.name)

    #plt.savefig(join(out_dir, f'{series.name}-cmap_1-{-vmax,vmax}.png'), bbox_inches='tight', dpi=400)