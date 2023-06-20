from pynv import Client
from glob import glob
from os.path import join
import pandas as pd
import nibabel as nib

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

col_name = 'Profiling intra- and inter-individual variability in child and adolescent brain development'

# read in neurovault personal access token
# without sharing it with the world via GH
#f = open("nv_access_token.txt", "r")
#token = f.read()
#print(type(token))

# initialize upload client
api = Client(access_token='0a0e8d4ad056a88728856f0f0f72c6edd0aa099e')
collection = api.create_collection(col_name)

# get a list of all niftis from the output folder
niftis = glob(join(PROJ_DIR, OUTP_DIR, '*.nii'))

N_by_meas = {
    'cortical-volume': 7397,
    'subcortical-volume': 7397,
    'cortical-area': 7397,
    'cortical-thickness': 7397,
    'tract-volume': 6848,
    'BOLD-variance': 6762,
    'tract-FA': 6848,
    'tract-MD': 6848,
    'tract-LD': 6848,
    'tract-TD': 6848,
    'cortical-gwcontrast': 7397,
    'cortical-RNI': 6848,
    'subcortical-RNI': 6848,
    'tract-RNI': 6848,
    'cortical-RND': 6848,
    'subcortical-RND': 6848,
    'tract-RND': 6848,
    'FCb': 6762,
    'FCw': 6762,
    'FCscs': 6762
}

long_names = {
    'cortical-thickness': ['cortical thickness', 'Structural_MRI'],
    'cortical-area': ['cortical area', 'Structural_MRI'], 
    'cortical-volume': ['gray matter volume', 'Structural_MRI'],
    'subcortical-volume': ['gray matter volume', 'Structural_MRI'],
    'tract-volume': ['white matter volume', 'Diffusion_MRI'],
    'cortical-gwcontrast': ['t1-t2 contrast', 'Structural_MRI'], 
    'tract-RNI': ['isotropic intracellular diffusion (WM)', 'Diffusion_MRI'], 
    'tract-RND': ['directional intracellular diffusion (WM)', 'Diffusion_MRI'],
    'cortical-RNI': ['isotropic intracellular diffusion (GM)', 'Diffusion_MRI'],
    'subcortical-RNI': ['subcortical isotropic intracellular diffusion (GM)', 'Diffusion_MRI'], 
    'cortical-RND': ['directional intracellular diffusion (GM)', 'Diffusion_MRI'],
    'subcortical-RND': ['directional intracellular diffusion (GM)', 'Diffusion_MRI'],
    'tract-FA': ['fractional anisotropy', 'Diffusion_MRI'], 
    'tract-MD': ['mean diffusivity', 'Diffusion_MRI'],
    'tract-TD': ['transverse diffusivity', 'Diffusion_MRI'], 
    'tract-LD': ['longitudinal diffusivity', 'Diffusion_MRI'],
    'cortical-BOLD-variance': ['bold variance', 'fMRI_BOLD'],
    'FCw': ['within-network functional connectivity', 'fMRI_BOLD'],
    'FCb': ['between-network functional connectivity', 'fMRI_BOLD'],
    'FCscs': ['subcortical-to-network functional connectivity', 'fMRI_BOLD']}

meta_data = pd.DataFrame.from_dict(N_by_meas, orient='index')
meta_data.columns = ['N']

name_df = pd.DataFrame.from_dict(long_names, orient='index')
name_df.columns = ['measure', 'modality']

meta_data = pd.concat([meta_data, name_df], axis=1)

# combine subcortical and cortical maps for same measure
# jk they have different affines and I'm tired

for nifti in niftis:
    fname = nifti.split('/')[-1]
    meas = fname.split('_')[-1].split('.')[0]
    var = fname.split('_')[0]
    measure_name = str(meta_data.loc[meas, 'measure'])

    if 'APC' in var:
        stat_name = 'annualized percent change'
        image_name = f'{stat_name.capitalize()} in {measure_name.title()} from age 9-12 years'
    else:
        stat_name = 'heteroscedasticity'
        measure_name = measure_name.title().replace('Gm', 'GM').replace('Wm', 'WM')
        image_name = f'{stat_name.capitalize()} in {measure_name} with respect to {var}'
    print(image_name)
    modality = meta_data.loc[meas, 'modality']

    image = api.add_image(
        collection['id'],
        nifti,
        name=image_name,
        modality=modality,
        map_type='R',
        analysis_level='group'
    )