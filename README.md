# deltaABCD_variability

[![DOI](https://zenodo.org/badge/406529480.svg)](https://zenodo.org/badge/latestdoi/406529480)



This work represents the analyses behind [_Profiling intra- and inter-individual differences in child and adolescent brain development_](https://doi.org/10.1101/2022.12.19.521089), studying variability in brain changes, both within and between individuals, during the transition to adolescence using data from the Adolescent Brain Cognitive Development (ABCD) Study. Currently includes code to assess annualized percent change as well as heteroscedasticity with respect to participant age, sex, puberty, and several sociodemographic variables. Only age-, sex-, and puberty-related heterogeneity of variance is reported in the preprint linked above. Sociodemographic and scanner-related heteroscedasticity will be detailed in a forthcoming manuscript. Stay tuned!
## Order of operations
For the most part, the scripts are numbered in the order they were run:
1. `0.0data_wrangling.py` shows exactly which variables were pulled from which data structures included in the 4.0 data release.
2. `0.1sample_demographics.py` calculates the demographic make-up of this sample.
3. `0.2nifti_to_variable_mapping.py` creates a dataframe mapping ABCD Study variable names to the values of their corresponding regions in nifi images, for use in visualizing results.
4. `0.3qc_filtering.py` filters the data from each imaging modality according to recommendations and best practices, then estimates the demographic make-up of the resulting, final sample. Note: this results in different numbers of individuals represented in analyses of structural, functional, and diffusion-weighted data throughout the rest of the analyses.
5. `1.0variance.py` computes the variance in annualized percent change scores for each imaging measure included, then computes heteroscedasticity in each measure across levels of developmental and demographic variables of interest here.
6. `1.1parsing_heterogeneity` tabulates and plots heteroscedasticity of  brain measures across developmental and demographic variables.
7. `1.2change_score_descriptives.py` computes and plots descriptive statistics of APΔ per brain region per measure.
8. `1.4start_change.py` computes product moment and partial correlations between baseline values, baseline age, and APΔ across brain measures; plots results.