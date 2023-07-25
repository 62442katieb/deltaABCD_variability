## Import Libraries

rm(list = ls())
graphics.off()
library(tidyverse)
library(plyr)
library(dplyr)
library(corrplot)
library(ggplot2)
library(cowplot)
library(readr)
library(gridExtra)
library(grid)
library(readxl)
library(here)
library(psych)
library(RColorBrewer)
library(reticulate)
use_python("/usr/bin/python3")
# print versions pls
print(sessionInfo())

#**Data :**  cleaned data, pre processed and
#grouped into two different datasets -
#one for RSI measures and other for Air
#pollutants exposure measures from ABCD data.
pd <- import("pandas")

mri_df <- read.csv("/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/csv/abcd_mri01.csv")

mri_df <- mri_df[mri_df$eventname == "baseline_year_1_arm_1", ]
rownames(mri_df) <- mri_df$subjectkey

# load cleaned and prepped  RSI data
base_dir <- "/Volumes/projects_herting/LABDOCS/Personnel"
proj_dir <- paste(base_dir,
                  "Katie/deltaABCD_SAaxis",
                  sep = "/")
data_dir <- "data"
figs_dir <- "figures"
outp_dir <- "output"
df <- pd$read_pickle(paste(proj_dir,
                     data_dir,
                     "data_qcd.pkl",
                     sep = "/",
                     collapse = NULL))

# this makes a new dataframe of cortical (gm) rnd measures
mri_info_deviceserialnumber <- mri_df[rownames(df),"mri_info_deviceserialnumber"]
df <- cbind(df, mri_info_deviceserialnumber)

cov_cols <- c("sex.baseline_year_1_arm_1",
              "interview_age.baseline_year_1_arm_1",
              #"site_id_l.baseline_year_1_arm_1",
              "mri_info_deviceserialnumber"
)
# fd_cols <- c("dmri_rsi_meanmotion", "dmri_rsi_meanmotion2")

# can copy paste from here to line 119 and just change "dmri_rsirnd.*" to the 
# cortical thickness and bold variance patterns


# add cortical thickness and bold variance dataframes to the c()
# repeat for cortical thickness ("smri_thick.*") and bold variance ("rsfmri_var.*")
# and "dmri_rsirnigm.*"
rnd_cols <- colnames(df[, grep(pattern = "dmri_rsirndgm.*\\change_score",
                     colnames(df))])

temp_df <- df[, c(rnd_cols, cov_cols)]

complete_df <- drop_na(temp_df)

#load cleaned and prepped brain measures & covariates
df_rsi <- complete_df[, rnd_cols]

df_covariates <- complete_df[, cov_cols]


## Performing residualization
###  MLRM for RSI data
#### REPEAT FOR CORTICAL THICKNESS AND BOLD VARIANCE

# dependent variables
Group1.Y <- as.matrix(df_rsi)
dim(Group1.Y)

# independent variables/covariates
Group1.X <- df_covariates
dim(Group1.X)

#Group1.X$dmri_rsi_meanmotion <- (complete_df$dmri_rsi_meanmotion + complete_df$dmri_rsi_meanmotion2) / 2

# note:
# 1) check all variables- continous/factor
# 2) run str()
# 3) check for normality, homoscedascity,linearity,collinearity

# Multiple regression model
lm.Group1 <- lm(Group1.Y ~ ##Group1.X$interview_age +
                 Group1.X$sex +
                 as.factor(Group1.X$mri_info_deviceserialnumber),
                 na.action = na.omit) # data =)


# R.square -> how well the model explains the variation
# in the data which is not random
# Theorotical model performace is defined as R square

Group1_residuals <- data.frame()
Group1_residuals <- as.data.frame(lm.Group1$residuals)

png(paste(proj_dir, figs_dir, "rnd-brain_change_residuals.png", sep = "/"),
    width = 5, height = 5, res = 600, units = "in")
plot(fitted(lm.Group1), residuals(lm.Group1))
dev.off()

# I don't know if this runs

save(Group1_residuals, file=paste(proj_dir,
                                  outp_dir,
                                  'residualized_rndscanner_sex.Rda',
                                  sep = "/",
                                  collapse = NULL))


# **CORTICAL THICKNESS **
# can copy paste from here to line 119 and just change "dmri_rsirnd.*" to the 
# cortical thickness and bold variance patterns


# add cortical thickness and bold variance dataframes to the c()
# repeat for cortical thickness ("smri_thick.*") and bold variance ("rsfmri_var.*")
# and "dmri_rsirnigm.*"
thick_cols <- colnames(df[, grep(pattern = "smri_thick.*\\change_score",
                               colnames(df))])

temp_df <- df[, c(thick_cols, cov_cols)]

complete_df <- drop_na(temp_df)

#load cleaned and prepped brain measures & covariates
df_thick <- complete_df[, thick_cols]

df_covariates <- complete_df[, cov_cols]


## Performing residualization
###  MLRM for RSI data
#### REPEAT FOR CORTICAL THICKNESS AND BOLD VARIANCE

# dependent variables
Group1.Y <- as.matrix(df_thick)
dim(Group1.Y)

# independent variables/covariates
Group1.X <- df_covariates
dim(Group1.X)

#Group1.X$dmri_rsi_meanmotion <- (complete_df$dmri_rsi_meanmotion + complete_df$dmri_rsi_meanmotion2) / 2

# note:
# 1) check all variables- continous/factor
# 2) run str()
# 3) check for normality, homoscedascity,linearity,collinearity

# Multiple regression model
lm.Group1 <- lm(Group1.Y ~ #Group1.X$interview_age +
                  Group1.X$sex +
                 as.factor(Group1.X$mri_info_deviceserialnumber),
                 na.action = na.omit) # data =)


# R.square -> how well the model explains the variation
# in the data which is not random
# Theorotical model performace is defined as R square

Group1_residuals <- data.frame()
Group1_residuals <- as.data.frame(lm.Group1$residuals)

png(paste(proj_dir, figs_dir, "thick-brain_change_residuals.png", sep = "/"),
    width = 5, height = 5, res = 600, units = "in")
plot(fitted(lm.Group1), residuals(lm.Group1))
dev.off()

# I don't know if this runs

save(Group1_residuals, file=paste(proj_dir,
                                  outp_dir,
                                  'residualized_thickscanner_sex.Rda',
                                  sep = "/",
                                  collapse = NULL))


# **BOLD VARIANCE**
# can copy paste from here to line 119 and just change "dmri_rsirnd.*" to the 
# cortical thickness and bold variance patterns


# add cortical thickness and bold variance dataframes to the c()
# repeat for cortical thickness ("smri_thick.*") and bold variance ("rsfmri_var.*")
# and "dmri_rsirnigm.*"
rsfmri_cols <- colnames(df[, grep(pattern = "rsfmri_var.*\\change_score",
                                 colnames(df))])

temp_df <- df[, c(rsfmri_cols, cov_cols)]

complete_df <- drop_na(temp_df)

#load cleaned and prepped brain measures & covariates
df_rsfmri <- complete_df[, rsfmri_cols]

df_covariates <- complete_df[, cov_cols]


## Performing residualization
###  MLRM for RSI data
#### REPEAT FOR CORTICAL THICKNESS AND BOLD VARIANCE

# dependent variables
Group1.Y <- as.matrix(df_rsfmri)
dim(Group1.Y)

# independent variables/covariates
Group1.X <- df_covariates
dim(Group1.X)

#Group1.X$dmri_rsi_meanmotion <- (complete_df$dmri_rsi_meanmotion + complete_df$dmri_rsi_meanmotion2) / 2

# note:
# 1) check all variables- continous/factor
# 2) run str()
# 3) check for normality, homoscedascity,linearity,collinearity

# Multiple regression model
lm.Group1 <- lm(Group1.Y ~ #Group1.X$interview_age +
                  Group1.X$sex +
                 as.factor(Group1.X$mri_info_deviceserialnumber),
                 na.action = na.omit) # data =)


# R.square -> how well the model explains the variation
# in the data which is not random
# Theorotical model performace is defined as R square

Group1_residuals <- data.frame()
Group1_residuals <- as.data.frame(lm.Group1$residuals)

png(paste(proj_dir, figs_dir, "rsfmri-brain_change_residuals.png", sep = "/"),
    width = 5, height = 5, res = 600, units = "in")
plot(fitted(lm.Group1), residuals(lm.Group1))
dev.off()

# I don't know if this runs

save(Group1_residuals, file=paste(proj_dir,
                                  outp_dir,
                                  'residualized_rsfmriscanner_sex.Rda',
                                  sep = "/",
                                  collapse = NULL))


# **dmri_rsirnigm**
# can copy paste from here to line 119 and just change "dmri_rsirnd.*" to the 
# cortical thickness and bold variance patterns


# add cortical thickness and bold variance dataframes to the c()
# repeat for cortical thickness ("smri_thick.*") and bold variance ("rsfmri_var.*")
# and "dmri_rsirnigm.*"
rnigm_cols <- colnames(df[, grep(pattern = "dmri_rsirnigm.*\\change_score",
                                  colnames(df))])

temp_df <- df[, c(rnigm_cols, cov_cols)]

complete_df <- drop_na(temp_df)

#load cleaned and prepped brain measures & covariates
df_rnigm <- complete_df[, rnigm_cols]

df_covariates <- complete_df[, cov_cols]


## Performing residualization
###  MLRM for RSI data
#### REPEAT FOR CORTICAL THICKNESS AND BOLD VARIANCE

# dependent variables
Group1.Y <- as.matrix(df_rnigm)
dim(Group1.Y)

# independent variables/covariates
Group1.X <- df_covariates
dim(Group1.X)

#Group1.X$dmri_rsi_meanmotion <- (complete_df$dmri_rsi_meanmotion + complete_df$dmri_rsi_meanmotion2) / 2

# note:
# 1) check all variables- continous/factor
# 2) run str()
# 3) check for normality, homoscedascity,linearity,collinearity

# Multiple regression model
lm.Group1 <- lm(Group1.Y ~ #Group1.X$interview_age +
                  Group1.X$sex +
                 as.factor(Group1.X$mri_info_deviceserialnumber),
                 na.action = na.omit) # data =)


# R.square -> how well the model explains the variation
# in the data which is not random
# Theorotical model performace is defined as R square

Group1_residuals <- data.frame()
Group1_residuals <- as.data.frame(lm.Group1$residuals)

png(paste(proj_dir, figs_dir, "rnigm-brain_change_residuals.png", sep = "/"),
    width = 5, height = 5, res = 600, units = "in")
plot(fitted(lm.Group1), residuals(lm.Group1))
dev.off()

# I don't know if this runs

save(Group1_residuals, file=paste(proj_dir,
                                  outp_dir,
                                  'residualized_rnigmscanner_sex.Rda',
                                  sep = "/",
                                  collapse = NULL))
