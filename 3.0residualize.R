## Import Libraries

rm(list = ls())
graphics.off()
library(tidyverse)
library(ExPosition)
library(TExPosition)
library(PTCA4CATA)
library(data4PCCAR)
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
use_python("/usr/local/bin/python")
# print versions pls
print(sessionInfo())

#**Data :**  cleaned data, pre processed and
#grouped into two different datasets -
#one for RSI measures and other for Air
#pollutants exposure measures from ABCD data.
pd <- import("pandas")


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




cov_cols <- c("sex",
              "interview_age",
              "site_id_l",
              "mri_info_manufacturer",
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
lm.Group1 <- lm(Group1.Y ~ Group1.X$interview_age +
                 Group1.X$sex +
                 as.factor(Group1.X$mri_info_manufacturer) +
                 as.factor(Group1.X$site_id_l), na.action = na.omit) # data =)


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
Group1_residuals$to_pickle(paste(proj_dir,
                                 outp_dir,
                                 'residualized_rnd.pkl',
                           sep = "/",
                           collapse = NULL))