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
library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(lavaan)
library(reticulate)
use_python("/usr/bin/python3")
# print versions pls
print(sessionInfo())

#**Data :**  cleaned data, pre processed and
#grouped into two different datasets -
#one for RSI measures and other for Air
#pollutants exposure measures from ABCD data.
pd <- import("pandas")


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_SAaxis/"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"


demo_df <- readRDS('/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.0/01_Demographics/ABCD_5.0_demographics_concise_final.RDS')
demo_df <- demo_df[demo_df$eventname == 'baseline_year_1_arm_1',]

pbty_df <- readRDS('/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.0/04_Physical_Health/ABCD_5.0_Physical_Health.RDS')
base_puberty <- pbty_df[pbty_df$eventname == 'baseline_year_1_arm_1','Puberty_Stage']
y2fu_puberty <- pbty_df[pbty_df$eventname == '2_year_follow_up_y_arm_1','Puberty_Stage']

pbty_df <- cbind(base_puberty,y2fu_puberty)
pbty_df$delta_puberty <- pbty_df$y2fu_puberty - pbty_df$base_puberty

df <- cbind(demo_df, pbty_df)
rownames(df) <- df$src_subject_id

thk <- pd$read_pickle(paste(PROJ_DIR, 
                           OUTP_DIR, 
                           'sa_thk_corrs-rci.pkl',
                           sep = "/",
                           collapse = NULL))

exes = c('base_puberty',
         'interview_age',
        'race_ethnicity_bl', 
        'demo_sex_v2_bl',
        'highest_parent_educ_bl',
        'y2fu_puberty',
        'household_income_4bins_tp',
        'site_id_l', 'rel_family_id_bl')
ppts <- rownames(thk)
temp <- df[df$demo_sex_v2_bl == "Female",]
f_ppts <- rownames(temp)
temp <- df[df$demo_sex_v2_bl == "Male",]
m_ppts <- rownames(temp)

# **CORTICAL THICKNESS **
temp_df <- cbind(thk, df[ppts,exes])
complete_df <- drop_na(temp_df)
write.table(complete_df, file=paste(PROJ_DIR, 
                                    OUTP_DIR, 
                                    'complete_smri.csv',
                                    sep = "/",
                                    collapse = NULL), row.names = TRUE, sep=',')
# female ppts
temp_f <- temp_df[f_ppts,]
complete_f <- drop_na(temp_f)
# male ptps
temp_m <- temp_df[m_ppts,]
complete_m <- drop_na(temp_m)

model_stats <- data.frame(matrix(ncol = 3, nrow = 12))
x <- c("F", "df", "p")
y <- c("thk", "thk_f", "thk_m",
       "rni", "rni_f", "rni_m",
       "rnd", "rnd_f", "rnd_m",
       "var", "var_f", "var_m")
colnames(model_stats) <- x
rownames(model_stats) <- y

thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + demo_sex_v2_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_df)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['thk','p'] <- p
model_stats['thk','F'] <- x$fstatistic['value']
model_stats['thk','df'] <- x$fstatistic['dendf']

tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                             OUTP_DIR,
                             'thk-sa_corr-results-rci.html',
                             sep = "/",
                             collapse = NULL))
# female ppts only
thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_f)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['thk_f','p'] <- p
model_stats['thk_f','F'] <- x$fstatistic['value']
model_stats['thk_f','df'] <- x$fstatistic['dendf']
tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                             OUTP_DIR,
                             'thk-sa_corr-results-female-rci.html',
                             sep = "/",
                             collapse = NULL))
# male ppts only
thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_m)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['thk_m','p'] <- p
model_stats['thk_m','F'] <- x$fstatistic['value']
model_stats['thk_m','df'] <- x$fstatistic['dendf']
tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                             OUTP_DIR,
                             'thk-sa_corr-results-male-rci.html',
                             sep = "/",
                             collapse = NULL))

######### REPEAT FOR RNI #############
thk <- pd$read_pickle(paste(PROJ_DIR, 
                            OUTP_DIR, 
                            'sa_rni_corrs-rci.pkl',
                            sep = "/",
                            collapse = NULL))

temp_df <- cbind(thk, df[ppts,exes])
complete_df <- drop_na(temp_df)
write.table(complete_df, file=paste(PROJ_DIR, 
                                    OUTP_DIR, 
                                    'complete_dmri.csv',
                                    sep = "/",
                                    collapse = NULL), row.names = TRUE, sep=',')


# female ppts
temp_f <- temp_df[f_ppts,]
complete_f <- drop_na(temp_f)
# male ptps
temp_m <- temp_df[m_ppts,]
complete_m <- drop_na(temp_m)

thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + demo_sex_v2_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_df)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['rni','p'] <- p
model_stats['rni','F'] <- x$fstatistic['value']
model_stats['rni','df'] <- x$fstatistic['dendf']

tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'rni-sa_corr-results-rci.html',
                                         sep = "/",
                                         collapse = NULL))
# female ppts only
thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_f)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['rni_f','p'] <- p
model_stats['rni_f','F'] <- x$fstatistic['value']
model_stats['rni_f','df'] <- x$fstatistic['dendf']

tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'rni-sa_corr-results-female-rci.html',
                                         sep = "/",
                                         collapse = NULL))
# male ppts only
thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_m)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['rni_m','p'] <- p
model_stats['rni_m','F'] <- x$fstatistic['value']
model_stats['rni_m','df'] <- x$fstatistic['dendf']

tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'rni-sa_corr-results-male-rci.html',
                                         sep = "/",
                                         collapse = NULL))


######### REPEAT FOR RND #############
thk <- pd$read_pickle(paste(PROJ_DIR, 
                            OUTP_DIR, 
                            'sa_rnd_corrs-rci.pkl',
                            sep = "/",
                            collapse = NULL))

temp_df <- cbind(thk, df[ppts,exes])
complete_df <- drop_na(temp_df)
# female ppts
temp_f <- temp_df[f_ppts,]
complete_f <- drop_na(temp_f)
# male ptps
temp_m <- temp_df[m_ppts,]
complete_m <- drop_na(temp_m)

thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + demo_sex_v2_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_df)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['rnd','p'] <- p
model_stats['rnd','F'] <- x$fstatistic['value']
model_stats['rnd','df'] <- x$fstatistic['dendf']
tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'rnd-sa_corr-results-rci.html',
                                         sep = "/",
                                         collapse = NULL))
# female ppts only
thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_f)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['rnd_f','p'] <- p
model_stats['rnd_f','F'] <- x$fstatistic['value']
model_stats['rnd_f','df'] <- x$fstatistic['dendf']
tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'rnd-sa_corr-results-female-rci.html',
                                         sep = "/",
                                         collapse = NULL))
# male ppts only
thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_m)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['rnd_m','p'] <- p
model_stats['rnd_m','F'] <- x$fstatistic['value']
model_stats['rnd_m','df'] <- x$fstatistic['dendf']
tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'rnd-sa_corr-results-male-rci.html',
                                         sep = "/",
                                         collapse = NULL))


######### REPEAT FOR VAR #############
thk <- pd$read_pickle(paste(PROJ_DIR, 
                            OUTP_DIR, 
                            'sa_var_corrs-rci.pkl',
                            sep = "/",
                            collapse = NULL))

temp_df <- cbind(thk, df[ppts,exes])
complete_df <- drop_na(temp_df)
write.table(complete_df, file=paste(PROJ_DIR, 
                                    OUTP_DIR, 
                                    'complete_rsfmri.csv',
                                    sep = "/",
                                    collapse = NULL), row.names = TRUE, sep=',')


# female ppts
temp_f <- temp_df[f_ppts,]
complete_f <- drop_na(temp_f)
# male ptps
temp_m <- temp_df[m_ppts,]
complete_m <- drop_na(temp_m)

thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + demo_sex_v2_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_df)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['var','p'] <- p
model_stats['var','F'] <- x$fstatistic['value']
model_stats['var','df'] <- x$fstatistic['dendf']
tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'var-sa_corr-results-rci.html',
                                         sep = "/",
                                         collapse = NULL))
# female ppts only
thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_f)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['var_f','p'] <- p
model_stats['var_f','F'] <- x$fstatistic['value']
model_stats['var_f','df'] <- x$fstatistic['dendf']
tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'var-sa_corr-results-female-rci.html',
                                         sep = "/",
                                         collapse = NULL))
# male ppts only
thk_lm <- lm('r ~ base_puberty + y2fu_puberty + interview_age + race_ethnicity_bl + highest_parent_educ_bl + household_income_4bins_tp + site_id_l',
             na.action = na.omit, data = complete_m)

summary(thk_lm)
x <- summary(thk_lm)
p <- pf(x$fstatistic[1],x$fstatistic[2],x$fstatistic[3],lower.tail=FALSE)

model_stats['var_m','p'] <- p
model_stats['var_m','F'] <- x$fstatistic['value']
model_stats['var_m','df'] <- x$fstatistic['dendf']
tab_model(thk_lm, digits = 4, file=paste(PROJ_DIR,
                                         OUTP_DIR,
                                         'var-sa_corr-results-male-rci.html',
                                         sep = "/",
                                         collapse = NULL))
write.csv(model_stats, paste(PROJ_DIR, OUTP_DIR, 'sa_corr-lm_model-stats-rci.csv', sep='/'))
