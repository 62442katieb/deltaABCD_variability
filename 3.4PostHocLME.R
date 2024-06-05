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
library(lme4)
library(naniar)
library(reticulate)
library(interactions)
use_python("/usr/bin/python3")
set_theme(base = theme_minimal())
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

regform <- 'r ~ interview_age.baseline_year_1_arm_1 + I(interview_age.baseline_year_1_arm_1^2) + demo_sex_v2_bl + baseline_Puberty	 + delta_Puberty +
baseline_Puberty*demo_sex_v2_bl + delta_Puberty*demo_sex_v2_bl+
race_ethnicity_c_bl + highest_parent_educ_bl + household_income_4bins_bl + (1|rel_family_id_bl:site_id_l)'

sexreg <- 'r ~ interview_age.baseline_year_1_arm_1 + I(interview_age.baseline_year_1_arm_1^2) + baseline_Puberty	 + delta_Puberty + race_ethnicity_c_bl + 
highest_parent_educ_bl + household_income_4bins_bl + (1|rel_family_id_bl:site_id_l)'

# **CORTICAL THICKNESS **
thk <- pd$read_pickle(paste(PROJ_DIR, 
                            OUTP_DIR, 
                            'thk_plus_demos-apd.pkl',
                            sep = "/",
                            collapse = NULL))

temp <- replace_with_na(
  thk,
  replace = list(highest_parent_educ_bl = "Missing/Refused")
)
complete_df <- drop_na(temp)
complete_df$race_ethnicity_c_bl <- factor(complete_df$race_ethnicity_c_bl, 
                                          levels = c('White', 'Asian/Other', 'Black', 'Hispanic'),
                                          labels = c('White', 'Asian/Other', 'Black', 'Hispanic'), 
                                          ordered = FALSE)
complete_df$household_income_4bins_bl <- factor(complete_df$household_income_4bins_bl,
                                                levels = c("<50k", 
                                                           "50k_100k",
                                                           ">100k",
                                                           "Don't know/Refuse to answer"
                                                ),
                                                labels = c("<50k", 
                                                           "50k_100k",
                                                           ">100k",
                                                           "Don't know/Refuse to answer"
                                                ),
                                                ordered = FALSE)
complete_df$highest_parent_educ_bl <- factor(complete_df$highest_parent_educ_bl,
                                             levels = c("< HS Diploma",
                                                        "HS Diploma/GED",
                                                        "Some College",
                                                        "Bachelor Degree",
                                                        "Post Graduate Degree"#,
                                                        #"Missing/Refused"
                                             ),
                                             labels = c("< HS Diploma",
                                                        "HS Diploma/GED",
                                                        "Some College",
                                                        "Bachelor Degree",
                                                        "Post Graduate Degree"#,
                                                        #"Missing/Refused"
                                             ),
                                             ordered = FALSE)
forward_puberty <- rownames(complete_df[complete_df$delta_Puberty >= 0,])
reverse_puberty <- rownames(complete_df[complete_df$delta_Puberty < 0,])

#f <- rownames(complete_df[complete_df$demo_sex_v2_bl == "Female",])
#m <- rownames(complete_df[complete_df$demo_sex_v2_bl == "Male",])

write.table(complete_df, file=paste(PROJ_DIR, 
                                    OUTP_DIR, 
                                    'complete_smri-apd.csv',
                                    sep = "/",
                                    collapse = NULL), row.names = TRUE, sep=',')


thk_lm <- lmer(regform,
               na.action = na.omit, data = complete_df)
thk_lm2 <- lmer(regform,
                na.action = na.omit, data = complete_df[forward_puberty,])

tab_model(thk_lm, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(PROJ_DIR,
                     OUTP_DIR,
                     'thk-sa_corr-results-apd_lme.html',
                     sep = "/",
                     collapse = NULL))

tab_model(thk_lm2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(PROJ_DIR,
                     OUTP_DIR,
                     'thk-sa_corr-results-apd_lme-fwd_only.html',
                     sep = "/",
                     collapse = NULL))

ss_timing <- sim_slopes(
  thk_lm2, 
  pred = "baseline_Puberty", 
  modx = "demo_sex_v2_bl"
)
ss_timing
ss_tempo <- sim_slopes(
  thk_lm2, 
  pred = "delta_Puberty", 
  modx = "demo_sex_v2_bl"
)
ss_tempo


ss1 <- interact_plot(
  thk_lm2, 
  pred = "baseline_Puberty", 
  modx = "demo_sex_v2_bl", 
  interval = TRUE, 
  plot.points = FALSE,
  colors = c("#ffb000", "#648fff")
)
ss1 + theme(
  axis.text = element_text(size = 16), 
  axis.title.x = element_text(size=16),
  axis.title.y = element_text(size=16),
  legend.position = "none"
)
ggsave(paste(PROJ_DIR, FIGS_DIR, "lmer_thk_timing-fwd_only-apd.png", sep = "/"),
       plot = ss1,
       bg = "#FFFFFF",
       device = "png",
       width = 4, height = 2,
       dpi = 300)

ss2 <- interact_plot(
  thk_lm2, 
  pred = "delta_Puberty", 
  modx = "demo_sex_v2_bl", 
  interval = TRUE, 
  plot.points = FALSE,
  colors = c("#ffb000", "#648fff")
)
ss2 + theme(
  axis.text = element_text(size = 16), 
  axis.title.x = element_text(size=16),
  axis.title.y = element_text(size=16),
  legend.position = "none")
ggsave(paste(PROJ_DIR, FIGS_DIR, "lmer_thk_tempo-fwd_only-apd.png", sep = "/"),
       plot = ss2,
       bg = "#FFFFFF",
       device = "png",
       width = 4, height = 2,
       dpi = 300)

######### REPEAT FOR RNI #############
rni <- pd$read_pickle(paste(PROJ_DIR, 
                            OUTP_DIR, 
                            'rni_plus_demos-apd.pkl',
                            sep = "/",
                            collapse = NULL))

temp <- replace_with_na(
  rni,
  replace = list(highest_parent_educ_bl = "Missing/Refused")
)
complete_df <- drop_na(temp)
complete_df$race_ethnicity_c_bl <- factor(complete_df$race_ethnicity_c_bl, 
                                          levels = c('White', 'Asian/Other', 'Black', 'Hispanic'),
                                          labels = c('White', 'Asian/Other', 'Black', 'Hispanic'), 
                                          ordered = FALSE)
complete_df$household_income_4bins_bl <- factor(complete_df$household_income_4bins_bl,
                                                levels = c("<50k", 
                                                           "50k_100k",
                                                           ">100k",
                                                           "Don't know/Refuse to answer"),
                                                labels = c("<50k", 
                                                           "50k_100k",
                                                           ">100k",
                                                           "Don't know/Refuse to answer"),
                                                ordered = FALSE)
complete_df$highest_parent_educ_bl <- factor(complete_df$highest_parent_educ_bl,
                                             levels = c("< HS Diploma",
                                                        "HS Diploma/GED",
                                                        "Some College",
                                                        "Bachelor Degree",
                                                        "Post Graduate Degree"#,
                                                        #"Missing/Refused"
                                                        ),
                                             labels = c("< HS Diploma",
                                                        "HS Diploma/GED",
                                                        "Some College",
                                                        "Bachelor Degree",
                                                        "Post Graduate Degree"#,
                                                        #"Missing/Refused"
                                                        ),
                                             ordered = FALSE)
forward_puberty <- rownames(complete_df[complete_df$delta_Puberty >= 0,])
reverse_puberty <- rownames(complete_df[complete_df$delta_Puberty < 0,])

write.table(complete_df, file=paste(PROJ_DIR, 
                                    OUTP_DIR, 
                                    'complete_dmri-apd.csv',
                                    sep = "/",
                                    collapse = NULL), row.names = TRUE, sep=',')

rni_lm <- lmer(regform,
               na.action = na.omit, data = complete_df)
rni_lm2 <- lmer(regform,
                na.action = na.omit, data = complete_df[forward_puberty,])


summary(rni_lm)

tab_model(rni_lm, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(PROJ_DIR,
                     OUTP_DIR,
                     'rni-sa_corr-results-apd_lme.html',
                     sep = "/",
                     collapse = NULL))
tab_model(rni_lm2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(PROJ_DIR,
                     OUTP_DIR,
                     'rni-sa_corr-results-apd_lme-fwd_only.html',
                     sep = "/",
                     collapse = NULL))

ss_timing <- sim_slopes(
  rni_lm2, 
  pred = "baseline_Puberty", 
  modx = "demo_sex_v2_bl"
)
ss_timing
ss_tempo <- sim_slopes(
  rni_lm2, 
  pred = "delta_Puberty", 
  modx = "demo_sex_v2_bl"
)
ss_tempo

ss1 <- interact_plot(
  rni_lm2, 
  pred = "baseline_Puberty", 
  modx = "demo_sex_v2_bl", 
  interval = TRUE, 
  plot.points = FALSE,
  colors = c("#ffb000", "#648fff")
)
ss1 + theme(
  axis.text = element_text(size = 16), 
  axis.title.x = element_text(size=16),
  axis.title.y = element_text(size=16),
  legend.position = "none"
)
ggsave(paste(PROJ_DIR, FIGS_DIR, "lmer_rni_timing-fwd_only-apd.png", sep = "/"),
       plot = ss1,
       bg = "#FFFFFF",
       device = "png",
       width = 4, height = 2,
       dpi = 300)

ss2 <- interact_plot(
  rni_lm2, 
  pred = "delta_Puberty", 
  modx = "demo_sex_v2_bl", 
  interval = TRUE, 
  colors = c("#ffb000", "#648fff")
)
ss2 + theme(
  axis.text = element_text(size = 16), 
  axis.title.x = element_text(size=16),
  axis.title.y = element_text(size=16),
  legend.position = "none")
ggsave(paste(PROJ_DIR, FIGS_DIR, "lmer_rni_tempo-fwd_only-apd.png", sep = "/"),
       plot = ss2,
       bg = "#FFFFFF",
       device = "png",
       width = 4, height = 2,
       dpi = 300)


######### REPEAT FOR RND #############
rnd <- pd$read_pickle(paste(PROJ_DIR, 
                            OUTP_DIR, 
                            'rnd_plus_demos-apd.pkl',
                            sep = "/",
                            collapse = NULL))

temp <- replace_with_na(
  rnd,
  replace = list(highest_parent_educ_bl = "Missing/Refused")
)
complete_df <- drop_na(temp)
complete_df$race_ethnicity_c_bl <- factor(complete_df$race_ethnicity_c_bl, 
                                          levels = c('White', 'Asian/Other', 'Black', 'Hispanic'),
                                          labels = c('White', 'Asian/Other', 'Black', 'Hispanic'), 
                                          ordered = FALSE)
complete_df$household_income_4bins_bl <- factor(complete_df$household_income_4bins_bl,
                                                levels = c("<50k", 
                                                           "50k_100k",
                                                           ">100k",
                                                           "Don't know/Refuse to answer"),
                                                labels = c("<50k", 
                                                           "50k_100k",
                                                           ">100k",
                                                           "Don't know/Refuse to answer"),
                                                ordered = FALSE)
complete_df$highest_parent_educ_bl <- factor(complete_df$highest_parent_educ_bl,
                                             levels = c("< HS Diploma",
                                                        "HS Diploma/GED",
                                                        "Some College",
                                                        "Bachelor Degree",
                                                        "Post Graduate Degree"#,
                                                        #"Missing/Refused"
                                                        ),
                                             labels = c("< HS Diploma",
                                                        "HS Diploma/GED",
                                                        "Some College",
                                                        "Bachelor Degree",
                                                        "Post Graduate Degree"#,
                                                        #"Missing/Refused"
                                                        ),
                                             ordered = FALSE)
forward_puberty <- rownames(complete_df[complete_df$delta_Puberty >= 0,])
reverse_puberty <- rownames(complete_df[complete_df$delta_Puberty < 0,])

rnd_lm <- lmer(regform,
               na.action = na.omit, data = complete_df)
rnd_lm2 <- lmer(regform,
                na.action = na.omit, data = complete_df[forward_puberty,])

summary(rnd_lm)

tab_model(rnd_lm, digits = 4,show.aic = T, show.std = "std2",
          file=paste(PROJ_DIR,
                     OUTP_DIR,
                     'rnd-sa_corr-results-apd_lme.html',
                     sep = "/",
                     collapse = NULL))
tab_model(rnd_lm2, digits = 4,show.aic = T, show.std = "std2",
          file=paste(PROJ_DIR,
                     OUTP_DIR,
                     'rnd-sa_corr-results-apd_lme-fwd_only.html',
                     sep = "/",
                     collapse = NULL))
ss_timing <- sim_slopes(
  rnd_lm2, 
  pred = "baseline_Puberty", 
  modx = "demo_sex_v2_bl"
)
ss_timing
ss_tempo <- sim_slopes(
  rnd_lm2, 
  pred = "delta_Puberty", 
  modx = "demo_sex_v2_bl"
)
ss_tempo
######### REPEAT FOR VAR #############
var <- pd$read_pickle(paste(PROJ_DIR, 
                            OUTP_DIR, 
                            'var_plus_demos-apd.pkl',
                            sep = "/",
                            collapse = NULL))

temp <- replace_with_na(
  var,
  replace = list(highest_parent_educ_bl = "Missing/Refused")
)
complete_df <- drop_na(temp)
complete_df$race_ethnicity_c_bl <- factor(complete_df$race_ethnicity_c_bl, 
                                          levels = c('White', 'Asian/Other', 'Black', 'Hispanic'),
                                          labels = c('White', 'Asian/Other', 'Black', 'Hispanic'), 
                                          ordered = FALSE)
complete_df$household_income_4bins_bl <- factor(complete_df$household_income_4bins_bl,
                                                levels = c("<50k", 
                                                           "50k_100k",
                                                           ">100k",
                                                           "Don't know/Refuse to answer"),
                                                labels = c("<50k", 
                                                           "50k_100k",
                                                           ">100k",
                                                           "Don't know/Refuse to answer"),
                                                ordered = FALSE)
complete_df$highest_parent_educ_bl <- factor(complete_df$highest_parent_educ_bl,
                                             levels = c("< HS Diploma",
                                                        "HS Diploma/GED",
                                                        "Some College",
                                                        "Bachelor Degree",
                                                        "Post Graduate Degree"#,
                                                        #"Missing/Refused"
                                                        ),
                                             labels = c("< HS Diploma",
                                                        "HS Diploma/GED",
                                                        "Some College",
                                                        "Bachelor Degree",
                                                        "Post Graduate Degree"#,
                                                        #"Missing/Refused"
                                                        ),
                                             ordered = FALSE)
forward_puberty <- rownames(complete_df[complete_df$delta_Puberty >= 0,])
reverse_puberty <- rownames(complete_df[complete_df$delta_Puberty < 0,])

write.table(complete_df, file=paste(PROJ_DIR, 
                                    OUTP_DIR, 
                                    'complete_rsfmri-apd.csv',
                                    sep = "/",
                                    collapse = NULL), row.names = TRUE, sep=',')


var_lm <- lmer(regform,
               na.action = na.omit, data = complete_df)

var_lm2 <- lmer(regform,
                na.action = na.omit, data = complete_df[forward_puberty,])

summary(var_lm2)
ss_timing <- sim_slopes(
  var_lm2, 
  pred = "baseline_Puberty", 
  modx = "demo_sex_v2_bl"
)
ss_timing
ss_tempo <- sim_slopes(
  var_lm2, 
  pred = "delta_Puberty", 
  modx = "demo_sex_v2_bl"
)
ss_tempo
tab_model(var_lm, digits = 4, show.fstat = T,
          show.aic = T, show.std = "std2",
          file=paste(PROJ_DIR,
                     OUTP_DIR,
                     'var-sa_corr-results-apd_lme.html',
                     sep = "/",
                     collapse = NULL))

tab_model(var_lm2, digits = 4, show.fstat = T,
          show.aic = T, show.std = "std2",
          file=paste(PROJ_DIR,
                     OUTP_DIR,
                     'var-sa_corr-results-apd_lme-fwd_only.html',
                     sep = "/",
                     collapse = NULL))

q <- plot_models(rnd_lm, rni_lm, var_lm, thk_lm,
                 grid = TRUE, 
                 spacing = 1, 
                 colors=c("#1b9e77", "#d95f02", "#7570b3", "#e7298a"), 
                 vline.color="#999999",
                 dot.size=2, 
                 std.est = 'std2', #robust = TRUE,
                 legend.title = "Brain Changes", 
                 m.labels=c("Neurite Density",
                            "Cellularity",
                            "Functional fluctuations",
                            "Cortical thickness"), 
                 rm.terms=c(
                   "race_ethnicity_c_bl [Asian/Other, Black, Hispanic]",
                   "highest_parent_educ_bl [HS Diploma/GED, Some College, Bachelor Degree, Post Graduate Degree]",
                   "household_income_4bins_bl [50k_100k, >100k]"
                 ),
                 show.p=TRUE) + ylim(-0.4,0.3)
q + theme(legend.position="bottom",
          legend.title=element_blank(),
          panel.spacing = unit(1, "lines"))
ggsave(paste(PROJ_DIR, FIGS_DIR, "lmer_fxs_grid-apd.png", sep = "/"),
       #plot = q,
       bg = "#FFFFFF",
       device = "png",
       dpi = "retina")

p <- plot_models(rnd_lm2, rni_lm2, var_lm2, thk_lm2,
                 grid = TRUE, 
                 spacing = 1, 
                 colors=c("#1b9e77", "#d95f02", "#7570b3", "#e7298a"), 
                 vline.color="#999999",
                 dot.size=2, 
                 std.est = 'std2', #robust = TRUE,
                 legend.title = "Brain Changes", 
                 m.labels=c("Neurite Density",
                            "Cellularity",
                            "Functional fluctuations",
                            "Cortical thickness"), 
                 rm.terms=c(
                   "race_ethnicity_c_bl [Asian/Other, Black, Hispanic]",
                   "highest_parent_educ_bl [HS Diploma/GED, Some College, Bachelor Degree, Post Graduate Degree]",
                   "household_income_4bins_bl [50k_100k, >100k]"
                 ),
                 show.p=TRUE) + ylim(-0.4,0.3)
p + theme(legend.position="bottom",
          legend.title=element_blank(),
          panel.spacing = unit(1, "lines"))

ggsave(paste(PROJ_DIR, FIGS_DIR, "lmer_fxs_grid-fwd_only-apd.png", sep = "/"),
       #plot = p,
       bg = "#FFFFFF",
       device = "png",
       dpi = "retina")







