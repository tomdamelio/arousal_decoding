library(magrittr)
library(reticulate)
library(ggbeeswarm)
library(dplyr)
library(stringr)
library(ggplot2)
install.packages("lme4")
library(lme4)
library(Rcpp)

#### Set parameters ####
measure <- 'eda'
y_stat <- 'var'
date_and_time <- '30-07--05-36' #extract from plots' directory
########################

if (.Platform$OS.type == "windows"){
  setwd("C:/Users/dadam/OneDrive/Escritorio/tomas_damelio")
  fname <- str_glue("./outputs/DEAP-bids/derivatives/mne-bids-pipeline-{measure}/")
  scores_dir <- str_glue('{measure}_{y_stat}_y_and_y_pred_scores/ratings_{measure}_{y_stat}.csv')

} else {
  setwd("/storage/store3/work/tdamelio/tomas_damelio")
  fname <- str_glue("./outputs/DEAP-bids/derivatives/mne-bids-pipeline-{measure}/")
  scores_dir <- str_glue('{measure}_scores--{date_and_time}-{y_stat}/ratings_{measure}_{y_stat}.csv')
    
}

fname_ratings <- fname + scores_dir 

ratings <- read.csv(
  fname_ratings,
  stringsAsFactor = T
)

ratings$Participant_id <- as.factor(ratings$Participant_id)
ratings$Experiment_id <- as.factor(ratings$Experiment_id)


### MODELING ###

# Follow this-> https://ourcodingclub.github.io/tutorials/mixed-models/

# Histogram of Arousal
#hist(ratings$Arousal) 

# Standarise explanatory variables
#ratings$y <- scale(ratings$y, center = TRUE, scale = TRUE)
#ratings$y_pred <- scale(ratings$y_pred, center = TRUE, scale = TRUE)
#ratings$y_pred_delta <- scale(ratings$y_pred_delta, center = TRUE, scale = TRUE)

# Fit linear model ignoring random effects
#lm_fit_basic <- lm(Arousal~ y, data = ratings)
#summary(lm_fit_basic)

# Plot 
#(prelim_plot <- ggplot(ratings, aes(x = y, y = Arousal)) +
#    geom_point() +
#    geom_smooth(method = "lm"))
# When plotting, it is clear that the linear model is fitted based on outliers.
# It seems there are different models (different possible lines) inside the same graph
# Simpson's paradox

# Assumptions ->
# 1. Plot residuals
#plot(lm_fit_basic, which = 1)
# 2. QQ plot
#plot(lm_fit_basic, which = 2)

#boxplot(Arousal ~ Participant_id, data = ratings) # certainly looks like something is going on here

# We could also plot it and colour points by Participants_id:
#(colour_plot <- ggplot(ratings, aes(x = y, y = Arousal, colour = Participant_id)) +
#    geom_point(size = 2) +
#    theme_classic() +
#    theme(legend.position = "none"))

# Run many separate analyses and fit a regression for each of the subjects
#(split_plot <- ggplot(aes(y, Arousal), data = ratings) + 
#    geom_point() + 
#    facet_wrap(~ Participant_id, scales = "free") + # create a facet for each mountain range
#    xlab("EDA") + 
#    ylab("Arousal"))

# Add subject information to out model
#lm_fit_delta <- lm(Arousal~ y + Participant_id, data = ratings)
#summary(lm_fit_delta)
# We are not interested in this

# Mixed linear model
# Nested randomd effects -> https://stats.stackexchange.com/questions/228800/crossed-vs-nested-random-effects-how-do-they-differ-and-how-are-they-specified
mixed_lmer <- lme4::lmer(Arousal ~ y * y_pred + (1 |Participant_id) + (1 |Experiment_id), data = ratings)
summary(mixed_lmer)

