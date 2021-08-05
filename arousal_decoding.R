library(tidyr)
library(magrittr)
library(reticulate)
library(ggbeeswarm)
library(dplyr)
library(stringr)
library(ggplot2)

#### Set parameters ####
measure <- 'eda'
y_stat <- 'var'
date_and_time <- '30-07--05-36' #extract from plots' directory
########################

if (.Platform$OS.type == "windows"){
  setwd("C:/Users/dadam/OneDrive/Escritorio/tomas_damelio")
} else {
  setwd("/storage/store3/work/tdamelio/tomas_damelio")
}

source('config.r')

np <- import("numpy")

# Create a for loop here to open dataframes, add a 'subject' column,
# concatenate this dfs and calculate mean of every column depending on subject

subjects <- sprintf("%02d", 1:32) 

measure_uppercase <- toupper(measure)

fname <- str_glue("./outputs/DEAP-bids/derivatives/mne-bids-pipeline-{measure}/")
scores_dir <- str_glue('{measure}_scores--{date_and_time}-{y_stat}')