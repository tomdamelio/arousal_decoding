library(tidyr)
library(magrittr)
library(reticulate)
library(ggbeeswarm)
library(dplyr)
library(stringr)
library(ggplot2)

if (.Platform$OS.type == "windows"){
  setwd("C:/Users/dadam/OneDrive/Escritorio/tomas_damelio")
} else {
  setwd("/storage/store2/work/tdamelio/tomas_damelio")
}

source('config.r')

np <- import("numpy")

# Create a for loop here to open dataframes, add a 'subject' column,
# concatenate this dfs and calculate mean of every column depending on subject

subjects <- sprintf("%02d", 1:32) 

measure <- 'eda'
scores_filename <- str_glue('sub-{sub}_all_scores_models_DEAP_{measure}_r2_2Fold.npy')

measure_uppercase <- toupper(measure)

if (.Platform$OS.type == "windows"){
  scores_dir <- str_glue('{measure}-scores-drago-all-freqs')
  fname <- str_glue("./outputs/DEAP-bids/derivatives/mne-bids-pipeline/")
} else {
  fname <- str_glue("./outputs/DEAP-bids/derivatives/mne-bids-pipeline-{measure}/")
}  

for (sub in subjects)
{

if (.Platform$OS.type == "windows"){
  fname_2 <- str_glue("{scores_dir}/")
} else {
  fname_2 <- str_glue("sub-{sub}/eeg/")
}  
  
fname_data <- fname + fname_2 + scores_filename
  
data <- np$load(
  fname_data,
  allow_pickle = T)[[1]] %>%
  as.data.frame()

# Create a toy df of 'data'
#data <- as.data.frame(matrix(rnorm(7*2, mean=0, sd=0.3), ncol=7))  
data <- data %>% rename(dummy = random,  diag = log_diag)

colnames(data)[6] <- "spoc_opt"
colnames(data)[7] <- "riemann_opt"

data <- data[,c(5,4,3,2,1,6,7)]

#data_comp_scores <- read.csv(
#  "fieldtrip_component_scores.csv",
#  stringsAsFactor = T
#)

# SUBSET TO FILTER ONLY FOLD 0 AND 1 FROM MY DATA
#data_comp_scores <- subset(data_comp_scores , fold_idx == 0 | fold_idx == 1)


# SUBSET TO N_COMPONENTS THAT EXIST IN .NPY FILES
#data_comp_scores <- data_comp_scores %>% filter(n_components %in% (1:32))

# CREATE A TOY DATA OF 'data_compo_scores'
#data_comp_scores <- as.data.frame(matrix(rnorm(7*32, mean=0, sd=0.3), ncol=7))  
#data_comp_scores <- data_comp_scores %>% rename(riemann = V1, spoc = V2, log_diag = V3,
#                        upper = V4, random = V5, spoc_opt = V6, riemann_opt = V7)

#DELETE VARIABLES THAT WILL CHANGE
#data_comp_scores<-data_comp_scores[(data_comp_scores$fold_idx==1 | data_comp_scores$fold_idx==2),]

# CEREATE 'ESTIMATOR' VARIABLE -> ~GATHER
#data_comp_scores_long <- data.frame(
#  score = c(data_comp_scores$spoc, data_comp_scores$riemann),
#  estimator = factor(rep(c("SPoC", "Riemann"),
#                         each = nrow(data_comp_scores))),
#  n_components = rep(data_comp_scores$n_components, times = 2),
#  fold_idx = factor(rep(data_comp_scores$fold_idx, times = 2))
#)

# CREATE A DF 'agg_scores' WITH SPOC AND RIEMANN (SUMMARIZED WITH FOLD MEAN)
#agg_scores <- aggregate(cbind(spoc, riemann) ~ n_components,
#                        data = data_comp_scores, FUN = mean)

# PRINT MEAN AND SD OF DUMMY MODEL SCORE
sprintf("Dummy score %0.3f (+/-%0.3f)", mean(data$dummy),
        sd(data$dummy))

# DELETE 'DUMMY' FROM MODELS TO COMPARE
data_x <- data[, (!names(data) %in% c("dummy"))]
n_splits <- nrow(data_x) # -> ONLY IF THIS IS FOR ONE SUBJECT

# CALCULATE MEAN ON DATA_
data_x$sub <- sub

data_x <- data_x %>%
  group_by(sub) %>%
  summarise_each(funs(mean(., na.rm = TRUE)))


# CONCATENATE ALL SUBJECTS
if (sub == '01'){
data_ <- data_x
  } else {
  data_ <- rbind(data_x, data_)  
  }

}

data_$sub <- NULL

# GATHER ON 'ESTIMATOR' VARIABLE
data_long <- data_ %>% gather(key = "estimator", value = "score")

#data_long <- group_by(data_long, estimator) %>% summarize(score = mean(score))

# move to long format
data_long$estimator <- factor(data_long$estimator)

# DEFINE ESTIMATOR TYPES
est_types <- c(
  "naive",
  "diag",
  "SPoC",
  "Riemann",
  "SPoC",
  "Riemann"
)

# DEFINE ESTIMATOR NAMES (TO PLOT)
est_names <- c(
  "upper",
  "diag",
  "SPoC",
  "Riemann",
  "SPoC_opt",
  "Riemann_opt"
)

est_labels <- setNames(
  c("upper", est_types[c(-1, -5, -6)]),
  est_types[c(-5, -6)]
)

# categorical colors based on: https://jfly.uni-koeln.de/color/
# beef up long data
data_long$est_type <- factor(rep(est_types, each = 32))

#data_long <- data_long %>% 
#                  group_by(estimator) %>% 
#                  mutate(n= n()) %>% 
#                  distinct(estimator, .keep_all=TRUE)

#data_long$n <- NULL

# Concatenate with other df

#data_long <- group_by(data_long, estimator) %>% summarize(score = mean(score))


# CALCULATE MEAN OVER FOLD


data_long$sub <- rep(1:32, times = length(est_types))

# prepare properly sorted x labels
sort_idx <- order(apply(data_, 2, mean))
# IS GOING TO BE USEFUL WHEN PLOTING
levels_est <- est_names[rev(sort_idx)]

my_color_cats <- setNames(
  with(
    color_cats,
    c(`sky blue`, `blueish green`, vermillon, orange)),
  c("naive", "diag", "SPoC", "Riemann"))

ggplot(data = subset(data_long, estimator != "dummy"),
       mapping = aes(y = score, x = reorder(estimator, I(-score)))) +
  geom_beeswarm(
    priority = 'random',
    mapping = aes(color = est_type,
                  alpha = 0.3),
    size = 2.5,
    show.legend = T, cex = 0.15) +
  scale_size_continuous(range = c(0.5, 2)) +
  scale_alpha_continuous(range = c(0.4, 0.7)) +
  geom_boxplot(mapping = aes(fill = est_type, color = est_type),
               alpha = 0.4,
               outlier.fill = NA, outlier.colour = NA) +
  stat_summary(geom = 'text',
               mapping = aes(label  = sprintf("%1.2f",
                                              ..y..)),
               fun.y= mean, size = 3.2, show.legend = FALSE,
               position = position_nudge(x=-0.49)) +
  my_theme +
  labs(y = expression(R^2), x = NULL, parse = T) +
  guides(size = F, alpha = F) +
  theme(legend.position = c(0.8, 0.86)) +
  coord_flip(ylim = c(-2, 2)) +
  scale_fill_manual(values = my_color_cats, breaks = names(my_color_cats),
                    labels = est_labels,
                    name = NULL) +
  scale_color_manual(values = my_color_cats, breaks = names(my_color_cats),
                     labels = est_labels,
                     name = NULL) +
  scale_x_discrete(labels = parse(text = levels_est)) +
  geom_hline(yintercept = mean(data$dummy), linetype = 'dashed') +
  annotate(geom = "text",
           y = mean(data$dummy) + 0.02, x = 2, label = str_glue('predicting~bar({measure_uppercase})'),
           size = annotate_text_size,
           parse = T, angle = 270)

score_out <- str_glue("fig_DEAP_{measure}_model_comp")

if (.Platform$OS.type == "windows"){
  fname_output <- fname + fname_2 + score_out
} else {
  fname_output <- fname + score_out
}  


ggsave(paste0(fname_output, ".png"),
       width = 8, height = 5, dpi = 300)
ggsave(paste0(fname_output, ".pdf"),
       useDingbats = F,
       width = 8, height = 5 , dpi = 300)

