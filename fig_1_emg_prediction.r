library(tidyr)
library(magrittr)
library(reticulate)
library(ggbeeswarm)
library(dplyr)

setwd("C:/Users/dadam/OneDrive/Escritorio/tomas_damelio")
source('config.r')

np <- import("numpy")

# Create a for loop here to open dataframes, add a 'subject' column,
# concatenate this dfs and calculate mean of every column depending on subject
data <- np$load(
  "./outputs/DEAP-bids/derivatives/mne-bids-pipeline/emg-scores/sub-01_all_scores_models_DEAP_emg_r2_2Fold.npy",
  allow_pickle = T)[[1]] %>%
  as.data.frame()

# Create a toy df of 'data'
data <- as.data.frame(matrix(rnorm(7*2, mean=0, sd=0.3), ncol=7))  
data <- data %>% rename(dummy = V1, naive = V2, 'log-diag' = V3,
                        spoc = V4, riemann = V5, spoc_opt = V6, riemann_opt = V7)

data_comp_scores <- read.csv(
  "fieldtrip_component_scores.csv",
  stringsAsFactor = T
)

# SUBSET TO FILTER ONLY FOLD 0 AND 1 FROM MY DATA
data_comp_scores <- subset(data_comp_scores , fold_idx == 0 | fold_idx == 1)


# SUBSET TO N_COMPONENTS THAT EXIST IN .NPY FILES
data_comp_scores <- data_comp_scores %>% filter(n_components %in% (1:32))

# CREATE A TOY DATA OF 'data_compo_scores'
#data_comp_scores <- as.data.frame(matrix(rnorm(7*32, mean=0, sd=0.3), ncol=7))  
#data_comp_scores <- data_comp_scores %>% rename(riemann = V1, spoc = V2, log_diag = V3,
#                        upper = V4, random = V5, spoc_opt = V6, riemann_opt = V7)

#DELETE VARIABLES THAT WILL CHANGE
#data_comp_scores<-data_comp_scores[(data_comp_scores$fold_idx==1 | data_comp_scores$fold_idx==2),]

# CEREATE 'ESTIMATOR' VARIABLE -> ~GATHER
data_comp_scores_long <- data.frame(
  score = c(data_comp_scores$spoc, data_comp_scores$riemann),
  estimator = factor(rep(c("SPoC", "Riemann"),
                         each = nrow(data_comp_scores))),
  n_components = rep(data_comp_scores$n_components, times = 2),
  fold_idx = factor(rep(data_comp_scores$fold_idx, times = 2))
)

# CREATE A DF 'agg_scores' WITH SPOC AND RIEMANN (SUMMARIZED WITH FOLD MEAN)
agg_scores <- aggregate(cbind(spoc, riemann) ~ n_components,
                        data = data_comp_scores, FUN = mean)

# PRINT MEAN AND SD OF DUMMY MODEL SCORE
sprintf("Dummy score %0.3f (+/-%0.3f)", mean(data$dummy),
        sd(data$dummy))

# DELETE 'DUMMY' FROM MODELS TO COMPARE
data_ <- data[, (!names(data) %in% c("dummy"))]
n_splits <- nrow(data_) # -> ONLY IF THIS IS FOR ONE SUBJECT

# GATHER ON 'ESTIMATOR' VARIABLE
data_long <- data_ %>% gather(key = "estimator", value = "score")
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
  sprintf("SPoC[%d]", which.max(agg_scores$spoc)),
  sprintf("Riemann[%d]", which.max(agg_scores$riemann))
)

est_labels <- setNames(
  c("upper", est_types[c(-1, -5, -6)]),
  est_types[c(-5, -6)]
)

# categorical colors based on: https://jfly.uni-koeln.de/color/
# beef up long data
data_long$est_type <- factor(rep(est_types, each = n_splits))

data_long$fold <- rep(1:n_splits, times = length(est_types))

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
    priority = 'density',
    mapping = aes(color = est_type,
                  alpha = 0.3),
    size = 2.5,
    show.legend = T, cex = 1.5) +
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
  coord_flip() +
  scale_fill_manual(values = my_color_cats, breaks = names(my_color_cats),
                    labels = est_labels,
                    name = NULL) +
  scale_color_manual(values = my_color_cats, breaks = names(my_color_cats),
                     labels = est_labels,
                     name = NULL) +
  scale_x_discrete(labels = parse(text = levels_est)) +
  geom_hline(yintercept = mean(data$dummy), linetype = 'dashed') +
  annotate(geom = "text",
           y = mean(data$dummy) + 0.02, x = 2, label = 'predicting~bar(EMG)',
           size = annotate_text_size,
           parse = T, angle = 270)

fname <- "./outputs/fig_fieldtrip_model_comp"
ggsave(paste0(fname, ".png"),
       width = 8, height = 5, dpi = 300)
ggsave(paste0(fname, ".pdf"),
       useDingbats = F,
       width = 8, height = 5 , dpi = 300)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))


component_labels <- setNames(
  rev(parse(text = est_names[-c(1:4)])),
  c("Riemann", "SPoC"))

fig_components <- ggplot(data = data_comp_scores_long,
                         mapping = aes(
                           x = n_components, y = score,
                           color = estimator, fill = estimator)) +
  stat_summary(fun.y = mean, geom = 'line', size = 1.5) +
  stat_summary(inherit.aes = F,
               mapping = aes(fill = estimator, x = n_components,
                             y = score),
               fun.ymin = function(x) mean(x)-sd(x), 
               fun.ymax = function(x) mean(x)+sd(x),
               geom = 'ribbon', alpha = 0.2) +
  my_theme +
  theme(legend.position = c(0.8, 0.15)) +
  scale_color_manual(
    values = my_color_cats[c("Riemann", "SPoC")],
    breaks = c("Riemann", "SPoC"),
    labels = component_labels,
    name = NULL) +
  scale_fill_manual(
    values = my_color_cats[c("Riemann", "SPoC")],
    breaks = c("Riemann", "SPoC"),
    labels = component_labels,
    name = NULL) +
  labs(x = '#components', y = expression(R ^ 2)) +
  geom_vline(
    xintercept = which.max(agg_scores$spoc),
    size = 0.6,
    color = my_color_cats['SPoC'], linetype = 'dashed') +
  geom_vline(
    xintercept = which.max(agg_scores$riemann),
    size = 0.6,
    color = my_color_cats['Riemann'], linetype = 'dashed') +
  scale_x_continuous(breaks = seq(0, 150, 10)) +
  scale_y_continuous(breaks = seq(-0.2, .7, .1)) +
  coord_cartesian(ylim = c(-0.20, 0.75))

fname <- "./outputs/fig_fieldtrip_component_selection"
ggsave(paste0(fname, ".png"),
       width = 8, height = 5, dpi = 300)

ggsave(paste0(fname, ".pdf"),
       useDingbats = F,
       width = 8, height = 5, dpi = 300)
embedFonts(file = paste0(fname, ".pdf"), outfile = paste0(fname, ".pdf"))