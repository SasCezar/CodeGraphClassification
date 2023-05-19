library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(ragg)


alpha=0.5
width=0.25

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/scripts/python/unannotated_avg.csv")
dt$threshold <- as.factor(dt$threshold)
dt <- dt %>% 
  mutate(transformation = replace(transformation, transformation == 'single_label', 'T1')) %>%
  mutate(transformation = replace(transformation, transformation == 'soft_label', 'Tp')) %>%
  mutate(transformation = replace(transformation, transformation == 'none', 'RAW')) %>%
  mutate(annotation = stringr::str_to_title(annotation)) %>%
  mutate(content = stringr::str_to_title(content)) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'yake', 'Yake')) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'w2v-so', 'W2V-SO')) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'cascade', 'CSC')) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'voting', 'VT')) %>%
  filter(algorithm != 'exp_voting') %>%
  filter(algorithm != 'max') %>%
  mutate(annotation = fct_relevel(annotation, 'Keyword', 'Similarity', 'Ensemble'))

p <- ggplot(dt, aes(y = unannotated , x=threshold, fill = threshold, color=threshold)) +
  #geom_violin(alpha=alpha,  show.legend = FALSE) +
  geom_boxplot(width = width, fill="white", outlier.shape = NA, show.legend = FALSE) +
  facet_nested(~ annotation + content + algorithm + transformation) +
  theme(text = element_text(size = 10),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Filtering Threshold") + ylab("Unannotated") 
p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_unannotated_percent.pdf', width=9, height=4)