library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_project_annotation_stats.csv")
dt$threshold <- as.factor(dt$threshold)
dt <- dt %>% 
  mutate(transformation = replace(transformation, transformation == 'single_label', 'T1')) %>%
  mutate(transformation = replace(transformation, transformation == 'soft_label', 'Tp')) %>%
  mutate(annotation = stringr::str_to_title(annotation)) %>%
  mutate(content = stringr::str_to_title(content)) %>%
  mutate(transformation = stringr::str_to_title(transformation)) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'yake', 'Yake')) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'w2v-so', 'W2V-SO')) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'cascade', 'Cascade')) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'voting', 'Voting')) %>%
  filter(algorithm != 'exp_voting') %>%
  filter(algorithm != 'max') %>%
  mutate(annotation = fct_relevel(annotation, 'Keyword', 'Similarity', 'Ensemble'))


alpha=0.5
width=0.3

p <- ggplot(dt, aes(y = jsd , x=threshold, fill = threshold, color=threshold)) +
  #geom_violin(alpha=alpha, show.legend = FALSE) +
  geom_boxplot(width = width, fill="white", outlier.shape = NA, show.legend = FALSE) +
  facet_nested(~ annotation + content + algorithm + transformation)+
  theme(text = element_text(size = 10),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Filtering Threshold") + ylab("JSD") 
p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/project_JSD.pdf', width=9, height=3)

width=0.3
p <- ggplot(dt, aes(y = percent_unannotated , x = threshold,  fill= threshold, color=threshold)) +
  #geom_violin(alpha=alpha,  show.legend = FALSE) +
  geom_boxplot(width = width, fill="white", outlier.shape = NA, show.legend = FALSE) +
  facet_nested(~ annotation + content + algorithm) +
  theme(text = element_text(size = 9),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Filtering Threshold") + ylab("Unannotated") 
p


ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/project_unannotated.pdf', width=4, height=3)

