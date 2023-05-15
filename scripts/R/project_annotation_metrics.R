library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(paletteer)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_project_annotation_metrics.csv")
dt$K <- as.factor(dt$K)
dt <- dt %>% filter(metric == 'recall')
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


#color_palette <- paletteer_d("ghibli::KikiMedium")[c(2,3,4)]
ggplot(dt, aes(fill=threshold, y=value, x=K)) +
  geom_bar(position="dodge", stat="identity") +
  facet_nested( ~ annotation + content + algorithm + transformation) + 
  #scale_fill_manual(values=color_palette) +
  xlab(element_blank()) + ylab("Score") + xlab("Recall @") +
  theme(text = element_text(size = 14),
        legend.text=element_text(size=11),
        legend.position="bottom",
        legend.key.size = unit(.3, 'cm')) + labs(fill = "Threshold")

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/project_metrics.pdf', width=9, height=4)
