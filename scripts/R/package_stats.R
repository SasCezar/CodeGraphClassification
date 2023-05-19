library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(ragg)
library(ggpubr)

#Our transformation function
scaleFUN <- function(x) sprintf("%.2f", x)

alpha=0.5
width=0.3

dtu <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/scripts/python/unannotated_avg.csv")
dtu$threshold <- as.factor(dtu$threshold)
dtu <- dtu %>% 
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

pu <- ggplot(dtu, aes(y = unannotated , x=threshold, fill = threshold, color=threshold)) +
#  geom_violin(alpha=alpha,  show.legend = FALSE) +
  geom_boxplot(width = width, fill="white", outlier.shape = NA, show.legend = FALSE) +
  facet_nested(~ annotation + content + algorithm + transformation) +
  theme(text = element_text(size = 14),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=11),
        strip.text.x = element_blank()) +
  xlab("Filtering Threshold") + ylab("Unannotated") 
pu

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_package_annotation_stats.csv")
dt$threshold <- as.factor(dt$threshold)
dt <- dt %>% 
  mutate(cohesion = replace(cohesion, TRUE, 1-cohesion)) %>%
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

pjsd <- ggplot(dt, aes(y = cohesion , x=threshold, fill = threshold, color=threshold)) +
#  geom_violin(alpha=alpha,  show.legend = FALSE) +
  geom_boxplot(width = width, fill="white", outlier.shape = NA, show.legend = FALSE) +
  facet_nested(~ annotation + content + algorithm + transformation) +
  scale_y_continuous(labels=scaleFUN) +
  theme(text = element_text(size = 14),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1, size=11)) +
  xlab(element_blank())  + ylab("Cohesion") 
pjsd

figure <- ggarrange(pjsd, pu,
                    ncol = 1, nrow = 2, heights=c(4, 3)) #2.5
figure


ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_stats.pdf', width=9, height=5)
#ggexport(figure, filename = "/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_stats.pdf",
#         nrow = 2, ncol = 1, width=20, height=8, res=300, pointsize=60)

