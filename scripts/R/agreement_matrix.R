library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(paletteer)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/src/pipelines/aggregated_agreement_ensamble.csv")
dt <- dt %>% filter(k == 10)
dt <- dt %>% filter(x_algorithm != 'jsd' & x_algorithm != 'exp_voting' & x_algorithm != 'max')
dt <- dt %>% filter(y_algorithm != 'jsd' & y_algorithm != 'exp_voting' & y_algorithm != 'max')
dt <- dt %>% filter(x_threshold == 0.25 | x_algorithm == 'voting' | x_algorithm == 'cascade')
dt <- dt %>% filter(y_threshold == 0.25 | y_algorithm == 'voting' | y_algorithm == 'cascade')
dt <- dt %>% 
  mutate(x_annotation = stringr::str_to_title(x_annotation)) %>%
  mutate(x_content = stringr::str_to_title(x_content)) %>%
  mutate(x_transformation = stringr::str_to_title(x_transformation)) %>%
  mutate(x_algorithm = replace(x_algorithm, x_algorithm == 'yake', 'Yake')) %>%
  mutate(x_algorithm = replace(x_algorithm, x_algorithm == 'w2v-so', 'W2V-SO')) %>%
  mutate(x_algorithm = replace(x_algorithm, x_algorithm == 'cascade', 'CSC')) %>%
  mutate(x_algorithm = replace(x_algorithm, x_algorithm == 'voting', 'VT')) %>% 
  mutate(y_annotation = stringr::str_to_title(y_annotation)) %>%
  mutate(y_content = stringr::str_to_title(y_content)) %>%
  mutate(y_transformation = stringr::str_to_title(y_transformation)) %>%
  mutate(y_algorithm = replace(y_algorithm, y_algorithm == 'yake', 'Yake')) %>%
  mutate(y_algorithm = replace(y_algorithm, y_algorithm == 'w2v-so', 'W2V-SO')) %>%
  mutate(y_algorithm = replace(y_algorithm, y_algorithm == 'cascade', 'CSC')) %>%
  mutate(y_algorithm = replace(y_algorithm, y_algorithm == 'voting', 'VT'))
dt <- dt %>% mutate(x_annotation = fct_relevel(x_annotation, 'Keyword', 'Similarity', 'Ensemble'))
dt <- dt %>% mutate(y_annotation = fct_relevel(y_annotation, 'Keyword', 'Similarity', 'Ensemble')) 

dt$k <- as.factor(dt$k)

p <- ggplot(dt, aes(fill=agreement_percent, ymax=1, ymin=0, xmax=agreement_percent, xmin=0, label=sprintf("%0.2f", round(agreement_percent, digits = 2)))) +
  geom_rect(linetype=0, alpha=1) +
  #geom_text(x=0, y=0, size=2.2, vjust = -1.6)  +
  #(x=0, y=0, size=3, vjust = -4)  +
  coord_polar(theta="y") +
  theme(panel.grid = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        text = element_text(size = 18),
        legend.key.width = unit(2.5, "cm"),
        legend.position="bottom") +
  scale_fill_gradientn(colours=paletteer_c("viridis::plasma", 10)) +
  labs(fill = "Agreement") +
  facet_nested(x_annotation + x_content + x_algorithm  ~ y_annotation + y_content + y_algorithm,
               switch='y')
p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/lf_agreement_ensamble.pdf', width=8, height=9)

