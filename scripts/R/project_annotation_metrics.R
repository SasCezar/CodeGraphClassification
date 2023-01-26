library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(paletteer)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_project_annotation_metrics.csv")
dt$K <- as.factor(dt$K)
dt <- dt %>% 
  mutate(transformation = replace(transformation, transformation == 'single_label', 'single')) %>%
  mutate(transformation = replace(transformation, transformation == 'soft_label', 'soft')) %>%
  mutate(filtering = replace(filtering, filtering == 'JSDivergence', 'JSD'))


ggplot(dt, aes(fill=metric, y=value, x=K)) +
    geom_bar(position="dodge", stat="identity") +
    facet_nested(filtering ~ annotation + content + algorithm + transformation) + 
    scale_fill_paletteer_d("rcartocolor::Bold") +
    xlab(element_blank()) + ylab("Score (%)") + 
    theme(text = element_text(size = 14)) 

ggplot(dt, aes(fill=filtering, y=value, x=K)) +
  geom_bar(position="dodge", stat="identity") +
  facet_nested(metric ~ annotation + content + algorithm + transformation) + 
  scale_fill_paletteer_d("rcartocolor::Bold") +
  xlab(element_blank()) + ylab("Score (%)") + 
  theme(text = element_text(size = 14)) 