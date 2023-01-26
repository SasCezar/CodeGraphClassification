library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_package_annotation_stats.csv")
dt <- dt %>% 
  mutate(transformation = replace(transformation, transformation == 'single_label', 'single')) %>%
  mutate(transformation = replace(transformation, transformation == 'soft_label', 'soft')) %>%
  mutate(filtering = replace(filtering, filtering == 'JSDivergence', 'JSD'))

alpha=0.5
width=0.15

p <- ggplot(dt, aes(y = jsd , x=filtering, fill = filtering, color=filtering)) +
  geom_violin(alpha=alpha) +
  geom_boxplot(width = width, fill="white") +
  facet_nested(~ content + annotation + algorithm + transformation)
p

p <- ggplot(dt, aes(y = cohesion , x=filtering, fill = filtering, color=filtering)) +
  geom_violin(alpha=alpha) +
  geom_boxplot(width = width, fill="white") +
  facet_nested(~ content + annotation + algorithm + transformation)
p