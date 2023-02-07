library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_package_annotation_stats.csv")
dt$threshold <- as.factor(dt$threshold)
dt <- dt %>% 
  mutate(transformation = replace(transformation, transformation == 'single_label', 'single')) %>%
  mutate(transformation = replace(transformation, transformation == 'soft_label', 'soft')) %>%
  mutate(filtering = replace(filtering, filtering == 'JSDivergence', 'JSD'))%>%
  mutate(cohesion = if_else(cohesion == -1, -1, 1 - cohesion))

alpha=0.5
width=0.15

p <- ggplot(dt, aes(y = jsd , x=threshold, fill = threshold, color=threshold)) +
  geom_violin(alpha=alpha) +
  geom_boxplot(width = width, fill="white") +
  facet_nested(clean~ annotation + content + algorithm + transformation)
p

p <- ggplot(dt, aes(y = cohesion , x=threshold, fill = threshold, color=threshold)) +
  geom_violin(alpha=alpha) +
  geom_boxplot(width = width, fill="white") +
  facet_nested(clean~ annotation + content  + algorithm + transformation)
p

p <- ggplot(dt, aes(y = unannotated , x=threshold, fill = threshold, color=threshold)) +
  geom_violin(alpha=alpha) +
  geom_boxplot(width = width, fill="white") +
  facet_nested(~ annotation + content  + algorithm + transformation)
p