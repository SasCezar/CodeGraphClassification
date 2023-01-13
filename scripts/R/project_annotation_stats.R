library(tidyverse)
library(ggplot2)
library(reshape2)



dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/project_annotation_stats.csv")


alpha=0.3

p <- ggplot(dt, aes(x = project_jsd , fill = filtering, color=filtering)) +
  #geom_density(alpha=alpha)+
  geom_histogram(alpha=alpha, position="identity") +
  facet_grid(transformation ~ .)
p

p <- ggplot(dt, aes(x = project_jsd , fill = transformation, color=transformation)) +
  geom_histogram(alpha=alpha, position="identity") +
  facet_grid(filtering ~ .)
p


p <- ggplot(dt, aes(x = percent_unannotated , fill = transformation, color = transformation)) +
  geom_histogram(alpha=alpha, position="identity") +
  facet_grid(filtering ~ .)
p
