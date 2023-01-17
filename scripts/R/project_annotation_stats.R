library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_project_annotation_stats.csv")


alpha=0.5
width=0.15

p <- ggplot(dt, aes(y = jsd , x=filtering, fill = filtering, color=filtering)) +
  geom_violin(alpha=alpha) +
  geom_boxplot(width = width, fill="white") +
  facet_nested(~ content + annotation + algorithm + transformation)
p


p <- ggplot(dt, aes(y = percent_unannotated , x = filtering,  fill= filtering, color=filtering)) +
  geom_violin(alpha=alpha) +
  geom_boxplot(width = width, fill="white") +
  facet_nested(~ content + annotation + algorithm)
p
