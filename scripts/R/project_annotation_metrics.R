library(tidyverse)
library(ggplot2)
library(reshape2)



dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/project_annotation_metrics.csv")
dt$at <- as.factor(dt$at)

ggplot(dt, aes(fill=metric, y=value, x=at)) +
    geom_bar(position="dodge", stat="identity") +
    facet_grid(filtering ~ transformation)