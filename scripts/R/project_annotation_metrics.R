library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(paletteer)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_project_annotation_metrics.csv")
dt$K <- as.factor(dt$K)

ggplot(dt, aes(fill=K, y=value, x=metric)) +
    geom_bar(position="dodge", stat="identity") +
    facet_nested(filtering ~ annotation + content + algorithm + transformation) + 
    scale_fill_paletteer_d("rcartocolor::Prism")