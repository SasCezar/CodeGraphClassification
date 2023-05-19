library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/src/pipelines/agreement.csv")
dt <- dt %>% filter(k == 10)
dt <- dt %>% filter(y_threshold != 0.25)
dt <- dt %>% filter(x_threshold != 0.25)

dt$k <- as.factor(dt$k)

alpha=0.5
width=0.15

p <- ggplot(dt, aes(x=agreement_percent, fill = k, color=k)) +
  geom_density(alpha=alpha, position = "stack") +
  facet_nested(x_annotation + x_content + x_algorithm + x_transformation + x_threshold ~ y_annotation + y_content + y_algorithm + y_transformation + y_threshold)
p
