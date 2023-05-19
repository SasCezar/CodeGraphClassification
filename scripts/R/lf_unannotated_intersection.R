library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(paletteer)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/src/pipelines/node_annot_intersection.csv")

p <- ggplot(dt, aes(x = "", y=intersection_percent)) +
  geom_violin() +
  theme(panel.grid = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        text = element_text(size = 14))  +
  labs(y = "Agreement") +
  facet_nested(x_annotation + x_content + x_algorithm + x_transformation  ~ y_annotation + y_content + y_algorithm + y_transformation,
               switch='y')
p

ggsave('', width=30, height=20)
