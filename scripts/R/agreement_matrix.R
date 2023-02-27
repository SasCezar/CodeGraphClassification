library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(paletteer)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/src/pipelines/aggregated_agreement_all.csv")
dt <- dt %>% filter(k == 10)
dt <- dt %>% filter(x_threshold == 0.25)
dt <- dt %>% filter(y_threshold == 0.25)


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
        text = element_text(size = 14),
        legend.key.width = unit(2.5, "cm"),
        legend.position="bottom") +
  scale_fill_gradientn(colours=paletteer_c("viridis::plasma", 10)) +
  labs(fill = "Agreement") +
  facet_nested(x_annotation + x_content + x_algorithm  ~ y_annotation + y_content + y_algorithm,
               switch='y')
p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/lf_agreement_small.pdf', width=8, height=9)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/src/pipelines/aggregated_agreement_all.csv")
dt <- dt %>% filter(k == 10)


dt$k <- as.factor(dt$k)


p <- ggplot(dt, aes(fill=agreement_percent, ymax=1, ymin=0, xmax=agreement_percent, xmin=0)) +
  geom_rect(linetype=0, alpha=1) +
  #geom_text( x=0, aes(y=0.5, label=agreement_percent), size=2)  +
  coord_polar(theta="y") +
  theme(panel.grid = element_blank(), 
        axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        text = element_text(size = 14))  +
  labs(fill = "Agreement") +
  facet_nested(x_annotation + x_content + x_algorithm + x_transformation + x_threshold ~ y_annotation + y_content + y_algorithm + y_transformation + y_threshold,
               switch='y')
p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/lf_agreement_all.pdf', width=30, height=20)
