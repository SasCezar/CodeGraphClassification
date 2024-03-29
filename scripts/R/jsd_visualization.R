library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(paletteer)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/notebooks/test/resources/example_jsd.csv")


color_palette <- paletteer_d("ghibli::KikiMedium")[c(3,4,2)]
p <- ggplot(dt, aes(x=x, fill=funct)) + 
  geom_density(alpha=.5, adjust = .25) +
  scale_x_continuous(limits = c(0,100), expand = c(0.01, 0.01)) +
  scale_y_continuous(limits = c(0,.17), expand = c(0.001, 0.001)) +
  #scale_fill_paletteer_d("ggthemes::wsj_dem_rep") +
  scale_fill_manual(values=color_palette) +
  theme(panel.background = element_blank(),text = element_text(size = 18), 
        legend.position = c(0.94, 0.8),
        legend.title=element_text(size=16), 
        legend.text=element_text(size=16)) +
  xlab("Label") + ylab("Probability") + labs(fill = "Distribution")

p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/notebooks/test/resources/jsd_example.pdf', width=9, height=4)