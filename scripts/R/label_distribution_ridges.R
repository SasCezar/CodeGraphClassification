library(tidyverse)
library(ggplot2)
library(reshape2)
library(ggridges)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/annotations/kl/name/projects_mean.csv")

ggplot(dt, aes(x = label , y = project, group=project, fill=after_stat(y))) +
  geom_density_ridges(alpha = 0.7, bandwidth = .5) + 
  scale_x_continuous(breaks = seq(0, 266, 20)) +
  theme(legend.position="none", text = element_text(size = 18)) +
  labs(x = "Labels", y=NULL)


ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/annotations/kl_project_level_name_mean.pdf',
       width = 8, height = 5, device=cairo_pdf())