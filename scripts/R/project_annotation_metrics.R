library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(paletteer)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_project_annotation_metrics.csv")
dt$K <- as.factor(dt$K)
dt <- dt %>% filter(metric == 'recall')
dt$threshold <- as.factor(dt$threshold)
dt <- dt %>% 
  mutate(transformation = replace(transformation, transformation == 'single_label', 'single')) %>%
  mutate(transformation = replace(transformation, transformation == 'soft_label', 'soft')) %>%
  mutate(filtering = replace(filtering, filtering == 'JSDivergence', 'JSD'))


ggplot(dt, aes(fill=metric, y=value, x=K)) +
    geom_bar(position="dodge", stat="identity") +
    facet_nested(threshold ~ annotation + content + algorithm + transformation) + 
    scale_fill_paletteer_d("rcartocolor::Bold") +
    xlab(element_blank()) + ylab("Score (%)") + 
    theme(text = element_text(size = 14)) 
ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/project_metrics_transposed.pdf', width=9, height=4)


#color_palette <- paletteer_d("ghibli::KikiMedium")[c(2,3,4)]
ggplot(dt, aes(fill=threshold, y=value, x=K)) +
  geom_bar(position="dodge", stat="identity") +
  facet_nested( ~ annotation + content + algorithm + transformation) + 
  #scale_fill_manual(values=color_palette) +
  xlab(element_blank()) + ylab("Recall") + 
  theme(text = element_text(size = 14),
        legend.title=element_text(size=10), 
        legend.text=element_text(size=10),
        legend.position="bottom",
        legend.key.size = unit(.5, 'cm')) + labs(fill = "Threshold")

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/project_metrics.pdf', width=9, height=4)
