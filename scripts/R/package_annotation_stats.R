library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)
library(ragg)

dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/output/stats/similarity_package_annotation_stats.csv")
dt$threshold <- as.factor(dt$threshold)
dt <- dt %>% 
  mutate(transformation = replace(transformation, transformation == 'single_label', 'single')) %>%
  mutate(transformation = replace(transformation, transformation == 'soft_label', 'soft')) %>%
  mutate(filtering = replace(filtering, filtering == 'JSDivergence', 'JSD')) #%>%
  #filter(annotation != 'similarity') #%>%
  #mutate(cohesion = if_else(cohesion < 0, 0, cohesion))

alpha=0.5
width=0.15
#file <- knitr::fig_path('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_JSD.pdf')
#pdf('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_JSD.pdf', width=9, height=8)
# agg_tiff(file, width = 1000, height = 500, res = 600)
p <- ggplot(dt, aes(y = jsd , x=threshold, fill = threshold, color=threshold)) +
  geom_violin(alpha=alpha,  show.legend = FALSE) +
  facet_nested(~ annotation + content + algorithm + transformation) +
  theme(text = element_text(size = 12),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Filtering Threshold") + ylab("JSD") 
p
#invisible(dev.off())

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_JSD.pdf', width=9, height=4)

p <- ggplot(dt, aes(y = cohesion , x=threshold, fill = threshold, color=threshold)) +
  geom_violin(alpha=alpha,  show.legend = FALSE) +
  facet_nested(~ annotation + content + algorithm + transformation) +
  theme(text = element_text(size = 12),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Filtering Threshold") + ylab("Cohesion (%)") 
#p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_cohesion.pdf', width=9, height=4)


p <- ggplot(dt, aes(y = cohesion_all , x=threshold, fill = threshold, color=threshold)) +
  geom_violin(alpha=alpha,  show.legend = FALSE) +
  facet_nested(~ annotation + content + algorithm + transformation) +
  theme(text = element_text(size = 12),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Filtering Threshold") + ylab("Cohesion ALL (%)") 
#p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_cohesion_all.pdf', width=9, height=4)


p <- ggplot(dt, aes(y = unannotated , x=threshold, fill = threshold, color=threshold)) +
  geom_violin(alpha=alpha,  show.legend = FALSE) +
  facet_nested( ~ annotation + content  + algorithm + transformation) +
  theme(text = element_text(size = 12),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Filtering Threshold") + ylab("Unannotated (%)") 
#p

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/package_unannotated_nodes.pdf', width=9, height=4)

