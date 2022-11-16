library(tidyverse)
library(ggplot2)
library(reshape2)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/annotations/name/GenomicParisCentre|eoulsan-25-fb040413e709e2463e9a99cbf88da25e27d0732b.csv")
#dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/annotations/name/SonarSource|sonar-java-8464-6400749499be832a3e37fa2f6beed47f47c04f36.csv")
#
#dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/data/processed/annotations/name/Waikato|weka-3.8-903-04804ccd6dff03534cbf3f2a71a35c73eef24fe8.csv")

dt2 <-melt(dt)

ggplot(dt2, aes(x = variable , y = node, fill = value)) +
  geom_raster() + 
  theme_minimal() +
  theme( 
    legend.position = "none",
    plot.margin=margin(grid::unit(0, "cm")),
    panel.border = element_blank(),
    panel.grid = element_blank(),
    panel.spacing = element_blank()) + 
  #scale_x_discrete(labels = seq(0, 266, by=1)) + 
  scale_x_discrete(labels = NULL, breaks = NULL) + 
  scale_y_discrete(labels = NULL, breaks = NULL) + 
  labs(x = "Labels", y="Nodes/Files")

ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/annotations/eoulsan_name_node_label_last.pdf',
       width = 12, height = 4,  device=cairo_pdf())


#mean_dt <- dt2 %>%
#  group_by(variable) %>%
#  summarise_at(vars(value), list(prob = mean))

#mean_col <- rep(c('mean'), times=nrow(mean_dt))
#mean_dt$mean <- mean_col

#proj_labels <- ggplot(mean_dt, aes(x = variable, y = mean, fill = prob)) +
#  geom_tile() + 
#  scale_x_discrete(labels = NULL, breaks = NULL) + 
#  scale_y_discrete(labels = NULL, breaks = NULL) + 
#  labs(title = project, x = "Labels", y="Nodes/Files")

#ggarrange(node_labels, proj_labels, heights = c(2, 0.7),
#          ncol = 1, nrow = 2)

#ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/annotations/weka_name_node_label.pdf',
#       width = 12, height = 4, device=cairo_pdf())