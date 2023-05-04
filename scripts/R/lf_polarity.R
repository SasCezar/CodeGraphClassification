library(tidyverse)
library(ggplot2)
library(reshape2)
library(dplyr)
library(ggh4x)


dt <- read.csv("/home/sasce/PycharmProjects/CodeGraphClassification/src/pipelines/polarity.csv")
dt$threshold <- as.factor(dt$threshold)
dt <- dt %>% 
  filter(annotation != 'ensemble') %>%
  mutate(transformation = replace(transformation, transformation == 'single_label', 'single')) %>%
  mutate(transformation = replace(transformation, transformation == 'soft_label', 'soft')) %>%
  mutate(annotation = stringr::str_to_title(annotation)) %>%
  mutate(content = stringr::str_to_title(content)) %>%
  mutate(transformation = stringr::str_to_title(transformation)) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'yake', 'Yake')) %>%
  mutate(algorithm = replace(algorithm, algorithm == 'w2v-so', 'W2V-SO'))

p <- ggplot(dt, aes(y = polarity , x=threshold, fill = threshold, color=threshold)) +
  geom_bar(stat="identity", show.legend = FALSE) +
  geom_text(aes(label=polarity), vjust=-0.3, size=2,  color="black") +
  facet_nested(~ annotation + content + algorithm + transformation) +
  coord_cartesian(ylim = c(70, 270)) +
  theme(text = element_text(size = 10),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  geom_hline(yintercept=267, linetype="dashed", color = "red") +
  xlab("Filtering Threshold") + ylab("Polarity") 
p


ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/statistics/lf_polarity_short.pdf', width=4, height=3.5)