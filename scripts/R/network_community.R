library(tidyverse)
library(igraph)
library(ggraph)
library(graphlayouts)
library(ggforce)
library(scatterpie)
require(tidygraph)


set.seed(1337)

g<-read.graph("/home/sasce/PycharmProjects/CodeGraphClassification/scripts/python/Waikato|weka-3.8_top_5.gml", format = "gml")
layout <- create_layout(g, layout = 'circlepack')

ggraph(layout) + 
  geom_edge_link() + 
  geom_node_point(aes(colour = topic)) +
  coord_fixed()


ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/annotations/weka_project.png')



graph <- as_tbl_graph(g) %>% 
  mutate(degree = centrality_degree())
lapply(c('stress', 'fr', 'lgl', 'graphopt'), function(layout) {
  ggraph(graph, layout = layout) + 
    geom_edge_link() +
    geom_node_point(aes(colour = topic), show.legend = FALSE) + 
    labs(caption = paste0('Layout: ', layout))
})