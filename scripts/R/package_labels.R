library(tidyverse)
library(igraph)
library(ggraph)
library(graphlayouts)
library(concaveman) 
library(ggforce)

g<-read.graph("/home/sasce/PycharmProjects/CodeGraphClassification/scripts/python/Waikato|weka-3.8_top_5.gml", format = "gml")

components <- igraph::clusters(g, mode="weak")
biggest_cluster_id <- which.max(components$csize)

# ids
vert_ids <- V(g)[components$membership == biggest_cluster_id]

# subgraph
g <- igraph::induced_subgraph(g, vert_ids)

g  <- delete_vertices(g, V(g)$labelV != 'container')
ggraph(g, 'treemap', aes(numfiles)) + 
  geom_node_tile(aes(fill = packagelabel), size = 0.25)