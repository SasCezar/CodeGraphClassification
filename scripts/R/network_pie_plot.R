library(tidyverse)
library(igraph)
library(ggraph)
library(graphlayouts)
library(ggforce)
library(scatterpie)
require(tidygraph)


set.seed(1439)

g<-read.graph("/home/sasce/PycharmProjects/CodeGraphClassification/scripts/python/Weka_node_241_subgraph.graphml", format = "gml")

#g<-read.graph("/home/sasce/PycharmProjects/CodeGraphClassification/scripts/python/Weka_node_1447_subgraph.graphml", format = "gml")

#precompute the layout
xy <- layout_with_stress(g)
#xy <- layout_with_umap(g,min_dist = 0.5)
V(g)$x <- xy[, 1]
V(g)$y <- xy[, 2]
ggraph(g, "manual", x = V(g)$x, y = V(g)$y) +
  geom_edge_link(
    aes(end_cap = circle(10, "pt")),
    edge_colour = "black",
  ) +
  geom_scatterpie(
    cols = c("machinelearning", "randomforest", "testautomation", "clusteranalysis", "database", "Other", "Unannotated"),
    data = as_data_frame(g, "vertices"),
    colour = "white",
    pie_scale = 3
  ) +
  geom_node_text(aes(label=simpleName, vjust = "outward"), size=5.5, vjust="inward", hjust="inward") +
  scale_fill_brewer(name=NULL, palette = "Spectral", labels = c("Machine Learning", "Random Forest", "Test Automation", "Cluster Analysis", "Database", "Other", "Unlabeled")) +
  coord_fixed() +
  theme_graph(base_family="sans") +
  theme(legend.position = "top", text = element_text(size = 18), plot.margin=grid::unit(c(0,0,0,0), "mm"))



ggsave('/home/sasce/PycharmProjects/CodeGraphClassification/reports/plots/annotations/weka_1307_graph_nodes_annotation.png')