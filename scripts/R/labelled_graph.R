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


#Layout
gs$membership <- V(g)$group
weights <- ifelse(igraph::crossing(gs, g), 1.5, 1)

layout_name <- "stress"
glay <- create_layout(g,  layout_name, weights=weights)


layout_name <- "focus"
glay <- create_layout(g,  layout_name, focus=length(V(g)))


ggraph(glay) +
  geom_edge_link0(width=0.2,colour="grey")+
  geom_node_point(aes(color = topic, shape=labelV), size=3)+
  geom_mark_hull(
    aes(x=x, y=y, fill=group, filter=group!='None'),
    #concavity = 2,
    #expand = unit(5, "mm"),
    alpha = 0.25
  ) +
#  scale_shape_manual(values=c(18, 21)) +
  theme_graph() +
  labs(title=layout_name)


gs <- simplify(g)
gs <- as.undirected(gs)
bb <- layout_as_backbone(gs,keep=.9)
E(gs)$col <- F
E(gs)$col[bb$backbone] <- T

ggraph(g,layout="manual", x=bb$xy[,1], y=bb$xy[,2])+
  geom_edge_link0(width=0.1)+
  geom_node_point(aes(color=topic, shape=labelV), size=3)+
  geom_mark_hull(
    aes(x=x, y=y, fill=group),
    concavity = 12324,
    expand = unit(2, "mm"),
    alpha = 0.25
  ) +
  theme_graph()
