from abc import ABC, abstractmethod

import igraph


class GraphLoader(ABC):
    @abstractmethod
    def load(self, path):
        raise NotImplemented


class ArcanGraphLoader(GraphLoader):
    def __init__(self, clean=False):
        self.clean_edges = ["isChildOf", "isImplementationOf", "nestedTo",
                            "belongsTo", "implementedBy", "definedBy",
                            "containerIsAfferentOf", "unitIsAfferentOf"]
        self.clean = clean

    def load(self, path):
        graph = igraph.Graph.Read_GraphML(path)
        graph = self._clean_graph(graph) if self.clean else graph
        return graph

    def _clean_graph(self, graph):
        graph.es['weight'] = graph.es['Weight']
        delete = [x.index for x in graph.vs if "$" in x['name']]
        graph.delete_vertices(delete)
        for edge_label in self.clean_edges:
            graph.es.select(labelE=edge_label).delete()

        graph.vs.select(_degree=0).delete()

        return graph
