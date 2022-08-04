from abc import ABC, abstractmethod

import networkx as nx
from typing import Tuple, Dict

Edge = Tuple[str, str, Dict]
Node = str

class GraphSimilarityBase(ABC):

    @abstractmethod
    def get_similarity_from_graphs(self, src_graph: nx.DiGraph, tgt_graph: nx.DiGraph) -> float:
        raise NotImplementedError

    def _get_root(self, graph: nx.DiGraph):
        node = [n for n, d in graph.in_degree() if d == 0][0]
        return node
