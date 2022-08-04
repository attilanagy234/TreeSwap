from abc import ABC, abstractmethod

import networkx as nx


class GraphSimilarityBase(ABC):

    @abstractmethod
    def get_similarity_from_sentences(self, src_sent: str, tgt_sent: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_similarity_from_graphs(self, src_graph: nx.DiGraph, tgt_graph: nx.DiGraph) -> float:
        raise NotImplementedError
