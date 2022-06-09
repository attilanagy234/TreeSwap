from abc import ABC, abstractmethod


class GraphSimilarityBase(ABC):

    @abstractmethod
    def get_similarity_from_sentences(self, src_sent, tgt_sent):
        raise NotImplementedError

    @abstractmethod
    def get_similarity_from_graphs(self, src_graph, tgt_graph):
        raise NotImplementedError
