from abc import ABC, abstractmethod
from operator import itemgetter


class AugmentatorBase(ABC):
    """
    Base class for different data augmentator algorithms
    """

    def __init__(self, config):
        self._config = config

    @abstractmethod
    def augment_sentence_from_dep_graph(self, dep_graph):
        raise NotImplementedError

    @staticmethod
    def reconstruct_sentence_from_node_ids(node_ids):
        return [y[0] for y in sorted([x.split('-') for x in node_ids], key=itemgetter(1))]
