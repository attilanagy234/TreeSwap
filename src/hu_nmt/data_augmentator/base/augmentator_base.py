from abc import ABC, abstractmethod


class AugmentatorBase(ABC):
    """
    Base class for different data augmentator algorithms
    """

    def __init__(self, config):
        self._config = config

    @abstractmethod
    def augment_sentence_from_dep_graph(self, dep_graph):
        raise NotImplementedError
