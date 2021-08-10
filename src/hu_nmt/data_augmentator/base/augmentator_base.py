from abc import ABC, abstractmethod
from operator import itemgetter
from typing import List


class AugmentatorBase(ABC):
    """
    Base class for different data augmentator algorithms
    """

    def __init__(self):
        pass

    @abstractmethod
    def augment(self, *args):
        raise NotImplementedError

    @staticmethod
    def reconstruct_sentence_from_node_ids(node_ids: list[str]) -> List[str]:
        splitted_node_ids = [x.split('_') for x in node_ids]
        splitted_node_ids = [(x[0], int(x[1])) for x in splitted_node_ids]
        return [y[0] for y in sorted(splitted_node_ids, key=itemgetter(1))]
