from abc import ABC, abstractmethod
from typing import Tuple, Dict

from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper

Edge = Tuple[str, str, Dict]
Node = str


class GraphSimilarityBase(ABC):

    @abstractmethod
    def get_similarity_from_graphs(self, src_graph: DependencyGraphWrapper, tgt_graph: DependencyGraphWrapper) -> float:
        raise NotImplementedError
