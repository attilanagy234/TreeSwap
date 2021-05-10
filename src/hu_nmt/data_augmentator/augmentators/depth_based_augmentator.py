import numpy as np

from hu_nmt.data_augmentator.base.augmentator_base import AugmentatorBase
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)
log.setLevel('DEBUG')


class DepthBasedAugmentator(AugmentatorBase):
    """
    Common-parent for all augmentation techniques
    that exploit dependency tree depth from the paper:
    Syntax-aware Data Augmentation for Neural Machine Translation
    https://arxiv.org/pdf/2004.14200.pdf
    """

    def __init__(self):
        super().__init__()

    def augment(self, dep_graph: DependencyGraphWrapper):
        raise NotImplementedError()

    def get_probabilities_from_tree_depth(self, dep_graph: DependencyGraphWrapper):
        def get_depth_factor(depth):
            return 1 - (1 / (2 ** (depth - 1)))

        # No need to include the artificial ROOT node in augmentation
        distances_from_root = dep_graph.get_distances_from_root()
        distances_from_root.pop('root-0', None)
        distances_from_root = distances_from_root.items()

        node_ids = [x[0] for x in distances_from_root]
        depth_factors = [get_depth_factor(x[1]) for x in distances_from_root]
        log.debug(f'Distances from root: {distances_from_root}')
        log.debug(f'Depth factors: {depth_factors}')
        probs = self.softmax(np.array(depth_factors))
        log.debug(f'Probabilities: {probs}')
        return node_ids, probs

    @staticmethod
    def sample_from_distribution(words, probs, sample_count=1):
        return set(np.random.choice(words, sample_count, p=probs))

    @staticmethod
    def softmax(x):
        return np.exp(x) / sum(np.exp(x))

    def get_word_indicies_to_blank(self, dep_graph: DependencyGraphWrapper):
        node_ids, probs = self.get_probabilities_from_tree_depth(dep_graph)
        words_to_augment = self.sample_from_distribution(node_ids, probs, 3)
        log.debug(f'Words selected to augment: {words_to_augment}')
        # Subtract 1 from index,
        # because we removed the artifical ROOT node
        # from the first position in the sentence
        indices_to_blank = [int(x.split('_')[-1])-1 for x in words_to_augment]
        return node_ids, indices_to_blank
