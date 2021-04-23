from hu_nmt.data_augmentator.augmentators.depth_based_augmentator import DepthBasedAugmentator
from hu_nmt.data_augmentator.dependency_graph_wrapper import DependencyGraphWrapper

from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)
log.setLevel('DEBUG')


class DepthBasedBlanking(DepthBasedAugmentator):

    def __init__(self, config):
        super().__init__(config)
        self.BLANK = 'BLANK'

    def augment_sentence_from_dep_graph(self, dep_graph: DependencyGraphWrapper):
        node_ids, indices_to_blank = self.get_word_indicies_to_blank(dep_graph)
        sentence = self.reconstruct_sentence_from_node_ids(node_ids)
        log.debug(f'Original list of tokens: {sentence}')
        for idx in indices_to_blank:
            sentence[idx] = self.BLANK
        log.debug(f'A list of tokens: {sentence}')
        return ' '.join(sentence)
