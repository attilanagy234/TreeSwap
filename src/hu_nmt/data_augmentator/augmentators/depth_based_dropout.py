from hu_nmt.data_augmentator.augmentators.depth_based_augmentator import DepthBasedAugmentator
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)


class DepthBasedDropout(DepthBasedAugmentator):

    def __init__(self, config):
        super().__init__(config)

    def augment_sentence_from_dep_graph(self, dep_graph):
        node_ids, indices_to_blank = self.get_word_indicies_to_blank(dep_graph)
        sentence = self.reconstruct_sentence_from_node_ids(node_ids)
        log.debug(f'Original list of tokens: {sentence}')
        sentence = [i for j, i in enumerate(sentence) if j not in set(indices_to_blank)]
        log.debug(f'A list of tokens: {sentence}')
        return ' '.join(sentence)
