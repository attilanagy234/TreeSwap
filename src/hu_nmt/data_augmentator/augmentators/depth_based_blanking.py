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
        node_ids, probs = self.get_probabilities_from_tree_depth(dep_graph)
        words_to_augment = self.sample_from_distribution(node_ids, probs, 3)
        log.debug(f'Words selected to augment: {words_to_augment}')
        indices_to_blank = [int(x.split('-')[1]) for x in words_to_augment]
        sentence = self.reconstruct_sentence_from_node_ids(node_ids)
        log.debug(f'Original list of tokens: {sentence}')
        for idx in indices_to_blank:
            # idx-1, because we removed the artifical ROOT node
            # from the first position in the sentence
            sentence[idx-1] = self.BLANK
        log.debug(f'A list of tokens: {sentence}')
        return ' '.join(sentence)
