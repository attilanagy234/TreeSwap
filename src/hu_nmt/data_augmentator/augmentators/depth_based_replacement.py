from hu_nmt.data_augmentator.augmentators.depth_based_augmentator import DepthBasedAugmentator


class DepthBasedReplacement(DepthBasedAugmentator):

    def __init__(self, config):
        super().__init__(config)

    def augment_sentence(self, sentence):
        pass
