from hu_nmt.data_augmentator.augmentators.depth_based_augmentator import DepthBasedAugmentator


class DepthBasedReplacement(DepthBasedAugmentator):

    def __init__(self):
        super().__init__()

    def augment(self, dep_graph):
        pass
