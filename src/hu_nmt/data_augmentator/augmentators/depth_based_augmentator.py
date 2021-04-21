from abc import ABC

from hu_nmt.data_augmentator.base.augmentator_base import AugmentatorBase


class DepthBasedAugmentator(AugmentatorBase):
    """
    Common-parent for all augmentation techniques
    that exploit dependency tree depth from the paper:
    Syntax-aware Data Augmentation for Neural Machine Translation
    https://arxiv.org/pdf/2004.14200.pdf
    """
    def __init__(self, config):
        super().__init__(config)

    def augment_sentence(self, sentence):
        raise NotImplementedError()
