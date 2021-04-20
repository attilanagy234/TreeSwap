from abc import ABC, abstractmethod


class DependencyParserBase(ABC):
    """
    Base class for language-specific dependency parsers
    """

    def __init__(self, subsample_df):
        self._subsample_df = subsample_df  # subsample of the original dataset, used for augmentation

    @abstractmethod
    def sentence_to_dep_parse_tree(self, sentence):
        raise NotImplementedError
