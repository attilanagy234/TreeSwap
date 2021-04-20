from abc import ABC, abstractmethod


class DependencyParserBase(ABC):
    """
    Base class for language-specific dependency parsers
    """
    def __init__(self, subsample_df):
        self._subsample_df = subsample_df

    @abstractmethod
    def sentence_to_dep_parse_tree(self):
        raise NotImplementedError

    @abstractmethod
    def dep_parse_tree_to_nx_graph(self):
        raise NotImplementedError
