from abc import ABC, abstractmethod


class DependencyParserBase(ABC):
    """
    Base class for language-specific dependency parsers
    """

    def __init__(self):
        pass

    @abstractmethod
    def sentence_to_dep_parse_tree(self, sentence):
        raise NotImplementedError
