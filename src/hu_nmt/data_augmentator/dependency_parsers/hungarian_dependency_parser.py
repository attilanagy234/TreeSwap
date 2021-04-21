from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
import spacy
import networkx as nx


class HungarianDependencyParser(DependencyParserBase):
    def __init__(self):
        super().__init__()

    def sentence_to_dep_parse_tree(self, sentence):
        pass
