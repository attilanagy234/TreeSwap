import spacy
import networkx as nx
from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase


class EnglishDependencyParser(DependencyParserBase):

    def __init__(self):
        super().__init__()
        self._nlp_pipe = spacy.load('en')

    def sentence_to_dep_parse_tree(self, sentence):
        """
        Args:
            sentence: space separated string of the input sentence
        Returns:
            A directed (networkx) graph representation of the dependency tree
        """
        doc = self._nlp_pipe(sentence)
        dep_graph = nx.DiGraph()
        for token in doc:   # token API: https://spacy.io/docs/api/token
            for child in token.children:
                source_key = f'{token.lower_}-{token.i}'
                target_key = f'{child.lower_}-{child.i}'
                dep_graph.add_node(source_key, postag=token.pos_)  # source node with POS tag
                dep_graph.add_node(target_key, postag=child.pos_)  # target node with POS tag
                dep_graph.add_edge(source_key, target_key, dep=child.dep_)  # edge with dependency relation
        return dep_graph
