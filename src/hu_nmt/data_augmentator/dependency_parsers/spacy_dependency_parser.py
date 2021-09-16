from typing import List

import hu_core_ud_lg
import networkx as nx
import spacy

from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase, NodeRelationship
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)

ROOT_KEY = 'root_0'


class SpacyDependencyParser(DependencyParserBase):
    def __init__(self, lang):
        if lang == 'hu':
            self.nlp_pipeline = hu_core_ud_lg.load()
        elif lang == 'de':
            self.nlp_pipeline = spacy.load("de_core_news_sm")
        elif lang == 'fr':
            self.nlp_pipeline = spacy.load("fr_core_news_sm")
        else:
            raise ValueError(f'Language {lang} is not supported by the SpacyDependencyParser.')
        super().__init__(self.nlp_pipeline, use_multiprocessing=False)

    @staticmethod
    def sentence_to_node_relationship_list(nlp_pipeline, sent: str) -> List[NodeRelationship]:
        doc = nlp_pipeline(sent)
        sents = [s for s in doc.sents]
        if str(sents[-1]) == '\n':
            del sents[-1]
        # We most likely will only pass single sentences.
        if len(sents) > 1:
            log.info(f'Sample has multiple sentences: {sents}')
        doc_sent = sents[0]

        node_relationship_list = []
        for token in doc_sent:
            target_key = f'{token}_{token.i + 1}'
            target_postag = token.pos_
            target_lemma = token.lemma_
            target_deprel = token.dep_
            if target_deprel == 'ROOT':
                source_key = ROOT_KEY
                source_postag = None
                source_lemma = None
            else:
                source_key = f'{token.head}_{token.head.i + 1}'
                source_postag = token.head.pos_
                source_lemma = token.head.lemma_

            node_relationship_list.append(NodeRelationship(target_key, target_postag, target_lemma, target_deprel, source_key, source_postag, source_lemma))

        return node_relationship_list

    def sentence_to_dep_parse_tree(self, sent):
        """
        Args:
            sent: space separated string of the input sentence
        Returns:
            A directed (networkx) graph representation of the dependency tree
        """
        dep_graph = nx.DiGraph()
        for node_rel in self.sentence_to_node_relationship_list(self.nlp_pipeline, sent):
            dep_graph.add_node(node_rel.source_key, postag=node_rel.source_postag, lemma=node_rel.source_lemma)
            dep_graph.add_node(node_rel.target_key, postag=node_rel.target_postag, lemma=node_rel.target_lemma)
            dep_graph.add_edge(node_rel.source_key, node_rel.target_key, dep=node_rel.target_deprel)
        return dep_graph

    @staticmethod
    def _sentence_pipeline_pair_to_node_relationship_list(pair):
        """
        Args:
            pair: tuple[pipeline, sentence]
        """
        return SpacyDependencyParser.sentence_to_node_relationship_list(pair[0], pair[1])
