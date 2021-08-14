from typing import List

import networkx as nx
import stanza

from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase, NodeRelationship
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)

ROOT_KEY = 'root_0'


class EnglishDependencyParser(DependencyParserBase):
    def __init__(self):
        self.nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')
        super().__init__(self.nlp_pipeline, use_multiprocessing=True)

    @staticmethod
    def sentence_to_node_relationship_list(nlp_pipeline, sent: str) -> List[NodeRelationship]:
        doc = nlp_pipeline(sent)
        # We most likely will only pass single sentences.
        if len(doc.sentences) != 1:
            log.info(f'Sample has multiple sentences: {[s.text for s in doc.sentences]}')
        sent = doc.sentences[0]

        node_relationship_list = []
        word_dicts = [word.to_dict() for word in sent.words]
        for word in sent.words:
            token = word.to_dict()
            target_key = f'{token["text"].lower()}_{token["id"]}'
            target_postag = token['upos']
            target_lemma = token['lemma']
            target_deprel = token['deprel']
            if token['head'] == 0:
                source_key = ROOT_KEY
                source_postag = None
                source_lemma = None
            else:
                head = word_dicts[int(token['head']) - 1]
                source_key = f'{head["text"].lower()}_{head["id"]}'
                source_postag = head['upos']
                source_lemma = head['lemma']

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
        # Add ROOT node
        dep_graph.add_node(ROOT_KEY)
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
        return EnglishDependencyParser.sentence_to_node_relationship_list(pair[0], pair[1])
