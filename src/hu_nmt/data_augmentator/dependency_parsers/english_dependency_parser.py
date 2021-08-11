from dataclasses import dataclass
from typing import List

import networkx as nx
import stanza
from tqdm import tqdm

from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)

ROOT_KEY = 'root_0'


@dataclass
class NodeRelationship:
    target_key: str
    target_postag: str
    target_lemma: str
    target_deprel: str
    source_key: str
    source_postag: str
    source_lemma: str


class EnglishDependencyParser(DependencyParserBase):

    def __init__(self):
        super().__init__()
        self.nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

    @staticmethod
    def extract_info_from_token(token: dict, token_list: List[dict]) -> NodeRelationship:
        target_key = f'{token["text"].lower()}_{token["id"]}'
        target_postag = token['upos']
        target_lemma = token['lemma']
        target_deprel = token['deprel']
        if token['head'] != 0:
            head = token_list[int(token['head']) - 1]
            source_key = f'{head["text"].lower()}_{head["id"]}'
            source_postag = head['upos']
            source_lemma = head['lemma']
        else:
            source_key = ROOT_KEY
            source_postag = None
            source_lemma = None
        return NodeRelationship(target_key, target_postag, target_lemma, target_deprel, source_key, source_postag, source_lemma)

    def sentence_to_node_relationship_list(self, sent: str) -> List[NodeRelationship]:
        doc = self.nlp_pipeline(sent)
        # We most likely will only pass single sentences.
        if len(doc.sentences) != 1:
            log.info(f'Sample has multiple sentences: {[s.text for s in doc.sentences]}')
        sent = doc.sentences[0]
        word_dicts = [word.to_dict() for word in sent.words]

        return [self.extract_info_from_token(word, word_dicts) for word in word_dicts]

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
        for node_rel in self.sentence_to_node_relationship_list(sent):
            dep_graph.add_node(node_rel.source_key, postag=node_rel.source_postag, lemma=node_rel.source_lemma)
            dep_graph.add_node(node_rel.target_key, postag=node_rel.target_postag, lemma=node_rel.target_lemma)
            dep_graph.add_edge(node_rel.source_key, node_rel.target_key, dep=node_rel.target_deprel)
        return dep_graph

    def sentences_to_serialized_dep_graph_files(self, sentences, output_dir, file_batch_size):
        """
        Args:
            sentences: list of sentences to process
            output_dir: location of tsv files containing the dep parsed sentences
            file_batch_size: amount of sentences to be parsed into a single file
        """
        file_idx = 1
        open_new_file = True
        file_batch_size = int(file_batch_size)

        for progress_idx, sent in tqdm(enumerate(sentences)):
            if open_new_file:
                file = open(f'{output_dir}/{file_idx}.tsv', 'w+')
                open_new_file = False

            for dep_rel in self.sentence_to_node_relationship_list(sent):
                graph_record = f'{dep_rel.target_key}\t{dep_rel.target_postag}\t{dep_rel.target_lemma}' \
                               f'\t{dep_rel.target_deprel}\t{dep_rel.source_key}\t{dep_rel.source_postag}\t{dep_rel.source_lemma}\n'
                file.write(graph_record)

            file.write('\n')  # Separate sentences with a new line

            if (progress_idx + 1) % file_batch_size == 0:
                file.close()
                file_idx += 1
                open_new_file = True
