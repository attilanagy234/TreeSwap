import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from math import floor
from typing import List, Iterator, Tuple

import networkx as nx
import psutil as psutil
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

    @staticmethod
    def sentence_to_node_relationship_list(nlp_pipeline, sent: str) -> List[NodeRelationship]:
        doc = nlp_pipeline(sent)
        # We most likely will only pass single sentences.
        if len(doc.sentences) != 1:
            log.info(f'Sample has multiple sentences: {[s.text for s in doc.sentences]}')
        sent = doc.sentences[0]
        word_dicts = [word.to_dict() for word in sent.words]

        return [EnglishDependencyParser.extract_info_from_token(word, word_dicts) for word in word_dicts]

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

    def file_to_serialized_dep_graph_files(self, sentences_path: str, output_dir: str, file_batch_size: int):
        sentence_generator = self.get_file_line_generator(sentences_path)
        self.sentences_to_serialized_dep_graph_files(sentence_generator, output_dir, file_batch_size)

    @staticmethod
    def get_file_line_generator(file_path: str):
        with open(file_path, 'r') as file:
            for line in file:
                yield line

    @staticmethod
    def _sentence_pipeline_pair_to_node_relationship_list(pair: Tuple[stanza.Pipeline, str]):
        return EnglishDependencyParser.sentence_to_node_relationship_list(pair[0], pair[1])

    def sentences_to_serialized_dep_graph_files(self, sentences_iter: Iterator[str], output_dir: str, file_batch_size: int):
        """
        Args:
            sentences_iter: iterator for the sentences to process
            output_dir: location of tsv files containing the dep parsed sentences
            file_batch_size: amount of sentences to be parsed into a single file
        """

        batch_of_sentences = []
        first_run = True
        have_more_sentences_to_process = True
        file_idx = 1
        with tqdm() as pbar:
            while have_more_sentences_to_process:

                # read one batch
                log.info('Reading one batch')
                try:
                    for _ in range(file_batch_size):
                        batch_of_sentences.append((self.nlp_pipeline, next(sentences_iter)))
                except StopIteration:
                    have_more_sentences_to_process = False

                if first_run:
                    # decide how many processes to run for optimal speed
                    process = psutil.Process(os.getpid())
                    process_used_bytes = process.memory_info().rss
                    available_bytes = psutil.virtual_memory().available
                    log.info(f'There could be {available_bytes / process_used_bytes} processes spawned to fill up the available memory')
                    process_count = max(floor(available_bytes / process_used_bytes), 1)
                    log.info(f'Decided to use {process_count} processes based on available memory')
                    first_run = False

                # map to processes
                log.info('Mapping sentences to processes')
                proc_pool = mp.Pool(process_count)
                list_of_dep_rel_lists = proc_pool.imap(self._sentence_pipeline_pair_to_node_relationship_list,
                                                       batch_of_sentences,
                                                       chunksize=int(len(batch_of_sentences)/process_count))
                proc_pool.close()
                proc_pool.join()

                # dump to file
                with open(os.path.join(output_dir, f'{file_idx}.tsv'), 'w') as output_file:
                    for dep_rel_list in list_of_dep_rel_lists:
                        for dep_rel in dep_rel_list:
                            graph_record = f'{dep_rel.target_key}\t{dep_rel.target_postag}\t{dep_rel.target_lemma}' \
                                           f'\t{dep_rel.target_deprel}\t{dep_rel.source_key}\t{dep_rel.source_postag}\t{dep_rel.source_lemma}\n'
                            output_file.write(graph_record)
                        output_file.write('\n')

                pbar.update(len(batch_of_sentences))

                file_idx += 1
