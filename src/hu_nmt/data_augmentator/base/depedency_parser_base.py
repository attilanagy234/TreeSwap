import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import floor, ceil
from typing import List, Iterator, Generator, Callable
import multiprocessing as mp
import networkx as nx
import psutil
from tqdm import tqdm

from hu_nmt.data_augmentator.utils.data_helpers import get_files_in_folder
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper

log = get_logger(__name__)


@dataclass
class NodeRelationship:
    target_key: str
    target_postag: str
    target_lemma: str
    target_deprel: str
    source_key: str
    source_postag: str
    source_lemma: str


Pipeline = Callable[[str], object]


@dataclass
class SentenceProcessUnit:
    pipeline: Pipeline
    sentence: str


@dataclass
class SentenceProcessBatch:
    pipeline_constructor: Callable[[], Pipeline]
    sentences: List[str]


class DependencyParserBase(ABC):
    """
    Base class for language-specific dependency parsers
    """

    def __init__(self, nlp_pipeline_constructor: Callable[[], Pipeline], use_multiprocessing: bool):
        self.nlp_pipeline_constructor: Callable[[], Pipeline] = nlp_pipeline_constructor
        self.nlp_pipeline = self.nlp_pipeline_constructor()
        self.use_multiprocessing = use_multiprocessing
        if self.use_multiprocessing:
            log.info('Using multiprocessing')

    @abstractmethod
    def sentence_to_dep_parse_tree(self, sentence):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sentence_to_node_relationship_list(nlp_pipeline, sent: str) -> List[NodeRelationship]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _sentence_process_unit_to_node_relationship_list(process_unit: SentenceProcessUnit) -> List[NodeRelationship]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _sentence_process_batch_to_node_relationship_list(process_batch: SentenceProcessBatch) \
            -> List[List[NodeRelationship]]:
        raise NotImplementedError

    @staticmethod
    def read_parsed_dep_trees_from_files(data_dir: str, per_file: bool = False) -> Generator[nx.DiGraph, None, None]:
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            """
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            """
            return [atoi(c) for c in re.split(r'(\d+)', text)]

        files_to_read = get_files_in_folder(data_dir)
        files_to_read.sort(key=natural_keys)
        for file in files_to_read:
            dep_graphs = []
            with open(f'{data_dir}/{file}') as f:
                graph = nx.DiGraph()
                for line in f:
                    if line == '\n':
                        if per_file:
                            dep_graphs.append(graph)
                        else:
                            yield graph
                        graph = nx.DiGraph()
                    else:
                        target_key, target_postag, target_lemma, target_deprel, \
                        source_key, source_postag, source_lemma = line.split('\t')

                        graph.add_node(source_key, postag=source_postag, lemma=source_lemma)
                        graph.add_node(target_key, postag=target_postag, lemma=target_lemma)
                        graph.add_edge(source_key, target_key, dep=target_deprel)
                if per_file:
                    yield dep_graphs

    def get_graph_wrappers_from_files(self, data_folder) -> List[DependencyGraphWrapper]:
        dep_graphs = self.read_parsed_dep_trees_from_files(data_folder)
        return [DependencyGraphWrapper(x) for x in dep_graphs]

    def file_to_serialized_dep_graph_files(self, sentences_path: str, output_dir: str, file_batch_size: int):
        sentence_generator = self.get_file_line_generator(sentences_path)
        self.sentences_to_serialized_dep_graph_files(sentence_generator, output_dir, file_batch_size)

    @staticmethod
    def get_file_line_generator(file_path: str):
        with open(file_path, 'r') as file:
            for line in file:
                yield line.strip()

    @staticmethod
    def create_mini_batches(number_of_small_batches: int, batch: List) -> List[List]:
        batch_size = len(batch)
        items_per_small_batch = ceil(batch_size / number_of_small_batches)
        small_batch_list = []
        for i in range(0, batch_size, items_per_small_batch):
            small_batch_list.append(batch[i:i + items_per_small_batch])

        small_batch_list.extend([[] for _ in range(number_of_small_batches - len(small_batch_list))])

        return small_batch_list

    def sentences_to_serialized_dep_graph_files(self, sentences_iter: Iterator[str], output_dir: str,
                                                file_batch_size: int):
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
                log.info(f'Reading batch {file_idx}')
                try:
                    for _ in range(file_batch_size):
                        batch_of_sentences.append(next(sentences_iter))
                except StopIteration:
                    have_more_sentences_to_process = False
                    if len(batch_of_sentences) == 0:
                        break

                if self.use_multiprocessing:
                    if first_run:
                        # decide how many processes to run for optimal speed
                        process = psutil.Process(os.getpid())
                        process_used_bytes = process.memory_info().rss
                        available_bytes = psutil.virtual_memory().available
                        log.info(f'There could be {available_bytes / process_used_bytes} processes spawned '
                                 f'to fill up the available memory')
                        process_count = min(max(floor(available_bytes / process_used_bytes), 1), mp.cpu_count())
                        log.info(f'Decided to use {process_count} processes based on available memory')
                        first_run = False

                    # create mini batches and package them with pipeline constructors
                    mini_batches_of_sentences = self.create_mini_batches(process_count, batch_of_sentences)
                    process_batches = [SentenceProcessBatch(self.nlp_pipeline_constructor, mini_batch)
                                       for mini_batch in mini_batches_of_sentences]

                    # map to processes
                    log.info('Mapping sentences to processes')
                    proc_pool = mp.get_context('spawn').Pool(process_count)
                    list_of_results = proc_pool.imap(self._sentence_process_batch_to_node_relationship_list,
                                                     process_batches,
                                                     chunksize=max(int(len(batch_of_sentences) / process_count), 1))
                    proc_pool.close()
                    proc_pool.join()

                    # unpack results
                    log.info('Unpacking results')
                    list_of_dep_rel_lists = []
                    for result in list_of_results:
                        list_of_dep_rel_lists.extend(result)
                else:
                    process_units = [SentenceProcessUnit(self.nlp_pipeline, sentence)
                                     for sentence in batch_of_sentences]
                    list_of_dep_rel_lists = map(self._sentence_process_unit_to_node_relationship_list, process_units)

                # dump to file
                self.write_dep_graphs_to_file(output_dir, file_idx, list_of_dep_rel_lists)

                pbar.update(len(batch_of_sentences))

                file_idx += 1
                batch_of_sentences = []

    @staticmethod
    def write_dep_graphs_to_file(output_dir, file_idx, list_of_dep_rel_lists):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, f'{file_idx}.tsv'), 'w') as output_file:
            for dep_rel_list in list_of_dep_rel_lists:
                for dep_rel in dep_rel_list:
                    graph_record = f'{dep_rel.target_key}\t{dep_rel.target_postag}\t{dep_rel.target_lemma}' \
                                   f'\t{dep_rel.target_deprel}\t{dep_rel.source_key}\t{dep_rel.source_postag}\t{dep_rel.source_lemma}\n'
                    output_file.write(graph_record)
                output_file.write('\n')
