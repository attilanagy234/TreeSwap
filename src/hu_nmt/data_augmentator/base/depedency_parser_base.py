import re
from typing import List

import networkx as nx
from abc import ABC, abstractmethod
from hu_nmt.data_augmentator.utils.data_helpers import get_files_in_folder
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
from tqdm import tqdm


class DependencyParserBase(ABC):
    """
    Base class for language-specific dependency parsers
    """

    def __init__(self):
        pass

    @abstractmethod
    def sentence_to_dep_parse_tree(self, sentence):
        raise NotImplementedError

    @staticmethod
    def read_parsed_dep_trees_from_files(data_dir: str) -> List[nx.DiGraph]:
        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            return [atoi(c) for c in re.split(r'(\d+)', text)]

        files_to_read = get_files_in_folder(data_dir)
        files_to_read.sort(key=natural_keys)
        dep_graphs = []
        for file in tqdm(files_to_read):
            with open(f'{data_dir}/{file}') as f:
                graph = nx.DiGraph()
                for line in f:
                    if line == '\n':
                        dep_graphs.append(graph)
                        graph = nx.DiGraph()
                    else:
                        target_key, target_postag, target_lemma, target_deprel, \
                        source_key, source_postag, source_lemma = line.split('\t')

                        graph.add_node(source_key, postag=source_postag, lemma=source_lemma)
                        graph.add_node(target_key, postag=target_postag, lemma=target_lemma)
                        graph.add_edge(source_key, target_key, dep=target_deprel)
        return dep_graphs

    def get_graph_wrappers_from_files(self, data_folder) -> List[DependencyGraphWrapper]:
        dep_graphs = self.read_parsed_dep_trees_from_files(data_folder)
        return [DependencyGraphWrapper(x) for x in dep_graphs]
