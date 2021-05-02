from hu_nmt.data_augmentator.base.augmentator_base import AugmentatorBase
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
import numpy as np
from tqdm import tqdm

log = get_logger(__name__)


class SubjectObjectAugmentator(AugmentatorBase):

    def __init__(self, eng_graphs, hun_graphs):
        super().__init__()
        if len(eng_graphs) != len(hun_graphs):
            raise ValueError('Length of sentences must be equal for both langugages')
        self._eng_graphs = eng_graphs
        self._hun_graphs = hun_graphs
        self._augmentation_candidate_sentence_pairs = []

    def group_candidates_by_predicate_lemmas(self):
        lemmas_to_graphs = {}  # tuple(hun_lemma, eng_lemma) --> tuple(hun_graph, eng graph)

        for hun_graph, eng_graph in self._augmentation_candidate_sentence_pairs:
            hun_nsubj_edge = hun_graph.get_edges_with_property('dep', 'nsubj')[0]  # filtered candidates will only have one
            eng_nsubj_edge = eng_graph.get_edges_with_property('dep', 'nsubj')[0]

            hun_predicate_lemma = hun_graph.graph.nodes[hun_nsubj_edge.source_node]['lemma'].strip()
            eng_predicate_lemma = eng_graph.graph.nodes[eng_nsubj_edge.source_node]['lemma'].strip()
            lemmas_key = (hun_predicate_lemma, eng_predicate_lemma)
            graph_tup = (hun_graph, eng_graph)
            if lemmas_key not in lemmas_to_graphs:
                lemmas_to_graphs[lemmas_key] = []
                lemmas_to_graphs[lemmas_key].append(graph_tup)
            else:
                lemmas_to_graphs[lemmas_key].append(graph_tup)

        return lemmas_to_graphs

    def augment(self, num_sentences_to_produce, random_seed):
        np.random.seed = random_seed
        log.info('Finding augmentable sentence pairs...')
        self.find_augmentable_candidates()
        log.info(f'Found {len(self._augmentation_candidate_sentence_pairs)} candidate sentence pairs')
        lemmas_to_graphs = self.group_candidates_by_predicate_lemmas()

        # print common lemmas starting with the most frequent ones.
        predicate_lemmas_sorted_by_freq = [k for k in sorted(lemmas_to_graphs, key=lambda x: len(lemmas_to_graphs[x]), reverse=True)]
        print(predicate_lemmas_sorted_by_freq)
        print(predicate_lemmas_sorted_by_freq[0])

        # most common key experiment
        for hun_graph, eng_graph in lemmas_to_graphs[predicate_lemmas_sorted_by_freq[0]][3:6]:
            hun_graph.display_graph()
            eng_graph.display_graph()
            # Remove root node from beginning of sentence
            print(' '.join(self.reconstruct_sentence_from_node_ids(hun_graph.graph.nodes)[1:]))
            print(' '.join(self.reconstruct_sentence_from_node_ids(eng_graph.graph.nodes)[1:]))






    def find_augmentable_candidates(self):
        for hun_graph, eng_graph in tqdm(zip(self._hun_graphs, self._eng_graphs)):
            if self.test_graph_pair(hun_graph, eng_graph):
                self._augmentation_candidate_sentence_pairs.append((hun_graph, eng_graph))


    def test_graph_pair(self, hun_graph: DependencyGraphWrapper, eng_graph: DependencyGraphWrapper) -> bool:
        """
        Tests if a sentence (graph) pair is eligible for augmentation
        """

        hun_nsubj_edges = hun_graph.get_edges_with_property('dep', 'nsubj')
        eng_nsubj_edges = eng_graph.get_edges_with_property('dep', 'nsubj')

        hun_obj_edges = hun_graph.get_edges_with_property('dep', 'obj')
        eng_obj_edges = eng_graph.get_edges_with_property('dep', 'obj')

        # Should contain one nsubj and one obj in both languages
        if len(hun_nsubj_edges) != 1 or len(eng_nsubj_edges) != 1 or len(hun_obj_edges) != 1 or len(eng_obj_edges) != 1:
            return False
        else:
            hun_nsubj_edge = hun_nsubj_edges[0]
            eng_nsubj_edge = eng_nsubj_edges[0]
            hun_obj_edge = hun_obj_edges[0]
            eng_obj_edge = eng_obj_edges[0]

        # nsubj and obj edges have the same ancestor (predicate)
        if hun_nsubj_edge.source_node != hun_obj_edge.source_node or eng_nsubj_edge.source_node != eng_obj_edge.source_node:
            return False
        object_hun = hun_obj_edge.target_node
        object_eng = eng_obj_edge.target_node
        hun_obj_subgraph = hun_graph.get_subtree_node_ids(object_hun)
        eng_obj_subgraph = eng_graph.get_subtree_node_ids(object_eng)

        # Object subtree is consecutive
        if not self.is_consecutive_subsequence(hun_obj_subgraph):
            return False
        if not self.is_consecutive_subsequence(eng_obj_subgraph):
            return False
        return True

    @staticmethod
    def is_consecutive_subsequence(node_ids):
        def check(lst):
            lst = sorted(lst)
            if lst:
                return lst == list(range(lst[0], lst[-1] + 1))
            else:
                return True
        """
        Params:
            node_ids (list of Strings): list of node ids
        Returns:
            Boolean value whether the words corresponding to nodes
             are a consecutive subsequence in the original sentence
        """
        return check([int(x.split('-')[-1]) for x in node_ids])


