from operator import itemgetter
import numpy as np
from hu_nmt.data_augmentator.base.augmentator_base import AugmentatorBase
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
from tqdm import tqdm
from itertools import combinations

log = get_logger(__name__)
log.setLevel('DEBUG')


class SubjectObjectAugmentator(AugmentatorBase):

    def __init__(self, eng_graphs, hun_graphs, augmented_data_ratio, random_seed, output_path):
        super().__init__()
        if len(eng_graphs) != len(hun_graphs):
            raise ValueError('Length of sentences must be equal for both langugages')
        self._num_augmented_sentences_to_generate_per_method = int(len(eng_graphs) * float(augmented_data_ratio))
        log.info(f'number of desired sentences/method: {self._num_augmented_sentences_to_generate_per_method}')
        np.random.seed = random_seed
        self._output_path = output_path
        self.error_cnt = 0
        self._eng_graphs = eng_graphs
        self._hun_graphs = hun_graphs
        self._augmentation_candidate_sentence_pairs = []
        self._augmented_sentence_pairs = {
            'obj_swapping_same_predicate_lemma': {
                'hun': [],
                'eng': []
            },
            'subj_swapping_same_predicate_lemma': {
                'hun': [],
                'eng': []
            },
            'subj_swapping': {
                'hun': [],
                'eng': []
            },
            'obj_swapping': {
                'hun': [],
                'eng': []
            },
            'predicate_swapping': {
                'hun': [],
                'eng': []
            }
        }

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

    def augment(self):
        log.info('Finding augmentable sentence pairs...')
        self.find_augmentable_candidates()
        log.info(f'Found {len(self._augmentation_candidate_sentence_pairs)} candidate sentence pairs')
        lemmas_to_graphs = self.group_candidates_by_predicate_lemmas()

        self.augment_subtree_swapping_with_same_predicate_lemmas(lemmas_to_graphs)
        self._augmentation_candidate_sentence_pairs = self._augmentation_candidate_sentence_pairs
        self.augment_predicate_swapping()
        self.augment_subtree_swapping()

        # for i in range(10):
        #     self.print_augmented_pairs(i)

        self.dump_augmented_sentences_to_files()

    def augment_predicate_swapping(self):
        log.info('Starting predicate swapping augmentation')
        all_permutations = list(combinations(self._augmentation_candidate_sentence_pairs, 2))
        # We divide the amount we want to generate/method by two,
        # because a subtree swapping on a sentence pairs, yields
        # two new augmented sentences.
        sample_cnt = int(self._num_augmented_sentences_to_generate_per_method / 2)
        permutations = self.sample_permutations(all_permutations, sample_cnt)
        self.swap_predicates_in_all_permutations(permutations)
        log.info('Finished predicate swapping augmentation')

    @staticmethod
    def sample_permutations(all_permutations, num_samples):
        all_indices = [x for x in range(len(all_permutations))]
        sampled_indices = np.random.choice(all_indices, num_samples, replace=False)
        return [all_permutations[idx] for idx in sampled_indices]

    def swap_predicates_in_all_permutations(self, permutations):
        for permutation in tqdm(permutations):
            try:
                hun_sents, eng_sents = self.augment_pair(permutation, 'predicate')
                self._augmented_sentence_pairs['predicate_swapping']['hun'].extend(hun_sents)
                self._augmented_sentence_pairs['predicate_swapping']['eng'].extend(eng_sents)
            except Exception as e:
                self.error_cnt += 1
                log.debug(f'Cannot process sentence: {e}')
        log.info(f'Could not perform {self.error_cnt} augmentations so far')

    def augment_subtree_swapping(self):
        """
        Swaps Obj and Nsubj subtrees within all the permutations of sentences
        """
        log.info('Starting subtree swapping on all permutations')
        all_permutations = list(combinations(self._augmentation_candidate_sentence_pairs, 2))
        # We divide the amount we want to generate/method by two,
        # because a subtree swapping on a sentence pairs, yields
        # two new augmented sentences.
        sample_cnt = int(self._num_augmented_sentences_to_generate_per_method / 2)
        permutations = self.sample_permutations(all_permutations, sample_cnt)
        self.swap_subtrees_among_permutations(permutations, same_predicate_lemma=False)
        log.info('Finished subtree swapping on all permutations')

    def swap_subtrees_among_permutations(self, permutations, same_predicate_lemma: bool):
        for permutation in tqdm(permutations):
            try:
                hun_sents, eng_sents = self.augment_pair(permutation, 'obj')
                if same_predicate_lemma:
                    key = 'obj_swapping_same_predicate_lemma'
                else:
                    key = 'obj_swapping'
                self._augmented_sentence_pairs[key]['hun'].extend(hun_sents)
                self._augmented_sentence_pairs[key]['eng'].extend(eng_sents)
                hun_sents, eng_sents = self.augment_pair(permutation, 'nsubj')
                if same_predicate_lemma:
                    key = 'subj_swapping_same_predicate_lemma'
                else:
                    key = 'subj_swapping'
                self._augmented_sentence_pairs[key]['hun'].extend(hun_sents)
                self._augmented_sentence_pairs[key]['eng'].extend(eng_sents)
            except Exception as e:
                self.error_cnt += 1
                log.debug(f'Cannot process sentence: {e}')
        log.info(f'Could not perform {self.error_cnt} augmentations so far')

    def augment_subtree_swapping_with_same_predicate_lemmas(self, lemmas_to_graphs):
        """
        Swaps Obj and Nsubj subtrees within the permutations of sentences
        that have the same same predicate lemma
        """
        predicate_lemmas_sorted_by_freq = [k for k in sorted(lemmas_to_graphs, key=lambda x: len(lemmas_to_graphs[x]),
                                                             reverse=True)]
        permutations = []
        log.info('Starting subtree swapping with same predicate lemma...')
        log.info('Precomputing same predicate lemma permutations...')
        for lemma_pair in tqdm(predicate_lemmas_sorted_by_freq):
            graph_pairs_with_same_lemma = lemmas_to_graphs[lemma_pair]
            if len(graph_pairs_with_same_lemma) == 1:  # cannot get combinations if only one sentence has that lemma
                break

            # get all permutations from graph_pairs_with_same_lemma
            permutations_per_lemma = list(combinations(graph_pairs_with_same_lemma, 2))
            permutations.extend(permutations_per_lemma)
        log.info(f'Got {len(permutations)} sentence pair permutations with the same lemma')
        sample_cnt = int(self._num_augmented_sentences_to_generate_per_method / 2)
        if len(permutations) > sample_cnt:
            permutations = self.sample_permutations(permutations, sample_cnt)

        self.swap_subtrees_among_permutations(permutations, same_predicate_lemma=True)
        log.info('Finished subtree swapping with same predicate lemmas')

    def augment_pair(self, sample_pair, augmentation_type):
        """
        Swaps the subtrees of the sentences
        Params:
            sample_pair (List of Tuples of DependencyGraphWrappers): Two pairs of Hungarian and English
                                                                     dependency parse trees selected for
                                                                     augmentation
            augmentation_type (String): defines what we swap (obj, nsubj or predicate)
        Returns:
             hun_augmented_sentences (list of Strings): Hungarian augmented sentence pairs
                                                        created from graph_pair_one and graph_pair_two
             eng_augmented_sentences (list of Strings): English augmented sentence pairs
                                                        created from graph_pair_one and graph_pair_two
        """
        hun_augmented_sentences = []
        eng_augmented_sentences = []

        hun_graph_1 = sample_pair[0][0]
        eng_graph_1 = sample_pair[0][1]
        hun_graph_2 = sample_pair[1][0]
        eng_graph_2 = sample_pair[1][1]

        # hun_graph_1.display_graph()
        # eng_graph_1.display_graph()
        # hun_graph_2.display_graph()
        # eng_graph_2.display_graph()

        if augmentation_type == 'predicate':
            hun_augmented_sentences.extend(self.swap_predicates(hun_graph_1, hun_graph_2))
            eng_augmented_sentences.extend(self.swap_predicates(eng_graph_1, eng_graph_2))
        elif augmentation_type == 'obj' or augmentation_type == 'nsubj':
            hun_augmented_sentences.extend(self.swap_subtrees(hun_graph_1, hun_graph_2, augmentation_type))
            eng_augmented_sentences.extend(self.swap_subtrees(eng_graph_1, eng_graph_2, augmentation_type))

        return hun_augmented_sentences, eng_augmented_sentences

    def swap_predicates(self, sentence_graph_1, sentence_graph_2):
        original_sentence_1 = self.reconstruct_sentence_from_node_ids(sentence_graph_1.graph.nodes)
        original_sentence_2 = self.reconstruct_sentence_from_node_ids(sentence_graph_2.graph.nodes)
        # Will have one edge only due to filtering. source node of nsubj edge --> predicate of sentence
        predicate_1 = sentence_graph_1.get_edges_with_property('dep', 'nsubj')[0].source_node
        predicate_2 = sentence_graph_2.get_edges_with_property('dep', 'nsubj')[0].source_node

        predicate_1_word, predicate_1_idx = predicate_1.split('_')
        predicate_2_word, predicate_2_idx = predicate_2.split('_')

        # Swap predicates
        original_sentence_1[int(predicate_1_idx)] = predicate_2_word
        original_sentence_2[int(predicate_2_idx)] = predicate_1_word

        return [' '.join(original_sentence_1[1:]), ' '.join(original_sentence_2[1:])]

    def swap_subtrees(self, sentence_graph_1, sentence_graph_2, subtree_type):
        original_sentence_1 = self.reconstruct_sentence_from_node_ids(sentence_graph_1.graph.nodes)
        original_sentence_2 = self.reconstruct_sentence_from_node_ids(sentence_graph_2.graph.nodes)
        subgraph_1 = self.get_subgraph_from_edge_type(sentence_graph_1, subtree_type)
        subgraph_2 = self.get_subgraph_from_edge_type(sentence_graph_2, subtree_type)
        subgraph_1_offsets = self.get_offsets_from_node_ids(subgraph_1)
        subgraph_2_offsets = self.get_offsets_from_node_ids(subgraph_2)
        subgraph_1 = [x.split('_')[0] for x in subgraph_1]
        subgraph_2 = [x.split('_')[0] for x in subgraph_2]

        subtree_indices = range(subgraph_1_offsets[0], subgraph_1_offsets[1]+1)
        # remove subtree from sent1
        original_sentence_1 = [i for j, i in enumerate(original_sentence_1) if j not in subtree_indices]
        # add new subtree of sent2 to sent1
        insert_at = subgraph_1_offsets[0]
        original_sentence_1[insert_at:insert_at] = subgraph_2

        subtree_indices = range(subgraph_2_offsets[0], subgraph_2_offsets[1]+1)
        # remove subtree from sent2
        original_sentence_2 = [i for j, i in enumerate(original_sentence_2) if j not in subtree_indices]
        # add new subtree of sent1 to sent2
        insert_at = subgraph_2_offsets[0]
        original_sentence_2[insert_at:insert_at] = subgraph_1
        return [' '.join(original_sentence_1[1:]), ' '.join(original_sentence_2[1:])]

    @staticmethod
    def get_subgraph_from_edge_type(graph, edge_type):
        # Because of prior filtering, we always will have one edge

        edges_with_type = graph.get_edges_with_property('dep', edge_type)[0]
        top_node_of_tree = edges_with_type.target_node
        node_ids = graph.get_subtree_node_ids(top_node_of_tree)
        splitted_node_ids = [x.split('_') for x in node_ids]
        splitted_node_ids = [(x[0], int(x[1])) for x in splitted_node_ids]
        return [f'{y[0]}_{y[1]}' for y in sorted(splitted_node_ids, key=itemgetter(1))]



    @staticmethod
    def get_offsets_from_node_ids(node_ids):
        """
        Returns the smallest and largest index from the list of node ids
        """
        ids = [int(x.split('_')[1]) for x in node_ids]
        return min(ids), max(ids)

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
        return check([int(x.split('_')[-1]) for x in node_ids])

    def dump_augmented_sentences_to_files(self):
        log.info(f'Saving augmented sentences at {self._output_path}')
        augmentation_types = self._augmented_sentence_pairs.keys()
        for augmentation_type in augmentation_types:
            with open(f'{self._output_path}/{augmentation_type}.tsv', 'w+') as f:
                zipped = zip(self._augmented_sentence_pairs[augmentation_type]['hun'],
                             self._augmented_sentence_pairs[augmentation_type]['eng'])
                for hun_sent, eng_sent in zipped:
                    f.write(f'{hun_sent}\t{eng_sent}')
                    f.write('\n')

    def print_augmented_pairs(self, idx):
        """
        Prints augmented sentence pairs at an idx,
        useful for debugging and viewing results
        """
        print('---------------')
        print('OBJECT SWAPPING WITH SAME PREDICATE LEMMA')
        print(self._augmented_sentence_pairs['obj_swapping_same_predicate_lemma']['hun'][idx])
        print(self._augmented_sentence_pairs['obj_swapping_same_predicate_lemma']['eng'][idx])
        print('---------------')
        print('SUBJECT SWAPPING WITH SAME PREDICATE LEMMA')
        print(self._augmented_sentence_pairs['subj_swapping_same_predicate_lemma']['hun'][idx])
        print(self._augmented_sentence_pairs['subj_swapping_same_predicate_lemma']['eng'][idx])
        print('---------------')
        print('OBJECT SWAPPING')
        print(self._augmented_sentence_pairs['obj_swapping']['hun'][idx])
        print(self._augmented_sentence_pairs['obj_swapping']['eng'][idx])
        print('---------------')
        print('SUBJECT SWAPPING')
        print(self._augmented_sentence_pairs['subj_swapping']['hun'][idx])
        print(self._augmented_sentence_pairs['subj_swapping']['eng'][idx])

    @staticmethod
    def softmax_with_temperature(x, t):
        norm_x = x / np.sqrt(np.sum(x ** 2))
        return np.exp(norm_x/t) / sum(np.exp(norm_x/t))




