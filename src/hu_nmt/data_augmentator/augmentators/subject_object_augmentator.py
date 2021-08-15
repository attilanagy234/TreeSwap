from itertools import combinations
from operator import itemgetter
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

from hu_nmt.data_augmentator.base.augmentator_base import AugmentatorBase
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.utils.translation_graph import TranslationGraph
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper

log = get_logger(__name__)
log.setLevel('DEBUG')


class SubjectObjectAugmentator(AugmentatorBase):

    def __init__(self, eng_graphs: List[DependencyGraphWrapper], hun_graphs: List[DependencyGraphWrapper], augmented_data_ratio: float, random_seed: int, output_path: str, output_format: str):
        super().__init__()
        if len(eng_graphs) != len(hun_graphs):
            raise ValueError('Length of sentences must be equal for both langugages')
        self._num_augmented_sentences_to_generate_per_method = int(len(eng_graphs) * float(augmented_data_ratio))
        log.info(f'number of desired sentences/method: {self._num_augmented_sentences_to_generate_per_method}')
        np.random.seed = random_seed
        self._output_path = output_path
        self.output_format = output_format
        self.error_cnt = 0
        self._eng_graphs: List[DependencyGraphWrapper] = eng_graphs
        self._hun_graphs: List[DependencyGraphWrapper] = hun_graphs
        self._augmentation_candidate_translations: List[TranslationGraph] = []
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

    def group_candidates_by_predicate_lemmas(self) -> Dict[Tuple[str, str], List[TranslationGraph]]:
        lemmas_to_graphs = {}  # tuple(hun_lemma, eng_lemma) --> tuple(hun_graph, eng graph)

        for translation in self._augmentation_candidate_translations:
            hun_nsubj_edge = translation.hun.get_edges_with_property('dep', 'nsubj')[0]  # filtered candidates will only have one
            eng_nsubj_edge = translation.eng.get_edges_with_property('dep', 'nsubj')[0]

            hun_predicate_lemma = translation.hun.graph.nodes[hun_nsubj_edge.source_node]['lemma'].strip()
            eng_predicate_lemma = translation.eng.graph.nodes[eng_nsubj_edge.source_node]['lemma'].strip()
            lemmas_key = (hun_predicate_lemma, eng_predicate_lemma)
            if lemmas_key not in lemmas_to_graphs:
                lemmas_to_graphs[lemmas_key] = []
            lemmas_to_graphs[lemmas_key].append(translation)

        return lemmas_to_graphs

    def augment(self):
        log.info('Finding augmentable sentence pairs...')
        self.find_augmentable_candidates()
        log.info(f'Found {len(self._augmentation_candidate_translations)} candidate sentence pairs')
        lemmas_to_graphs = self.group_candidates_by_predicate_lemmas()

        self.augment_subtree_swapping_with_same_predicate_lemmas(lemmas_to_graphs)
        self.augment_predicate_swapping()
        self.augment_subtree_swapping()

        # for i in range(10):
        #     self.print_augmented_pairs(i)

        self.dump_augmented_sentences_to_files()

    @staticmethod
    def sample_item_pairs(items: List, sample_count: int):
        sampled_index_pairs: set[Tuple[int, int]] = set()
        while len(sampled_index_pairs) < sample_count:
            random_index_pair = np.random.choice(len(items), 2, replace=False)
            sampled_index_pairs.add(random_index_pair)

        return [(items[x], items[y]) for x, y in sampled_index_pairs]

    def augment_predicate_swapping(self):
        log.info('Starting predicate swapping augmentation')
        # all_permutations = list(combinations(self._augmentation_candidate_sentence_pairs, 2))
        # We divide the amount we want to generate/method by two,
        # because a subtree swapping on a sentence pairs, yields
        # two new augmented sentences.
        sample_cnt = int(self._num_augmented_sentences_to_generate_per_method / 2)
        sampled_translation_pairs = self.sample_item_pairs(self._augmentation_candidate_translations, sample_cnt)
        self.swap_predicates_in_all_combinations(sampled_translation_pairs)
        log.info('Finished predicate swapping augmentation')

    @staticmethod
    def sample_list(from_list, num_samples):
        all_indices = [x for x in range(len(from_list))]
        sampled_indices = np.random.choice(all_indices, num_samples, replace=False)
        return [from_list[idx] for idx in sampled_indices]

    def swap_predicates_in_all_combinations(self, translation_combinations):
        for translation_pair in tqdm(translation_combinations):
            try:
                hun_sents, eng_sents = self.augment_pair(translation_pair, 'predicate')
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
        sample_cnt = int(self._num_augmented_sentences_to_generate_per_method / 2)
        sampled_translation_pairs = self.sample_item_pairs(self._augmentation_candidate_translations, sample_cnt)
        self.swap_subtrees_among_combinations(sampled_translation_pairs, same_predicate_lemma=False)
        log.info('Finished subtree swapping on all permutations')

    def swap_subtrees_among_combinations(self, translation_pairs: List[Tuple[TranslationGraph, TranslationGraph]], same_predicate_lemma: bool):
        for translation_pair in tqdm(translation_pairs):
            try:
                hun_sents, eng_sents = self.augment_pair(translation_pair, 'obj')
                if same_predicate_lemma:
                    key = 'obj_swapping_same_predicate_lemma'
                else:
                    key = 'obj_swapping'
                self._augmented_sentence_pairs[key]['hun'].extend(hun_sents)
                self._augmented_sentence_pairs[key]['eng'].extend(eng_sents)
                hun_sents, eng_sents = self.augment_pair(translation_pair, 'nsubj')
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

    def augment_subtree_swapping_with_same_predicate_lemmas(self, lemmas_to_graphs: Dict[Tuple[str, str], List[TranslationGraph]]):
        """
        Swaps Obj and Nsubj subtrees within the combinations of sentences
        that have the same same predicate lemma
        """
        predicate_lemmas_sorted_by_freq = sorted(lemmas_to_graphs, key=lambda x: len(lemmas_to_graphs[x]), reverse=True)
        translation_combinations = []
        log.info('Starting subtree swapping with same predicate lemma...')
        log.info('Precomputing same predicate lemma combinations...')
        for lemma_pair in tqdm(predicate_lemmas_sorted_by_freq):
            translation_graphs_with_same_lemma = lemmas_to_graphs[lemma_pair]
            if len(translation_graphs_with_same_lemma) == 1:  # cannot get combinations if only one translation has that lemma
                break

            # get all combinations from translation_graphs_with_same_lemma
            combinations_per_lemma = list(combinations(translation_graphs_with_same_lemma, 2))
            translation_combinations.extend(combinations_per_lemma)
        log.info(f'Got {len(translation_combinations)} translation combinations with the same lemma')
        sample_cnt = int(self._num_augmented_sentences_to_generate_per_method / 2)  # = how many translation pairs do we need to sample -> a translation pair/combination gives 2 new translations
        if len(translation_combinations) > sample_cnt:
            translation_combinations = self.sample_list(translation_combinations, sample_cnt)

        self.swap_subtrees_among_combinations(translation_combinations, same_predicate_lemma=True)
        log.info('Finished subtree swapping with same predicate lemmas')

    def augment_pair(self, translation_pair: Tuple[TranslationGraph, TranslationGraph], augmentation_type) -> Tuple[List[str], List[str]]:
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

        hun_graph_1 = translation_pair[0].hun
        eng_graph_1 = translation_pair[0].eng
        hun_graph_2 = translation_pair[1].hun
        eng_graph_2 = translation_pair[1].eng

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

    def swap_predicates(self, sentence_graph_1: DependencyGraphWrapper, sentence_graph_2: DependencyGraphWrapper) -> List[str]:
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

    def build_original_sentence_with_subgraph(self, sentence_graph: DependencyGraphWrapper, subtree_type: str) -> Tuple[List[str], Tuple[int, int], List[str]]:
        original_sentence_words = self.reconstruct_sentence_from_node_ids(sentence_graph.graph.nodes)
        subgraph_words_with_ids = self.get_subgraph_from_edge_type(sentence_graph, subtree_type)
        subgraph_offsets = self.get_offsets_from_node_ids(subgraph_words_with_ids)
        subgraph_words = [x.split('_')[0] for x in subgraph_words_with_ids]

        return original_sentence_words, subgraph_offsets, subgraph_words

    @staticmethod
    def swap_subtree(original_sentence: List[str], subgraph_offsets: Tuple[int, int], subgraph_to_insert: List[str]) -> List[str]:
        subtree_indices = range(subgraph_offsets[0], subgraph_offsets[1] + 1)
        # remove subtree from sent1
        original_sentence = [i for j, i in enumerate(original_sentence) if j not in subtree_indices]
        # add new subtree of sent2 to sent1
        insert_at = subgraph_offsets[0]
        original_sentence[insert_at:insert_at] = subgraph_to_insert

        return original_sentence

    def swap_subtrees(self, sentence_graph_1: DependencyGraphWrapper, sentence_graph_2: DependencyGraphWrapper, subtree_type: str) -> List[str]:
        original_sentence_1, subgraph_offsets_1, subgraph_1 = self.build_original_sentence_with_subgraph(sentence_graph_1, subtree_type)
        original_sentence_2, subgraph_offsets_2, subgraph_2 = self.build_original_sentence_with_subgraph(sentence_graph_2, subtree_type)

        original_sentence_1 = self.swap_subtree(original_sentence_1, subgraph_offsets_1, subgraph_2)
        original_sentence_2 = self.swap_subtree(original_sentence_2, subgraph_offsets_2, subgraph_1)

        return [' '.join(original_sentence_1[1:]), ' '.join(original_sentence_2[1:])]

    @staticmethod
    def get_subgraph_from_edge_type(graph, edge_type) -> List[str]:
        # Because of prior filtering, we always will have one edge

        edges_with_type = graph.get_edges_with_property('dep', edge_type)[0]
        top_node_of_tree = edges_with_type.target_node
        node_ids = graph.get_subtree_node_ids(top_node_of_tree)
        splitted_node_ids = [x.split('_') for x in node_ids]
        splitted_node_ids = [(x[0], int(x[1])) for x in splitted_node_ids]
        return [f'{y[0]}_{y[1]}' for y in sorted(splitted_node_ids, key=itemgetter(1))]

    @staticmethod
    def get_offsets_from_node_ids(node_ids: List[str]) -> Tuple[int, int]:
        """
        Returns the smallest and largest index from the list of node ids
        """
        ids = [int(x.split('_')[1]) for x in node_ids]
        return min(ids), max(ids)

    def find_augmentable_candidates(self):
        for hun_graph, eng_graph in tqdm(zip(self._hun_graphs, self._eng_graphs)):
            if self.is_eligible_for_augmentation(hun_graph, eng_graph):
                self._augmentation_candidate_translations.append(TranslationGraph(hun_graph, eng_graph))

    def is_eligible_for_augmentation(self, hun_graph: DependencyGraphWrapper, eng_graph: DependencyGraphWrapper) -> bool:
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
    def is_consecutive_subsequence(node_ids: List[str]) -> bool:
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
        if self.output_format == 'tsv':
            for augmentation_type in augmentation_types:
                with open(f'{self._output_path}/{augmentation_type}.tsv', 'w+') as f:
                    zipped = zip(self._augmented_sentence_pairs[augmentation_type]['hun'],
                                 self._augmented_sentence_pairs[augmentation_type]['eng'])
                    for hun_sent, eng_sent in zipped:
                        f.write(f'{hun_sent}\t{eng_sent}')
                        f.write('\n')
        else:
            for augmentation_type in augmentation_types:
                for lang in ['hun', 'eng']:
                    with open(f'{self._output_path}/augmentation_type.{lang[:2]}') as f:
                        for sent in self._augmented_sentence_pairs[augmentation_type][lang]:
                            f.write(sent)
                            f.write('\n')

    def print_augmented_pairs(self, idx: int):
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




