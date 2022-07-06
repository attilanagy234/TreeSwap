import copy
from itertools import combinations
from operator import itemgetter
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

from hu_nmt.data_augmentator.base.augmentator_base import AugmentatorBase
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.utils.translation_graph import TranslationGraph
from hu_nmt.data_augmentator.utils.types.postag_types import PostagType
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.filters.filter import Filter

log = get_logger(__name__)
log.setLevel('DEBUG')


class SubjectObjectAugmentator(AugmentatorBase):

    def __init__(self,
                 src_graphs: Optional[List[DependencyGraphWrapper]] = None,
                 tgt_graphs: Optional[List[DependencyGraphWrapper]] = None,
                 augmented_data_ratio: float = 0.5,
                 random_seed: int = 123,
                 filters: List[Filter] = None,
                 output_path: str = './augmentations',
                 output_format: str = 'tsv',
                 save_original: bool = False,
                 separate_augmentation: bool = False,
                 filter_consecutive_subsequence: bool = True,
                 filter_same_pos_tag: bool = True,
                 filter_for_noun_tags: bool = False):
        super().__init__()
        if src_graphs and tgt_graphs and len(src_graphs) != len(tgt_graphs):
            raise ValueError('Length of sentences must be equal for both langugages')

        self.augmented_data_ratio = augmented_data_ratio
        if src_graphs is None and tgt_graphs is None:
            self.late_setup = True
            self._num_augmented_sentences_to_generate_per_method = -1
            self._pre_filter_sentence_count = 0
        else:
            self.late_setup = False
            self._src_graphs: List[DependencyGraphWrapper] = src_graphs
            self._tgt_graphs: List[DependencyGraphWrapper] = tgt_graphs
            self._pre_filter_sentence_count = len(self._src_graphs)
            self._num_augmented_sentences_to_generate_per_method = int(
                self._pre_filter_sentence_count * float(self.augmented_data_ratio))

        np.random.seed = random_seed
        self.filters = filters
        self._output_path = output_path
        self.output_format = output_format
        self.save_original = save_original
        self.separate_augmentation = separate_augmentation
        self.filter_consecutive_subsequence = filter_consecutive_subsequence
        self.filter_same_pos_tag = filter_same_pos_tag
        self.filter_for_noun_tags = filter_for_noun_tags
        self.error_cnt = 0

        self._augmentation_candidate_translations: List[TranslationGraph] = []
        self._candidate_translations: Dict[str, List[TranslationGraph]] = {'obj': [], 'nsubj': [], 'both': []}
        sentence_pairs_template = {
            'obj_swapping_same_predicate_lemma': {
                'src': [],
                'tgt': []
            },
            'subj_swapping_same_predicate_lemma': {
                'src': [],
                'tgt': []
            },
            'subj_swapping': {
                'src': [],
                'tgt': []
            },
            'obj_swapping': {
                'src': [],
                'tgt': []
            },
            # 'predicate_swapping': {
            #     'src': [],
            #     'tgt': []
            # }
        }
        self._augmented_sentence_pairs = copy.deepcopy(sentence_pairs_template)
        if self.save_original:
            self._original_augmentation_sentence_pairs = copy.deepcopy(sentence_pairs_template)

    def group_candidates_by_predicate_lemmas(self) -> Dict[Tuple[str, str], List[TranslationGraph]]:
        lemmas_to_graphs = {}  # tuple(src_lemma, tgt_lemma) --> tuple(src_graph, tgt_graph)

        for translation in self._candidate_translations["both"]:
            src_nsubj_edge = translation.src.get_edges_with_property('dep', 'nsubj')[0]
            # filtered candidates will only have one
            tgt_nsubj_edge = translation.tgt.get_edges_with_property('dep', 'nsubj')[0]

            src_predicate_lemma = translation.src.graph.nodes[src_nsubj_edge.source_node]['lemma'].strip()
            tgt_predicate_lemma = translation.tgt.graph.nodes[tgt_nsubj_edge.source_node]['lemma'].strip()
            lemmas_key = (src_predicate_lemma, tgt_predicate_lemma)
            if lemmas_key not in lemmas_to_graphs:
                lemmas_to_graphs[lemmas_key] = []
            lemmas_to_graphs[lemmas_key].append(translation)

        return lemmas_to_graphs

    def augment(self):
        if self.late_setup:
            self._num_augmented_sentences_to_generate_per_method = int(
                self._pre_filter_sentence_count * float(self.augmented_data_ratio))
        else:
            log.info('Finding augmentable sentence pairs...')
            self._candidate_translations = self.find_candidates(self._tgt_graphs,
                                                                self._src_graphs,
                                                                with_progress_bar=True,
                                                                separate_augmentation=self.separate_augmentation)
            log.info(f'Working with {len(self._candidate_translations["obj"])} object candidate sentence pairs')
            log.info(f'Working with {len(self._candidate_translations["nsubj"])} candidate sentence pairs')
            log.info(f'Working with {len(self._candidate_translations["both"])} candidate sentence pairs')

        log.info(
            f'Going to generate {self._num_augmented_sentences_to_generate_per_method} augmented sentences per method')
        # lemmas_to_graphs = self.group_candidates_by_predicate_lemmas()

        # self.augment_subtree_swapping_with_same_predicate_lemmas(lemmas_to_graphs)
        # self.augment_predicate_swapping()
        self.augment_subtree_swapping()

        # filter
        if len(self.filters) > 0:
            number_of_sents_per_aug_method = {aug_name: len(sents['src']) for aug_name, sents in
                                              self._augmented_sentence_pairs.items()}
            log.info(f'Number of sentences per method before filtering: {number_of_sents_per_aug_method}')
            log.info('Filtering sentences...')
            for aug_method_name, sentences in self._augmented_sentence_pairs.items():
                for filter in self.filters:
                    filtered_src, filtered_tgt = filter.filter(sentences['src'], sentences['tgt'])
                    self._augmented_sentence_pairs[aug_method_name]['src'] = filtered_src
                    self._augmented_sentence_pairs[aug_method_name]['tgt'] = filtered_tgt
            number_of_sents_per_aug_method = {aug_name: len(sents['src']) for aug_name, sents in
                                              self._augmented_sentence_pairs.items()}
            log.info(f'Number of sentences per method after filtering: {number_of_sents_per_aug_method}')

        self.dump_augmented_sentences_to_files()

    @staticmethod
    def sample_item_pairs(items: List, sample_count: int):
        sampled_index_pairs: set[Tuple[int, int]] = set()
        while len(sampled_index_pairs) < sample_count:
            random_index_pair = np.random.choice(len(items), 2, replace=False)
            sampled_index_pairs.add(tuple(random_index_pair))

        return [(items[x], items[y]) for x, y in sampled_index_pairs]

    def augment_predicate_swapping(self):
        log.info('Starting predicate swapping augmentation')
        # all_permutations = list(combinations(self._augmentation_candidate_sentence_pairs, 2))
        # We divide the amount we want to generate/method by two,
        # because a subtree swapping on a sentence pairs, yields
        # two new augmented sentences.
        sample_cnt = int(self._num_augmented_sentences_to_generate_per_method / 2)
        sampled_translation_pairs = self.sample_item_pairs(self._candidate_translations["both"], sample_cnt)
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
                if self.save_original:
                    # save original translation pairs
                    original_src_sents, original_tgt_sents = self.reconstruct_translation_pair(translation_pair)
                    self._original_augmentation_sentence_pairs['predicate_swapping']['src'].extend(original_src_sents)
                    self._original_augmentation_sentence_pairs['predicate_swapping']['tgt'].extend(original_tgt_sents)

                # augment translation pair
                src_sents, tgt_sents = self.augment_pair(translation_pair, 'predicate')
                self._augmented_sentence_pairs['predicate_swapping']['src'].extend(src_sents)
                self._augmented_sentence_pairs['predicate_swapping']['tgt'].extend(tgt_sents)
            except Exception as e:
                self.error_cnt += 1
                log.exception('Cannot process sentence')
        log.info(f'Could not perform {self.error_cnt} augmentations so far')

    def augment_subtree_swapping(self):
        """
        Swaps Obj and Nsubj subtrees within all the permutations of sentences
        """
        log.info('Starting subtree swapping on all permutations')
        sample_cnt = int(self._num_augmented_sentences_to_generate_per_method / 2)
        # increase sample count to
        pre_filter_multiplier = np.prod([filter.get_pre_filter_data_multiplier() for filter in self.filters])
        pre_filter_sample_cnt = int(sample_cnt * pre_filter_multiplier)

        if self.separate_augmentation:
            object_translation_pairs = self.sample_item_pairs(self._candidate_translations['obj'],
                                                              pre_filter_sample_cnt)

            subject_translation_pairs = self.sample_item_pairs(self._candidate_translations['nsubj'],
                                                               pre_filter_sample_cnt)
        else:
            object_translation_pairs = self.sample_item_pairs(self._candidate_translations["both"],
                                                              pre_filter_sample_cnt)
            # self.swap_subtrees_among_combinations(sampled_translation_pairs, same_predicate_lemma=False)
            subject_translation_pairs = object_translation_pairs
        self.swap_dep_subtrees(object_translation_pairs, 'obj', same_predicate_lemma=False)
        self.swap_dep_subtrees(subject_translation_pairs, 'nsubj', same_predicate_lemma=False)

        log.info('Finished subtree swapping on all permutations')

    def swap_dep_subtrees(self, translation_pairs: List[Tuple[TranslationGraph, TranslationGraph]], dep: str,
                          same_predicate_lemma: bool):
        if dep == 'obj':
            augmentation_key = 'obj_swapping'
        elif dep == 'nsubj':
            augmentation_key = 'subj_swapping'
        else:
            raise ValueError("Invalid dependency name value!")
        for translation_pair in tqdm(translation_pairs):
            try:
                if self.save_original:
                    original_src_sents, original_tgt_sents = self.reconstruct_translation_pair(translation_pair)

                # swapping
                src_sents, tgt_sents = self.augment_pair(translation_pair, dep)
                if same_predicate_lemma:
                    key = f'{augmentation_key}_same_predicate_lemma'
                else:
                    key = augmentation_key
                if self.save_original:
                    self._original_augmentation_sentence_pairs[key]['src'].extend(original_src_sents)
                    self._original_augmentation_sentence_pairs[key]['tgt'].extend(original_tgt_sents)
                self._augmented_sentence_pairs[key]['src'].extend(src_sents)
                self._augmented_sentence_pairs[key]['tgt'].extend(tgt_sents)
            except Exception as e:
                self.error_cnt += 1
                log.exception(f'Cannot process sentence')
        log.info(f'Could not perform {self.error_cnt} augmentations so far')

    def augment_subtree_swapping_with_same_predicate_lemmas(self, lemmas_to_graphs: Dict[
        Tuple[str, str], List[TranslationGraph]]):
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
        sample_cnt = int(
            self._num_augmented_sentences_to_generate_per_method / 2)  # = how many translation pairs do we need to sample -> a translation pair/combination gives 2 new translations
        if len(translation_combinations) > sample_cnt:
            translation_combinations = self.sample_list(translation_combinations, sample_cnt)

        self.swap_dep_subtrees(translation_combinations, 'obj', same_predicate_lemma=True)
        self.swap_dep_subtrees(translation_combinations, 'nsubj', same_predicate_lemma=True)
        log.info('Finished subtree swapping with same predicate lemmas')

    def augment_pair(self, translation_pair: Tuple[TranslationGraph, TranslationGraph], augmentation_type) -> Tuple[
        List[str], List[str]]:
        """
        Swaps the subtrees of the sentences
        Params:
            sample_pair (List of Tuples of DependencyGraphWrappers): Two pairs of src_lang and tgt_lang
                                                                     dependency parse trees selected for
                                                                     augmentation
            augmentation_type (String): defines what we swap (obj, nsubj or predicate)
        Returns:
             src_augmented_sentences (list of Strings): Source language augmented sentence pairs
                                                        created from graph_pair_one and graph_pair_two
             tgt_augmented_sentences (list of Strings): Target language augmented sentence pairs
                                                        created from graph_pair_one and graph_pair_two
        """
        src_augmented_sentences = []
        tgt_augmented_sentences = []

        src_graph_1 = translation_pair[0].src
        tgt_graph_1 = translation_pair[0].tgt
        src_graph_2 = translation_pair[1].src
        tgt_graph_2 = translation_pair[1].tgt

        # src_graph_1.display_graph()
        # tgt_graph_1.display_graph()
        # src_graph_2.display_graph()
        # tgt_graph_2.display_graph()

        if augmentation_type == 'predicate':
            src_augmented_sentences.extend(self.swap_predicates(src_graph_1, src_graph_2))
            tgt_augmented_sentences.extend(self.swap_predicates(tgt_graph_1, tgt_graph_2))
        elif augmentation_type == 'obj' or augmentation_type == 'nsubj':
            src_augmented_sentences.extend(self.swap_subtrees(src_graph_1, src_graph_2, augmentation_type))
            tgt_augmented_sentences.extend(self.swap_subtrees(tgt_graph_1, tgt_graph_2, augmentation_type))

        return src_augmented_sentences, tgt_augmented_sentences

    def swap_predicates(self, sentence_graph_1: DependencyGraphWrapper, sentence_graph_2: DependencyGraphWrapper) -> \
            List[str]:
        original_sentence_1 = self.reconstruct_sentence_from_node_ids(sentence_graph_1.graph.nodes)
        original_sentence_2 = self.reconstruct_sentence_from_node_ids(sentence_graph_2.graph.nodes)
        # Will have one edge only due to filtering. source node of nsubj edge --> predicate of sentence
        predicate_1 = sentence_graph_1.get_edges_with_property('dep', 'nsubj')[0].source_node
        predicate_2 = sentence_graph_2.get_edges_with_property('dep', 'nsubj')[0].source_node

        predicate_1_word, _, predicate_1_idx = predicate_1.rpartition('_')
        predicate_2_word, _, predicate_2_idx = predicate_2.rpartition('_')

        # Swap predicates
        original_sentence_1[int(predicate_1_idx)] = predicate_2_word
        original_sentence_2[int(predicate_2_idx)] = predicate_1_word

        return [' '.join(original_sentence_1[1:]), ' '.join(original_sentence_2[1:])]

    def build_original_sentence_with_subgraph(self, sentence_graph: DependencyGraphWrapper, subtree_type: str) -> Tuple[
        List[str], Tuple[int, int], List[str]]:
        original_sentence_words = self.reconstruct_sentence_from_node_ids(sentence_graph.graph.nodes)
        subgraph_words_with_ids = self.get_subgraph_from_edge_type(sentence_graph, subtree_type)
        subgraph_offsets = self.get_offsets_from_node_ids(subgraph_words_with_ids)
        subgraph_words = [x.rpartition('_')[0] for x in subgraph_words_with_ids]

        return original_sentence_words, subgraph_offsets, subgraph_words

    @staticmethod
    def swap_subtree(original_sentence: List[str], subgraph_offsets: Tuple[int, int], subgraph_to_insert: List[str]) -> \
            List[str]:
        subtree_indices = range(subgraph_offsets[0], subgraph_offsets[1] + 1)
        # remove subtree from sent1
        original_sentence = [i for j, i in enumerate(original_sentence) if j not in subtree_indices]
        # add new subtree of sent2 to sent1
        insert_at = subgraph_offsets[0]
        original_sentence[insert_at:insert_at] = subgraph_to_insert

        return original_sentence

    def swap_subtrees(self, sentence_graph_1: DependencyGraphWrapper, sentence_graph_2: DependencyGraphWrapper,
                      subtree_type: str) -> List[str]:
        original_sentence_1, subgraph_offsets_1, subgraph_1 = self.build_original_sentence_with_subgraph(
            sentence_graph_1, subtree_type)
        original_sentence_2, subgraph_offsets_2, subgraph_2 = self.build_original_sentence_with_subgraph(
            sentence_graph_2, subtree_type)

        original_sentence_1 = self.swap_subtree(original_sentence_1, subgraph_offsets_1, subgraph_2)
        original_sentence_2 = self.swap_subtree(original_sentence_2, subgraph_offsets_2, subgraph_1)

        return [' '.join(original_sentence_1[1:]), ' '.join(original_sentence_2[1:])]

    @staticmethod
    def get_subgraph_from_edge_type(graph, edge_type) -> List[str]:
        # Because of prior filtering, we always will have one edge

        edges_with_type = graph.get_edges_with_property('dep', edge_type)[0]
        top_node_of_tree = edges_with_type.target_node
        node_ids = graph.get_subtree_node_ids(top_node_of_tree)
        splitted_node_ids = [x.rpartition('_') for x in node_ids]
        splitted_node_ids = [(x[0], int(x[2])) for x in splitted_node_ids]
        return [f'{y[0]}_{y[1]}' for y in sorted(splitted_node_ids, key=itemgetter(1))]

    @staticmethod
    def get_offsets_from_node_ids(node_ids: List[str]) -> Tuple[int, int]:
        """
        Returns the smallest and largest index from the list of node ids
        """
        ids = [int(x.rpartition('_')[-1]) for x in node_ids]
        return min(ids), max(ids)

    def find_candidates(self, src_graphs: List[DependencyGraphWrapper], tgt_graphs: List[DependencyGraphWrapper],
                        with_progress_bar: bool = False, separate_augmentation: bool = False) \
            -> Dict[str, List[TranslationGraph]]:
        candidates = {'obj': [], 'nsubj': [], 'both': []}

        if with_progress_bar:
            iterable = tqdm(zip(src_graphs, tgt_graphs))
        else:
            iterable = zip(src_graphs, tgt_graphs)
        if separate_augmentation:
            for src_graph, tgt_graph in iterable:
                if self.is_eligible_for_augmentation(src_graph, tgt_graph, 'obj'):
                    candidates['obj'].append(TranslationGraph(src_graph, tgt_graph))
                if self.is_eligible_for_augmentation(src_graph, tgt_graph, 'nsubj'):
                    candidates['nsubj'].append(TranslationGraph(src_graph, tgt_graph))
            return candidates
        else:
            for src_graph, tgt_graph in iterable:
                if self.is_eligible_for_both_augmentation(src_graph, tgt_graph):
                    candidates['both'].append(TranslationGraph(src_graph, tgt_graph))
            return candidates

    def add_augmentable_candidates(self, src_graphs: List[DependencyGraphWrapper],
                                   tgt_graphs: List[DependencyGraphWrapper]):
        self._pre_filter_sentence_count += len(tgt_graphs)
        new_candidates = self.find_candidates(src_graphs, tgt_graphs, separate_augmentation=self.separate_augmentation)
        for k in self._candidate_translations.keys():
            self._candidate_translations[k].extend(new_candidates[k])

    def is_eligible_for_both_augmentation(self, src_graph: DependencyGraphWrapper, tgt_graph: DependencyGraphWrapper) -> bool:
        """
        Tests if a sentence (graph) pair is eligible for augmentation
        Conditions checked both for nsubj and obj edges
        """

        # Should contain one nsubj and one obj in both languages
        if not self.is_eligible_for_augmentation(src_graph, tgt_graph, 'obj') or \
                not self.is_eligible_for_augmentation(src_graph, tgt_graph, 'nsubj'):
            return False

        src_nsubj_edges = src_graph.get_edges_with_property('dep', 'nsubj')
        tgt_nsubj_edges = tgt_graph.get_edges_with_property('dep', 'nsubj')

        src_obj_edges = src_graph.get_edges_with_property('dep', 'obj')
        tgt_obj_edges = tgt_graph.get_edges_with_property('dep', 'obj')

        # take the only subject and object edge from the trees
        src_nsubj_edge = src_nsubj_edges[0]
        tgt_nsubj_edge = tgt_nsubj_edges[0]
        src_obj_edge = src_obj_edges[0]
        tgt_obj_edge = tgt_obj_edges[0]

        # nsubj and obj edges have the same ancestor (predicate)
        if src_nsubj_edge.source_node != src_obj_edge.source_node or tgt_nsubj_edge.source_node != tgt_obj_edge.source_node:
            return False

        return True

    def is_eligible_for_augmentation(self, src_graph: DependencyGraphWrapper, tgt_graph: DependencyGraphWrapper,
                                     dep: str) -> bool:
        """
        Tests if a sentence (graph) pair is eligible for augmentation
        Conditions only checked for one dependency relation
        """

        src_dep_edges = src_graph.get_edges_with_property('dep', dep)
        tgt_dep_edges = tgt_graph.get_edges_with_property('dep', dep)

        # Should contain exactly one of the given dependency in each language
        if len(src_dep_edges) != 1 or len(tgt_dep_edges) != 1:
            return False
        else:
            src_dep_edge = src_dep_edges[0]
            tgt_dep_edge = tgt_dep_edges[0]

        dep_src = src_dep_edge.target_node
        dep_tgt = tgt_dep_edge.target_node
        src_dep_subgraph = src_graph.get_subtree_node_ids(dep_src)
        tgt_dep_subgraph = tgt_graph.get_subtree_node_ids(dep_tgt)

        # Subtree is consecutive
        if self.filter_consecutive_subsequence:
            if not SubjectObjectAugmentator.is_consecutive_subsequence(src_dep_subgraph):
                return False
            if not SubjectObjectAugmentator.is_consecutive_subsequence(tgt_dep_subgraph):
                return False

        if self.filter_for_noun_tags:
            src_dep_subtree = DependencyGraphWrapper(src_graph.get_subtree(dep_src))
            tgt_dep_subtree = DependencyGraphWrapper(tgt_graph.get_subtree(dep_tgt))
            # Should contain at least one NOUN property both in tgt and src
            if not (src_dep_subtree.get_nodes_with_property('postag', PostagType.NOUN.name)
                    + src_dep_subtree.get_nodes_with_property('postag', PostagType.PROPN.name)):
                return False
            if not (tgt_dep_subtree.get_nodes_with_property('postag', PostagType.NOUN.name)
                    + tgt_dep_subtree.get_nodes_with_property('postag', PostagType.PROPN.name)):
                return False

        # Roots of the subgraphs to be swapped have the same POS tag
        if self.filter_same_pos_tag:
            src_dep_subtree_root_postag = src_graph.get_node_property(src_dep_edge.target_node, 'postag')
            tgt_dep_subtree_root_postag = tgt_graph.get_node_property(tgt_dep_edge.target_node, 'postag')
            if src_dep_subtree_root_postag != tgt_dep_subtree_root_postag:
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
        return check([int(x.rpartition('_')[-1]) for x in node_ids])

    def dump_augmented_sentences_to_files(self):
        log.info(f'Saving augmented sentences at {self._output_path} with output format {self.output_format}')
        if not os.path.exists(self._output_path):
            os.mkdir(self._output_path)
        augmentation_types = self._augmented_sentence_pairs.keys()
        if self.output_format == 'tsv':
            for augmentation_type in augmentation_types:
                with open(f'{self._output_path}/{augmentation_type}.tsv', 'w+') as f:
                    zipped = zip(self._augmented_sentence_pairs[augmentation_type]['src'],
                                 self._augmented_sentence_pairs[augmentation_type]['tgt'])
                    for src_sent, tgt_sent in zipped:
                        f.write(f'{src_sent}\t{tgt_sent}')
                        f.write('\n')
        else:
            for augmentation_type in augmentation_types:
                os.mkdir(f'{self._output_path}/{augmentation_type}')
                if self.save_original:
                    os.mkdir(f'{self._output_path}/original_{augmentation_type}')
                for lang in ['src', 'tgt']:
                    if self.save_original:
                        with open(f'{self._output_path}/original_{augmentation_type}/{augmentation_type}.{lang}',
                                  'w+') as original_file:
                            for sent in self._original_augmentation_sentence_pairs[augmentation_type][lang]:
                                original_file.write(sent)
                                original_file.write('\n')
                    with open(f'{self._output_path}/{augmentation_type}/{augmentation_type}.{lang}', 'w+') as f:
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
        print(self._augmented_sentence_pairs['obj_swapping_same_predicate_lemma']['src'][idx])
        print(self._augmented_sentence_pairs['obj_swapping_same_predicate_lemma']['tgt'][idx])
        print('---------------')
        print('SUBJECT SWAPPING WITH SAME PREDICATE LEMMA')
        print(self._augmented_sentence_pairs['subj_swapping_same_predicate_lemma']['src'][idx])
        print(self._augmented_sentence_pairs['subj_swapping_same_predicate_lemma']['tgt'][idx])
        print('---------------')
        print('OBJECT SWAPPING')
        print(self._augmented_sentence_pairs['obj_swapping']['src'][idx])
        print(self._augmented_sentence_pairs['obj_swapping']['tgt'][idx])
        print('---------------')
        print('SUBJECT SWAPPING')
        print(self._augmented_sentence_pairs['subj_swapping']['src'][idx])
        print(self._augmented_sentence_pairs['subj_swapping']['tgt'][idx])

    @staticmethod
    def softmax_with_temperature(x, t):
        norm_x = x / np.sqrt(np.sum(x ** 2))
        return np.exp(norm_x / t) / sum(np.exp(norm_x / t))

    def reconstruct_translation_pair(self, translation_pair: Tuple[TranslationGraph, TranslationGraph]) -> Tuple[
        List[str], List[str]]:
        src_sents = [' '.join(self.reconstruct_sentence_from_node_ids(translation_pair[i].src.graph.nodes)[1:]) for i in
                     range(2)]
        tgt_sents = [' '.join(self.reconstruct_sentence_from_node_ids(translation_pair[i].tgt.graph.nodes)[1:]) for i in
                     range(2)]

        return src_sents, tgt_sents
