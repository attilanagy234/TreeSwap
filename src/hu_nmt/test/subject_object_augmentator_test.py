import unittest
from unittest.mock import MagicMock

from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.spacy_dependency_parser import SpacyDependencyParser
from hu_nmt.data_augmentator.utils.translation_graph import TranslationGraph
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


class SubjectObjectAugmentatorTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.english_dep_parser = EnglishDependencyParser()
        cls.hungarian_dep_parser = SpacyDependencyParser(lang='hu')

    def test_subject_object_augmentator_setup(self):
        # action
        augmentator = SubjectObjectAugmentator(list(range(5)), list(range(5)), 0.5, 123, [], '', '')

        # assert
        self.assertEqual(2, augmentator._num_augmented_sentences_to_generate_per_method)

    def test_is_eligible_for_augmentation_and_find_augmentable_candidates(self):
        # setup
        sentence_pairs = [
            ('I walked my dog down the street.', 'Sétáltattam a kutyámat az utcán.'),  # no subj in hun
            ('Have witnessed a miracle.', 'Én láttam egy csodát.'),  # no subj in eng
            ('I have witnessed a miracle.', 'Én egy csodának voltam a szemtanúja.'),  # no obj in hun
            ('I talked too much.', 'Én túl sokat beszéltem.'),  # no obj in eng
            ('I like to make pancakes.', 'Én szeretek palacsintát csinálni.'),  # not same ancestor for hun and eng
            ('I like ice cream', 'Én szeretem a fagyit.'),
            ('He loves your huge ego.', 'Ő szereti a nagy egód.'),
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator(eng_graph_wrappers, hun_graph_wrappers, 0.5, 123, [], '', 'tsv')

        # assert
        assert_values = [False, False, False, False, False, True]
        for value, hun_g, eng_g in zip(assert_values, hun_graph_wrappers, eng_graph_wrappers):
            self.assertEqual(value, augmentator.is_eligible_for_augmentation(hun_g, eng_g))

        augmentator._candidate_translations = augmentator.find_candidates(augmentator._hun_graphs, augmentator._eng_graphs)
        self.assertEqual(2, len(augmentator._candidate_translations['both']))

    def test_is_eligible_for_separate_augmentation_and_find_augmentable_candidates(self):
        # setup
        sentence_pairs = [
            ('I walked my dog down the street.', 'Sétáltattam a kutyámat az utcán.'),  # no subj in hun
            ('Have witnessed a miracle.', 'Én láttam egy csodát.'),  # no subj in eng
            ('I have witnessed a miracle.', 'Én egy csodának voltam a szemtanúja.'),  # no obj in hun
            ('I talked too much.', 'Én túl sokat beszéltem.'),  # no obj in eng
            ('I like ice cream', 'Én szeretem a fagyit.'),  # both
            ('He loves your huge ego.', 'Ő szereti a nagy egód.'), # both
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator(eng_graph_wrappers, hun_graph_wrappers, 0.5, 123, [], '', 'tsv')

        # assert
        assert_obj_values = [True, True, False, False, True, True]
        assert_nsubj_values = [False, False, True, True, True, True]
        for obj_value, nsubj_value, hun_g, eng_g in zip(assert_obj_values, assert_nsubj_values, hun_graph_wrappers, eng_graph_wrappers):
            self.assertEqual(obj_value, augmentator.is_eligible_for_separate_augmentation(hun_g, eng_g, 'obj'))
            self.assertEqual(nsubj_value, augmentator.is_eligible_for_separate_augmentation(hun_g, eng_g, 'nsubj'))

        augmentator._candidate_translations = augmentator.find_candidates(augmentator._hun_graphs, augmentator._eng_graphs, separate_augmentation=True)
        self.assertEqual(4, len(augmentator._candidate_translations['obj']))
        self.assertEqual(4, len(augmentator._candidate_translations['nsubj']))

    def test_add_augmentable_candidates_both(self):
        # setup
        sentence_pairs = [
            ('I walked my dog down the street.', 'Sétáltattam a kutyámat az utcán.'),  # no subj in hun
            ('Have witnessed a miracle.', 'Én láttam egy csodát.'),  # no subj in eng
            ('He loves your huge ego.', 'Ő szereti a nagy egód.'),
            ('I have witnessed a miracle.', 'Én egy csodának voltam a szemtanúja.'),  # no obj in hun
            ('I talked too much.', 'Én túl sokat beszéltem.'),  # no obj in eng
            ('I like to make pancakes.', 'Én szeretek palacsintát csinálni.'),  # not same ancestor for hun and eng
            ('I like ice cream', 'Én szeretem a fagyit.'),
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in
                         sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in
                         sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator(eng_graph_wrappers, hun_graph_wrappers, 0.5, 123, [], '', 'tsv')

        # action
        augmentator.add_augmentable_candidates(hun_graph_wrappers[:4], eng_graph_wrappers[:4])
        augmentator.add_augmentable_candidates(hun_graph_wrappers[4:], eng_graph_wrappers[4:])

        # assert
        self.assertEqual(2, len(augmentator._candidate_translations['both']))

    def test_group_candidates_by_predicate_lemmas(self):
        # setup
        sentence_pairs = [
            ('I like ice cream', 'Én szeretem a fagyit.'),
            ('I like bikes.', 'Én szeretem a bicikliket.'),
            ('They liked fish.', 'Ők szerették a halakat.'),
            ('You hate the red cars.', 'Ti utáljátok a piros autókat.'),
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator(eng_graph_wrappers, hun_graph_wrappers, 0.5, 123, [], '', 'tsv')
        augmentator._candidate_translations = augmentator.find_candidates(augmentator._hun_graphs, augmentator._eng_graphs)
        self.assertEqual(4, len(augmentator._candidate_translations['both']))

        # action
        lemmas_to_graphs = augmentator.group_candidates_by_predicate_lemmas()

        # assert
        grouping_values = [(('szeret', 'like'), 3), (('utál', 'hate'), 1)]
        self.assertEqual(len(grouping_values), len(lemmas_to_graphs))
        for grouping_value, (lemma_pair, graphs) in zip(grouping_values, lemmas_to_graphs.items()):
            lemmas, count = grouping_value
            self.assertEqual(lemmas, lemma_pair)
            self.assertEqual(count, len(graphs))

    def test_swap_predicates(self):
        sentence_pairs = [
            ('I like ice cream!', 'Én szeretem a fagyit!'),
            ('He loves your huge ego.', 'Ő szereti a nagy egód.'),
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator([], [], 2, 123, [], '', '')

        # action
        eng_res = augmentator.swap_predicates(eng_graph_wrappers[0], eng_graph_wrappers[1])
        hun_res = augmentator.swap_predicates(hun_graph_wrappers[0], hun_graph_wrappers[1])

        # assert
        desired_eng_res = ['i loves ice cream !', 'he like your huge ego .']  # TODO this is not the best
        self.assertEqual(desired_eng_res, eng_res)
        desired_hun_res = ['Én szereti a fagyit !', 'Ő szeretem a nagy egód .']  # TODO this is not the best
        self.assertEqual(desired_hun_res, hun_res)

    def test_swap_subtrees(self):
        # setup
        sentence_pairs = [
            ('I like ice cream!', 'Én szeretem a fagyit!'),
            ('He loves your huge ego.', 'Ő szereti a nagy egód.'),
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator([], [], 2, 123, [], '', '')

        # action
        eng_obj_res = augmentator.swap_subtrees(eng_graph_wrappers[0], eng_graph_wrappers[1], 'obj')
        hun_obj_res = augmentator.swap_subtrees(hun_graph_wrappers[0], hun_graph_wrappers[1], 'obj')
        eng_subj_res = augmentator.swap_subtrees(eng_graph_wrappers[0], eng_graph_wrappers[1], 'nsubj')
        hun_subj_res = augmentator.swap_subtrees(hun_graph_wrappers[0], hun_graph_wrappers[1], 'nsubj')

        # assert
        desired_eng_obj_res = ['i like your huge ego !', 'he loves ice cream .']
        self.assertEqual(desired_eng_obj_res, eng_obj_res)
        desired_hun_obj_res = ['Én szeretem a nagy egód !', 'Ő szereti a fagyit .']
        self.assertEqual(desired_hun_obj_res, hun_obj_res)
        desired_eng_subj_res = ['he like ice cream !', 'i loves your huge ego .']  # TODO this is not the best
        self.assertEqual(desired_eng_subj_res, eng_subj_res)
        desired_hun_subj_res = ['Ő szeretem a fagyit !', 'Én szereti a nagy egód .']  # TODO this is not the best
        self.assertEqual(desired_hun_subj_res, hun_subj_res)

    def test_augment_subtree_swapping_with_same_predicate_lemmas(self):
        # setup
        sentence_pairs = [
            ('I like ice cream', 'Én szeretem a fagyit.'),
            ('I like bikes.', 'Én szeretem a bicikliket.'),
            ('They liked fish.', 'Ők szerették a halakat.'),
            ('You hate the red cars.', 'Ti utáljátok a piros autókat.'),
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator(eng_graph_wrappers, hun_graph_wrappers, 2, 123, [], '', 'tsv')
        augmentator._candidate_translations = augmentator.find_candidates(augmentator._hun_graphs, augmentator._eng_graphs)
        self.assertEqual(4, len(augmentator._candidate_translations['both']))

        # action
        lemmas_to_graphs = augmentator.group_candidates_by_predicate_lemmas()
        augmentator.augment_subtree_swapping_with_same_predicate_lemmas(lemmas_to_graphs)

        # assert
        self.assertEqual(6, len(augmentator._augmented_sentence_pairs['obj_swapping_same_predicate_lemma']['hun']))
        self.assertEqual(6, len(augmentator._augmented_sentence_pairs['obj_swapping_same_predicate_lemma']['eng']))
        result = {'hun': ['Én szeretem a bicikliket .', 'Én szeretem a fagyit .', 'Én szeretem a halakat .', 'Ők szerették a fagyit .', 'Én szeretem a halakat .', 'Ők szerették a bicikliket .'],
                  'eng': ['i like bikes', 'i like ice cream .', 'i like fish', 'they liked ice cream .', 'i like fish .', 'they liked bikes .']}
        self.assertEqual(result, augmentator._augmented_sentence_pairs['obj_swapping_same_predicate_lemma'])

    def test_augment_subtree_swapping_with_same_predicate_lemmas_less_sample_count(self):
        # setup
        sentence_pairs = [
            ('I like ice cream', 'Én szeretem a fagyit.'),
            ('I like bikes.', 'Én szeretem a bicikliket.'),
            ('They liked fish.', 'Ők szerették a halakat.'),
            ('You hate the red cars.', 'Ti utáljátok a piros autókat.'),
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator(eng_graph_wrappers, hun_graph_wrappers, 0.5, 123, [], '', 'tsv')
        augmentator._candidate_translations = augmentator.find_candidates(augmentator._hun_graphs, augmentator._eng_graphs)
        self.assertEqual(4, len(augmentator._candidate_translations['both']))

        translation_sample = [tuple([TranslationGraph(hun_graph_wrappers[i], eng_graph_wrappers[i]) for i in range(2)])]
        augmentator.sample_list = MagicMock(return_value=translation_sample)

        # action
        lemmas_to_graphs = augmentator.group_candidates_by_predicate_lemmas()
        augmentator.augment_subtree_swapping_with_same_predicate_lemmas(lemmas_to_graphs)

        # assert
        augmentator.sample_list.assert_called_once()
        translation_combinations, sample_count = augmentator.sample_list.call_args[0]
        self.assertEqual(3, len(translation_combinations))
        self.assertEqual(1, sample_count)
        result = {'hun': ['Én szeretem a bicikliket .', 'Én szeretem a fagyit .'],
                  'eng': ['i like bikes', 'i like ice cream .']}
        self.assertEqual(result, augmentator._augmented_sentence_pairs['obj_swapping_same_predicate_lemma'])
        print(augmentator._augmented_sentence_pairs['obj_swapping_same_predicate_lemma'])

    def test_swap_object_subtrees(self):
        # setup
        sentence_pairs = [
            ('I walked my dog down the street.', 'Sétáltattam a kutyámat az utcán.'),  # no subj in hun
            ('Have witnessed a miracle.', 'Én láttam egy csodát.'),  # no subj in eng
            ('I have witnessed a miracle.', 'Én egy csodának voltam a szemtanúja.'),  # no obj in hun
            ('I talked too much.', 'Én túl sokat beszéltem.'),  # no obj in eng
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in
                         sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in
                         sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator(eng_graph_wrappers, hun_graph_wrappers, 0.5, 123, [], '',
                                               'tsv', separate_augmentation=True)
        augmentator._candidate_translations = augmentator.find_candidates(hun_graph_wrappers,
                                                                          eng_graph_wrappers,
                                                                          with_progress_bar=False,
                                                                          separate_augmentation=True)
        object_translation_pairs = SubjectObjectAugmentator.sample_item_pairs(
            augmentator._candidate_translations['obj'], 1)

        # action
        augmentator.swap_object_subtrees(object_translation_pairs, False)

        # assert
        obj_result1 = {'hun': ['Sétáltattam egy csodát az utcán .', 'Én láttam a kutyámat .'],
                  'eng': ['i walked a miracle down the street .', 'have witnessed my dog .']}
        obj_result2 = {'hun': ['Én láttam a kutyámat .', 'Sétáltattam egy csodát az utcán .'],
                   'eng': ['have witnessed my dog .', 'i walked a miracle down the street .']}
        self.assertIn(augmentator._augmented_sentence_pairs['obj_swapping'], [obj_result1, obj_result2])

    def test_swap_subject_subtrees(self):
        # setup
        sentence_pairs = [
            ('I walked my dog down the street.', 'Sétáltattam a kutyámat az utcán.'),  # no subj in hun
            ('Have witnessed a miracle.', 'Én láttam egy csodát.'),  # no subj in eng
            ('Yesterday I walked in the forest.', 'Én tegnap sétáltam az erdőben.'),  # no obj in hun
            ('The man talked too much.', 'A férfi túl sokat beszélt.'),  # no obj in eng
        ]
        eng_dep_trees = [self.english_dep_parser.sentence_to_dep_parse_tree(sent_pair[0]) for sent_pair in
                         sentence_pairs]
        hun_dep_trees = [self.hungarian_dep_parser.sentence_to_dep_parse_tree(sent_pair[1]) for sent_pair in
                         sentence_pairs]
        eng_graph_wrappers = [DependencyGraphWrapper(tree) for tree in eng_dep_trees]
        hun_graph_wrappers = [DependencyGraphWrapper(tree) for tree in hun_dep_trees]
        augmentator = SubjectObjectAugmentator(eng_graph_wrappers, hun_graph_wrappers, 0.5, 123, [], '',
                                               'tsv', separate_augmentation=True)
        augmentator._candidate_translations = augmentator.find_candidates(hun_graph_wrappers,
                                                                          eng_graph_wrappers,
                                                                          with_progress_bar=False,
                                                                          separate_augmentation=True)
        subject_translation_pairs = SubjectObjectAugmentator.sample_item_pairs(
            augmentator._candidate_translations['nsubj'], 1)

        # action
        augmentator.swap_subject_subtrees(subject_translation_pairs, False)

        # assert
        nsubj_result1 = {'hun': ['A férfi tegnap sétáltam az erdőben .', 'Én túl sokat beszélt .'],
                  'eng': ['yesterday the man walked in the forest .', 'i talked too much .']}
        nsubj_result2 = {'hun': ['Én túl sokat beszélt .', 'A férfi tegnap sétáltam az erdőben .'],
                   'eng': ['i talked too much .', 'yesterday the man walked in the forest .']}
        self.assertIn(augmentator._augmented_sentence_pairs['subj_swapping'], [nsubj_result1, nsubj_result2])



