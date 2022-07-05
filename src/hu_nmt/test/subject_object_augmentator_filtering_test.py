import unittest

import networkx as nx

from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator
from hu_nmt.data_augmentator.utils.types.postag_types import PostagType
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


class SubjectObjectAugmentatorFilteringTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def testPosTagFilteringTrue(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("one_1", postag='NOUN', lemma='')
        src_graph.add_node("two_2", postag='THIS_SHOULD_MATCH', lemma='')
        src_graph.add_node("three_3", postag='NOUN', lemma='')
        src_graph.add_edge('one_1', 'two_2', dep='obj')
        src_graph.add_edge('two_2', 'three_3', dep='aux')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("egy_1", postag='NOUN', lemma='')
        trg_graph.add_node("ketto_2", postag='THIS_SHOULD_MATCH', lemma='')
        trg_graph.add_node("harom_3", postag='NOUN', lemma='')
        trg_graph.add_edge('egy_1', 'ketto_2', dep='obj')
        trg_graph.add_edge('ketto_2', 'harom_3', dep='aux')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertTrue(SubjectObjectAugmentator.is_eligible_for_augmentation(src_graph_wrp, trg_graph_wrp, 'obj'))

    def testPosTagFilteringFalse(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("one_1", postag='NOUN', lemma='')
        src_graph.add_node("two_2", postag='THIS_SHOULD_MATCH', lemma='')
        src_graph.add_node("three_3", postag='NOUN', lemma='')
        src_graph.add_edge('one_1', 'two_2', dep='obj')
        src_graph.add_edge('two_2', 'three_3', dep='aux')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("egy_1", postag='NOUN', lemma='')
        trg_graph.add_node("ketto_2", postag='THIS_SHOULD_MATCH_BUT_WONT', lemma='')
        trg_graph.add_node("harom_3", postag='NOUN', lemma='')
        trg_graph.add_edge('egy_1', 'ketto_2', dep='obj')
        trg_graph.add_edge('ketto_2', 'harom_3', dep='aux')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_augmentation(src_graph_wrp, trg_graph_wrp, 'obj'))

    def test_src_graph_contains_more_obj(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("one_1", postag='NOUN', lemma='')
        src_graph.add_node("two_2", postag='THIS_SHOULD_MATCH', lemma='')
        src_graph.add_node("three_3", postag='NOUN', lemma='')
        src_graph.add_node("four_4", postag='NOUN', lemma='')
        src_graph.add_edge('one_1', 'two_2', dep='obj')
        src_graph.add_edge('one_1', 'four_4', dep='obj')
        src_graph.add_edge('two_2', 'three_3', dep='aux')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("egy_1", postag='NOUN', lemma='')
        trg_graph.add_node("ketto_2", postag='THIS_SHOULD_MATCH', lemma='')
        trg_graph.add_node("harom_3", postag='NOUN', lemma='')
        trg_graph.add_edge('egy_1', 'ketto_2', dep='obj')
        trg_graph.add_edge('ketto_2', 'harom_3', dep='aux')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_augmentation(src_graph_wrp, trg_graph_wrp, 'obj'))

    def test_trg_graph_contains_more_obj(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("one_1", postag='NOUN', lemma='')
        src_graph.add_node("two_2", postag='THIS_SHOULD_MATCH', lemma='')
        src_graph.add_node("three_3", postag='NOUN', lemma='')
        src_graph.add_edge('one_1', 'two_2', dep='obj')
        src_graph.add_edge('two_2', 'three_3', dep='aux')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("egy_1", postag='NOUN', lemma='')
        trg_graph.add_node("ketto_2", postag='THIS_SHOULD_MATCH', lemma='')
        trg_graph.add_node("harom_3", postag='NOUN', lemma='')
        src_graph.add_node("negy_4", postag='NOUN', lemma='')
        trg_graph.add_edge('egy_1', 'ketto_2', dep='obj')
        src_graph.add_edge('egy_1', 'negy_4', dep='obj')
        trg_graph.add_edge('ketto_2', 'harom_3', dep='aux')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_augmentation(src_graph_wrp, trg_graph_wrp, 'obj'))

    def test_trg_graph_not_consecutive(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("one_1", postag='NOUN', lemma='')
        src_graph.add_node("two_2", postag='THIS_SHOULD_MATCH', lemma='')
        src_graph.add_node("three_3", postag='NOUN', lemma='')
        src_graph.add_edge('one_1', 'two_2', dep='obj')
        src_graph.add_edge('two_2', 'three_3', dep='aux')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("egy_1", postag='NOUN', lemma='')
        trg_graph.add_node("ketto_2", postag='THIS_SHOULD_MATCH', lemma='')
        trg_graph.add_node("harom_3", postag='NOUN', lemma='')
        trg_graph.add_node("negy_4", postag='NOUN', lemma='')
        trg_graph.add_edge('egy_1', 'ketto_2', dep='obj')
        trg_graph.add_edge('ketto_2', 'negy_4', dep='nsubj')
        trg_graph.add_edge('egy_2', 'harom_3', dep='aux')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_augmentation(src_graph_wrp, trg_graph_wrp, 'obj'))

    def test_src_graph_not_consecutive(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("one_1", postag='NOUN', lemma='')
        src_graph.add_node("two_2", postag='THIS_SHOULD_MATCH', lemma='')
        src_graph.add_node("three_3", postag='NOUN', lemma='')
        src_graph.add_node("four_4", postag='NOUN', lemma='')
        src_graph.add_edge('one_1', 'two_2', dep='obj')
        src_graph.add_edge('two_2', 'four_4', dep='nsubj')
        src_graph.add_edge('one_1', 'three_3', dep='aux')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("egy_1", postag='NOUN', lemma='')
        trg_graph.add_node("ketto_2", postag='THIS_SHOULD_MATCH', lemma='')
        trg_graph.add_node("harom_3", postag='NOUN', lemma='')
        trg_graph.add_node("negy_4", postag='NOUN', lemma='')
        trg_graph.add_edge('egy_1', 'ketto_2', dep='obj')
        trg_graph.add_edge('ketto_2', 'harom_3', dep='aux')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_augmentation(src_graph_wrp, trg_graph_wrp, 'obj'))

    def test_both_graph_not_consecutive(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("one_1", postag='NOUN', lemma='')
        src_graph.add_node("two_2", postag='THIS_SHOULD_MATCH', lemma='')
        src_graph.add_node("three_3", postag='NOUN', lemma='')
        src_graph.add_node("four_4", postag='NOUN', lemma='')
        src_graph.add_edge('one_1', 'two_2', dep='obj')
        src_graph.add_edge('two_2', 'four_4', dep='nsubj')
        src_graph.add_edge('one_1', 'three_3', dep='aux')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("egy_1", postag='NOUN', lemma='')
        trg_graph.add_node("ketto_2", postag='THIS_SHOULD_MATCH', lemma='')
        trg_graph.add_node("harom_3", postag='NOUN', lemma='')
        trg_graph.add_node("negy_4", postag='NOUN', lemma='')
        trg_graph.add_edge('egy_1', 'ketto_2', dep='obj')
        trg_graph.add_edge('ketto_2', 'negy_4', dep='nsubj')
        trg_graph.add_edge('egy_2', 'harom_3', dep='aux')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_augmentation(src_graph_wrp, trg_graph_wrp, 'obj'))

    def test_eligible_for_both_aug(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("Deemed_1", postag='NOUN', lemma='')
        src_graph.add_node("universities_2", postag='NOUN', lemma='')
        src_graph.add_node("charge_3", postag='NOUN', lemma='')
        src_graph.add_node("huge_4", postag='NOUN', lemma='')
        src_graph.add_node("fees_5", postag='NOUN', lemma='')
        src_graph.add_edge('charge_3', 'universities_2', dep='nsubj')
        src_graph.add_edge('charge_3', 'fees_5', dep='obj')
        src_graph.add_edge('universities_2', 'Deemed_1', dep='compound')
        src_graph.add_edge('fees_5', 'huge_4', dep='amod')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("Vélt_1", postag='NOUN', lemma='')
        trg_graph.add_node("egyetemek_2", postag='NOUN', lemma='')
        trg_graph.add_node("felszámolnak_3", postag='NOUN', lemma='')
        trg_graph.add_node("hatalmas_4", postag='NOUN', lemma='')
        trg_graph.add_node("összeget_5", postag='NOUN', lemma='')
        trg_graph.add_edge('felszámolnak_3', 'egyetemek_2', dep='nsubj')
        trg_graph.add_edge('felszámolnak_3', 'összeget_5', dep='obj')
        trg_graph.add_edge('egyetemek_2', 'Vélt_1', dep='compound')
        trg_graph.add_edge('összeget_5', 'hatalmas_4', dep='amod')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertTrue(SubjectObjectAugmentator.is_eligible_for_both_augmentation(src_graph_wrp, trg_graph_wrp))

    def test_eligible_for_both_aug_no_same_nsubj_ancestor(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("Deemed_1", postag='NOUN', lemma='')
        src_graph.add_node("universities_2", postag='NOUN', lemma='')
        src_graph.add_node("charge_3", postag='NOUN', lemma='')
        src_graph.add_node("huge_4", postag='NOUN', lemma='')
        src_graph.add_node("fees_5", postag='NOUN', lemma='')
        src_graph.add_edge('charge_3', 'universities_2', dep='nsubj')
        src_graph.add_edge('charge_3', 'fees_5', dep='obj')
        src_graph.add_edge('universities_2', 'Deemed_1', dep='compound')
        src_graph.add_edge('fees_5', 'huge_4', dep='amod')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("Vélt_1", postag='NOUN', lemma='')
        trg_graph.add_node("egyetemek_2", postag='NOUN', lemma='')
        trg_graph.add_node("felszámolnak_3", postag='NOUN', lemma='')
        trg_graph.add_node("hatalmas_4", postag='NOUN', lemma='')
        trg_graph.add_node("összeget_5", postag='NOUN', lemma='')
        trg_graph.add_node("confusor_6", postag='NOUN', lemma='')

        trg_graph.add_edge('felszámolnak_3', 'confusor_6', dep='conf')
        trg_graph.add_edge('confusor_6', 'egyetemek_2', dep='nsubj')
        trg_graph.add_edge('felszámolnak_3', 'összeget_5', dep='obj')
        trg_graph.add_edge('egyetemek_2', 'Vélt_1', dep='compound')
        trg_graph.add_edge('összeget_5', 'hatalmas_4', dep='amod')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_both_augmentation(src_graph_wrp, trg_graph_wrp))

    def test_eligible_for_both_aug_no_same_obj_ancestor(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("Deemed_1", postag='NOUN', lemma='')
        src_graph.add_node("universities_2", postag='NOUN', lemma='')
        src_graph.add_node("charge_3", postag='NOUN', lemma='')
        src_graph.add_node("huge_4", postag='NOUN', lemma='')
        src_graph.add_node("fees_5", postag='NOUN', lemma='')
        src_graph.add_edge('charge_3', 'universities_2', dep='nsubj')
        src_graph.add_edge('charge_3', 'fees_5', dep='obj')
        src_graph.add_edge('universities_2', 'Deemed_1', dep='compound')
        src_graph.add_edge('fees_5', 'huge_4', dep='amod')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("Vélt_1", postag='NOUN', lemma='')
        trg_graph.add_node("egyetemek_2", postag='NOUN', lemma='')
        trg_graph.add_node("felszámolnak_3", postag='NOUN', lemma='')
        trg_graph.add_node("hatalmas_4", postag='NOUN', lemma='')
        trg_graph.add_node("összeget_5", postag='NOUN', lemma='')
        trg_graph.add_node("confusor_6", postag='NOUN', lemma='')

        trg_graph.add_edge('felszámolnak_3', 'egyetemek_2', dep='nsubj')
        trg_graph.add_edge('felszámolnak_3', 'confusor_6', dep='conf')
        trg_graph.add_edge('confusor_6', 'összeget_5', dep='obj')
        trg_graph.add_edge('egyetemek_2', 'Vélt_1', dep='compound')
        trg_graph.add_edge('összeget_5', 'hatalmas_4', dep='amod')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_both_augmentation(src_graph_wrp, trg_graph_wrp))

    def test_eligible_for_both_aug_no_NOUN_in_src(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("Deemed_1", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("universities_2", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("charge_3", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("huge_4", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("fees_5", postag=PostagType.VERB.name, lemma='')
        src_graph.add_edge('charge_3', 'universities_2', dep='nsubj')
        src_graph.add_edge('charge_3', 'fees_5', dep='obj')
        src_graph.add_edge('universities_2', 'Deemed_1', dep='compound')
        src_graph.add_edge('fees_5', 'huge_4', dep='amod')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("Vélt_1", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("egyetemek_2", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("felszámolnak_3", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("hatalmas_4", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("összeget_5", postag=PostagType.NOUN.name, lemma='')
        trg_graph.add_edge('felszámolnak_3', 'egyetemek_2', dep='nsubj')
        trg_graph.add_edge('felszámolnak_3', 'összeget_5', dep='obj')
        trg_graph.add_edge('egyetemek_2', 'Vélt_1', dep='compound')
        trg_graph.add_edge('összeget_5', 'hatalmas_4', dep='amod')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_both_augmentation(src_graph_wrp, trg_graph_wrp))

    def test_eligible_for_both_aug_no_NOUN_in_trg(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("Deemed_1", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("universities_2", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("charge_3", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("huge_4", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("fees_5", postag=PostagType.NOUN.name, lemma='')
        src_graph.add_edge('charge_3', 'universities_2', dep='nsubj')
        src_graph.add_edge('charge_3', 'fees_5', dep='obj')
        src_graph.add_edge('universities_2', 'Deemed_1', dep='compound')
        src_graph.add_edge('fees_5', 'huge_4', dep='amod')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("Vélt_1", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("egyetemek_2", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("felszámolnak_3", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("hatalmas_4", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("összeget_5", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_edge('felszámolnak_3', 'egyetemek_2', dep='nsubj')
        trg_graph.add_edge('felszámolnak_3', 'összeget_5', dep='obj')
        trg_graph.add_edge('egyetemek_2', 'Vélt_1', dep='compound')
        trg_graph.add_edge('összeget_5', 'hatalmas_4', dep='amod')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertFalse(SubjectObjectAugmentator.is_eligible_for_both_augmentation(src_graph_wrp, trg_graph_wrp))

    def test_eligible_for_both_aug_one_NOUN_in_both(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("Deemed_1", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("universities_2", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("charge_3", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("huge_4", postag=PostagType.VERB.name, lemma='')
        src_graph.add_node("fees_5", postag=PostagType.NOUN.name, lemma='')
        src_graph.add_edge('charge_3', 'universities_2', dep='nsubj')
        src_graph.add_edge('charge_3', 'fees_5', dep='obj')
        src_graph.add_edge('universities_2', 'Deemed_1', dep='compound')
        src_graph.add_edge('fees_5', 'huge_4', dep='amod')
        src_graph_wrp = DependencyGraphWrapper(src_graph)

        trg_graph = nx.DiGraph()
        trg_graph.add_node("Vélt_1", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("egyetemek_2", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("felszámolnak_3", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("hatalmas_4", postag=PostagType.VERB.name, lemma='')
        trg_graph.add_node("összeget_5", postag=PostagType.NOUN.name, lemma='')
        trg_graph.add_edge('felszámolnak_3', 'egyetemek_2', dep='nsubj')
        trg_graph.add_edge('felszámolnak_3', 'összeget_5', dep='obj')
        trg_graph.add_edge('egyetemek_2', 'Vélt_1', dep='compound')
        trg_graph.add_edge('összeget_5', 'hatalmas_4', dep='amod')
        trg_graph_wrp = DependencyGraphWrapper(trg_graph)

        self.assertTrue(SubjectObjectAugmentator.is_eligible_for_both_augmentation(src_graph_wrp, trg_graph_wrp))
