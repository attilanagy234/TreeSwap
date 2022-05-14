import unittest

import networkx as nx

from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


class SubjectObjectAugmentatorFilteringTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def testPosTagFilteringTrue(self):
        src_graph = nx.DiGraph()
        src_graph.add_node("one_1", postag='NOUN', lemma='')
        src_graph.add_node("two_2", postag='THIS_SHOULD_MATCH', lemma='')
        src_graph.add_node("three_2", postag='NOUN', lemma='')
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
        src_graph.add_node("three_2", postag='NOUN', lemma='')
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
