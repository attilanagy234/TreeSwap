import os
import unittest
import filecmp
import networkx as nx
from hu_nmt.data_augmentator.dependency_parsers.spacy_dependency_parser import SpacyDependencyParser
from hu_nmt.data_augmentator.utils.data_helpers import get_files_in_folder

dirname = os.path.dirname(__file__)


class SpacyDependencyParserTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.hun_dep_parser = SpacyDependencyParser(lang='hu')

    def test_sentence_to_dep_parse_tree_hu(self):
        hun_sentence = 'Péter elvitte a kutyát sétálni az erdőbe.'
        dep_graph = self.hun_dep_parser.sentence_to_dep_parse_tree(hun_sentence)
        if not isinstance(dep_graph, nx.DiGraph):
            raise TypeError('Dependency graph needs to be a nx DiGraph')

    def test_sentence_to_dep_parse_tree_de(self):
        de_dep_parser = SpacyDependencyParser(lang='de')
        de_sentence = 'Ich liebe lange Spaziergänge in den Bergen.'
        dep_graph = de_dep_parser.sentence_to_dep_parse_tree(de_sentence)
        if not isinstance(dep_graph, nx.DiGraph):
            raise TypeError('Dependency graph needs to be a nx DiGraph')

    def test_sentences_to_serialized_dep_graph_files(self):
        sentences = [
            'Péter elvitte a kutyát sétálni az erdőbe.',
            'Ma elmegyek a boltba.',
            'Róbert kihívta Ákost egy futóversenyre.'
        ]
        test_base_dir = './resources/spacy_dep_parser_test'
        output_dir = os.path.join(test_base_dir, 'output')
        expected_dir = os.path.join(test_base_dir, 'expected')
        file_batch_size = 1
        self.hun_dep_parser.sentences_to_serialized_dep_graph_files(sentences, output_dir, file_batch_size)
        output_file_cnts = len(get_files_in_folder(output_dir))
        expected_file_cnts = len(get_files_in_folder(expected_dir))
        self.assertEqual(expected_file_cnts, output_file_cnts)
        self.assertTrue(filecmp.cmp(os.path.join(expected_dir, '1.tsv'), os.path.join(output_dir, '1.tsv')))
        self.assertTrue(filecmp.cmp(os.path.join(expected_dir, '2.tsv'), os.path.join(output_dir, '2.tsv')))
        self.assertTrue(filecmp.cmp(os.path.join(expected_dir, '3.tsv'), os.path.join(output_dir, '3.tsv')))
