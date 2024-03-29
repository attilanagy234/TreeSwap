import os
import pathlib
import unittest
import filecmp
import networkx as nx
import pytest

from hu_nmt.data_augmentator.dependency_parsers.spacy_nlp_pipeline import SpacyNlpPipeline
from hu_nmt.data_augmentator.utils.data_helpers import get_files_in_folder

dirname = os.path.dirname(__file__)


class SpacyDependencyParserTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.hun_dep_parser = SpacyNlpPipeline(lang='hu')

    def test_sentence_to_dep_parse_tree_hu(self):
        hun_sentence = 'Péter elvitte a kutyát sétálni az erdőbe.'
        dep_graph = self.hun_dep_parser.sentence_to_dep_parse_tree(hun_sentence)
        if not isinstance(dep_graph, nx.DiGraph):
            raise TypeError('Dependency graph needs to be a nx DiGraph')

    @pytest.mark.skipif(os.getenv('CICD') == 'true', reason='Do not run test in CI/CD build.')
    def test_sentence_to_dep_parse_tree_de(self):
        de_dep_parser = SpacyNlpPipeline(lang='de')
        de_sentence = 'Ich liebe lange Spaziergänge in den Bergen.'
        dep_graph = de_dep_parser.sentence_to_dep_parse_tree(de_sentence)
        if not isinstance(dep_graph, nx.DiGraph):
            raise TypeError('Dependency graph needs to be a nx DiGraph')

    @pytest.mark.skipif(os.getenv('CICD') == 'true', reason='Do not run test in CI/CD build.')
    def test_sentence_to_dep_parse_tree_fr(self):
        de_dep_parser = SpacyNlpPipeline(lang='fr')
        de_sentence = "J'ai 20 ans."
        dep_graph = de_dep_parser.sentence_to_dep_parse_tree(de_sentence)
        if not isinstance(dep_graph, nx.DiGraph):
            raise TypeError('Dependency graph needs to be a nx DiGraph')

    def test_sentences_to_serialized_dep_graph_files(self):
        sentences = [
            'Péter elvitte a kutyát sétálni az erdőbe.',
            'Ma elmegyek a boltba.',
            'Róbert kihívta Ákost egy futóversenyre.'
        ]
        test_base_dir = pathlib.Path(__file__).parent.resolve() / 'resources' / 'spacy_dep_parser_test'
        output_dir = test_base_dir / 'output'
        expected_dir = test_base_dir / 'expected'
        file_batch_size = 1
        self.hun_dep_parser.sentences_to_serialized_dep_graph_files(iter(sentences), str(output_dir), file_batch_size)
        output_file_cnts = len(get_files_in_folder(str(output_dir)))
        expected_file_cnts = len(get_files_in_folder(str(expected_dir)))
        self.assertEqual(expected_file_cnts, output_file_cnts)
        self.assertTrue(filecmp.cmp(str(expected_dir / '1.tsv'), str(output_dir / '1.tsv')))
        self.assertTrue(filecmp.cmp(str(expected_dir / '2.tsv'), str(output_dir / '2.tsv')))
        self.assertTrue(filecmp.cmp(str(expected_dir / '3.tsv'), str(output_dir / '3.tsv')))

    def test_sentence_count(self):
        sentence = 'Ez egy mondat: ez még ugyanaz. Ez egy másik!'
        doc = self.hun_dep_parser.tokenize(sentence)

        count = self.hun_dep_parser.count_sentences(doc)

        self.assertEqual(count, 2)

    def test_token_count_from_graph(self):
        sentence = 'Ez egy mondat: ez még ugyanaz.'
        tree = self.hun_dep_parser.sentence_to_dep_parse_tree(sentence)

        count = self.hun_dep_parser.count_tokens_from_graph(tree)

        self.assertEqual(count, 6)

    def test_token_count_from_doc(self):
        sentence = 'Ez egy mondat: ez még ugyanaz.'
        doc = self.hun_dep_parser.tokenize(sentence)

        count = self.hun_dep_parser.count_tokens_from_doc(doc)

        self.assertEqual(count, 6)

