import os
import pathlib
import unittest
from typing import Iterator
from unittest.mock import patch

from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
from hu_nmt.data_augmentator.dependency_parsers.dependency_parser_factory import DependencyParserFactory


class EmptyLineCountMismatch(Exception):
    pass


class DependencyParserBaseTests(unittest.TestCase):

    resources_path = pathlib.Path(__file__).parent.parent.resolve() / 'resources'
    hu_en_path = resources_path / 'hu-en'

    def tearDown(self) -> None:
        os.environ['USE_MULTIPROCESSING'] = 'False'

    @staticmethod
    def check_line_count_in_files(files):
        for file in files:
            empty_line_count = 0
            with open(file) as f:
                for line in f:
                    if len(line.strip()) == 0:
                        empty_line_count += 1
            if (file.name == '3.tsv' and empty_line_count != 1) or (file.name != '3.tsv' and empty_line_count != 2):
                raise EmptyLineCountMismatch(
                    f'In {file.name} the expected empty line count did not match the actual {empty_line_count}!'
                )

    @patch('stanza.Pipeline')
    def test_multiprocessing_trigger_not_triggered(self, _):
        os.environ['USE_MULTIPROCESSING'] = 'False'
        parser = DependencyParserFactory.get_dependency_parser('en')
        self.assertFalse(parser.use_multiprocessing)

    @patch('stanza.Pipeline')
    def test_multiprocessing_trigger_triggered(self, _):
        os.environ['USE_MULTIPROCESSING'] = 'True'
        parser = DependencyParserFactory.get_dependency_parser('en')
        self.assertTrue(parser.use_multiprocessing)

    def test_sentences_to_serialized_dep_graph_files(self):
        # setup
        en_input_path = self.hu_en_path / 'en.tsv'
        dep_tree_output_path = self.hu_en_path / 'outputs' / 'dep_trees'
        en_output_path = dep_tree_output_path / 'en'
        batch_size = 2

        if os.path.exists(en_output_path):
            for f in os.listdir(en_output_path):
                os.remove(os.path.join(en_output_path, f))

        with open(en_input_path) as en_input_file:
            sentences = en_input_file.readlines()

        # action
        en_dep_parser = DependencyParserFactory.get_dependency_parser('en')
        en_dep_parser.sentences_to_serialized_dep_graph_files(iter(sentences), str(en_output_path), batch_size)

        # assert
        files = [file for file in dep_tree_output_path.glob('en/*') if file.is_file()]
        self.assertEqual(3, len(files))
        try:
            self.check_line_count_in_files(files)
        except EmptyLineCountMismatch as e:
            self.fail(f'There was an empty line count mismatch: {e}')

    @patch('multiprocessing.cpu_count')
    def test_sentences_to_serialized_dep_graph_files_multiprocessing(self, cpu_count_mock):
        os.environ['USE_MULTIPROCESSING'] = 'True'
        cpu_count_mock.return_value = 1
        self.test_sentences_to_serialized_dep_graph_files()

    def test_create_mini_batches(self):
        # setup
        number_of_small_batches = 3
        batch = [1, 2, 3, 4, 5]

        # action
        small_batches = DependencyParserBase.create_mini_batches(number_of_small_batches, batch)

        # assert
        self.assertEqual(number_of_small_batches, len(small_batches))
        for i, size in enumerate([2, 2, 1]):
            self.assertEqual(size, len(small_batches[i]))

    def test_create_mini_batches_too_many_mini_batches_wanted(self):
        # setup
        number_of_mini_batches = 7
        batch = [1, 2, 3, 4, 5]

        # action
        small_batches = DependencyParserBase.create_mini_batches(number_of_mini_batches, batch)

        # assert
        self.assertEqual(number_of_mini_batches, len(small_batches))
        for i, size in enumerate([1, 1, 1, 1, 1, 0, 0]):
            self.assertEqual(size, len(small_batches[i]))




