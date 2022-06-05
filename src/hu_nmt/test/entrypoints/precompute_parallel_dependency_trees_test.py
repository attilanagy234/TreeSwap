import pathlib
import unittest

from click.testing import CliRunner

from hu_nmt.data_augmentator.entrypoints.precompute_parallel_dependency_trees import main

resources_path = pathlib.Path(__file__).parent.parent.resolve() / 'resources'
de_en_path = resources_path / 'de-en'


class PrecomputeParallelDependencyTreesTests(unittest.TestCase):
    def test_main_entrypoint(self):
        # setup
        src_lang_code = 'de'
        tgt_lang_code = 'en'
        src_input_path = de_en_path / 'de.tsv'
        tgt_input_path = de_en_path / 'en.tsv'
        dep_tree_output_path = de_en_path / 'outputs' / 'dep_trees'
        file_batch_size = 2

        # action
        runner = CliRunner()
        result = runner.invoke(main, [
            src_lang_code,
            tgt_lang_code,
            str(src_input_path),
            str(tgt_input_path),
            str(dep_tree_output_path),
            str(file_batch_size)
        ])

        # assert
        self.assertEqual(0, result.exit_code)
        files = [file for file in dep_tree_output_path.glob('**/*') if file.is_file()]
        self.assertEqual(6, len(files))
        for file in files:
            empty_line_count = 0
            with open(file) as f:
                for line in f:
                    if len(line.strip()) == 0:
                        empty_line_count += 1
            if file.name == '3.tsv':
                self.assertEqual(1, empty_line_count)
            else:
                self.assertEqual(2, empty_line_count)
