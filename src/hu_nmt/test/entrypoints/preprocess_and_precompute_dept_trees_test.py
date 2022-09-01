import pathlib
import unittest

from click.testing import CliRunner

from hu_nmt.data_augmentator.entrypoints.preprocess_and_precompute_dep_trees import main


class PrecomputeParallelDependencyTreesTests(unittest.TestCase):

    resources_path = pathlib.Path(__file__).parent.parent.resolve() / 'resources'
    preprocess_and_precompute = resources_path / 'preprocess_and_precompute'

    def test_main_entrypoint(self):
        # setup
        src_lang_code = 'hu'
        tgt_lang_code = 'en'
        src_input_path = self.preprocess_and_precompute / 'data.hu'
        tgt_input_path = self.preprocess_and_precompute / 'data.en'
        dep_tree_output_path = self.preprocess_and_precompute / 'output' / 'dep_trees'
        preprocessed_output_path = self.preprocess_and_precompute / 'output' / 'preprocessed'
        config_path = self.preprocess_and_precompute / 'config.yaml'

        # action
        runner = CliRunner()
        result = runner.invoke(main, [
            str(src_lang_code),
            str(tgt_lang_code),
            str(src_input_path),
            str(tgt_input_path),
            str(dep_tree_output_path),
            str(preprocessed_output_path),
            str(config_path)
        ])

        # assert
        self.assertEqual(0, result.exit_code)
        # check preprocessing
        files = [file for file in preprocessed_output_path.glob('**/*') if file.is_file()]
        self.assertEqual(2, len(files))
        for file in files:
            with open(file) as f:
                line_count = sum(1 for line in f if line.strip())
                self.assertEqual(7, line_count)
        # check dependency trees
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
                self.assertEqual(3, empty_line_count)
