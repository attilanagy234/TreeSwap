import pathlib
import unittest

from hu_nmt.data_augmentator.preprocessor.preprocessor import Preprocessor


class PreprocessorTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        test_resources_base_path = pathlib.Path(__file__).parent.parent.resolve() / 'resources'
        source_data_path = test_resources_base_path / 'hu-en' / 'hu.tsv'
        target_data_path = test_resources_base_path / 'hu-en' / 'en.tsv'
        config_path = test_resources_base_path / 'configs' / 'preprocessor_test_config.yaml'
        cls.preprocessor = Preprocessor(str(source_data_path), str(target_data_path), str(config_path), '', '')

    def test_is_correct_language_true(self):
        source_sentence = 'Ez egy magyar mondat.'
        target_sentence = 'This is an English sentence.'

        self.assertTrue(self.preprocessor.is_correct_language(source_sentence, target_sentence))

    def test_is_correct_language_false(self):
        source_sentence = 'Это русское предложение.'
        target_sentence = 'This is an English sentence.'

        self.assertFalse(self.preprocessor.is_correct_language(source_sentence, target_sentence))