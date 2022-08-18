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

    def test_contains_one_sentence_true(self):
        source_sentence = 'Ez egy magyar mondat.'
        target_sentence = 'This is an English sentence.'
        source_doc = self.preprocessor.source_tokenizer.tokenize(source_sentence)
        target_doc = self.preprocessor.target_tokenizer.tokenize(target_sentence)

        self.assertTrue(self.preprocessor._contains_one_sentence(source_doc, target_doc))

    def test_contains_one_sentence_false(self):
        source_sentence = 'Ez egy magyar mondat. Ez is.'
        target_sentence = 'This is an English sentence.'
        source_doc = self.preprocessor.source_tokenizer.tokenize(source_sentence)
        target_doc = self.preprocessor.target_tokenizer.tokenize(target_sentence)

        self.assertFalse(self.preprocessor._contains_one_sentence(source_doc, target_doc))

    def test_is_good_word_count_less(self):
        self.assertFalse(self.preprocessor._is_good_word_count(-1))

    def test_is_good_word_count_more(self):
        self.assertFalse(self.preprocessor._is_good_word_count(81))

    def test_is_good_word_count_true(self):
        self.assertTrue(self.preprocessor._is_good_word_count(45))

    def test_is_good_ratio_good_diff(self):
        self.assertTrue(self.preprocessor._is_good_ratio(8, 14))

    def test_is_good_ratio_good_ratio(self):
        self.assertTrue(self.preprocessor._is_good_ratio(50, 57))

    def test_is_good_ratio_false(self):
        self.assertFalse(self.preprocessor._is_good_ratio(8, 16))

    def test_is_good_length_true(self):
        source_sentence = 'Ez egy magyar mondat.'
        target_sentence = 'This is an English sentence.'

        self.assertTrue(self.preprocessor.is_good_length(source_sentence, target_sentence))

    def test_is_good_length_false(self):
        source_sentence = 'Ez egy magyar mondat. Ez egy másik.'
        target_sentence = 'This is an English sentence.'

        self.assertFalse(self.preprocessor.is_good_length(source_sentence, target_sentence))

    def test_clean_sentence_remove_double_quotes(self):
        sentence = '"\'`Ez egy idézet`\'"'

        cleaned = self.preprocessor.clean_sentence(sentence)

        expected = 'Ez egy idézet'
        self.assertEqual(expected, cleaned)

    def test_clean_sentence_remove_single_quotes(self):
        sentence = '\'Ez egy idézet`"'

        cleaned = self.preprocessor.clean_sentence(sentence)

        expected = 'Ez egy idézet'
        self.assertEqual(expected, cleaned)

    def test_clean_sentence_remove_hyphen(self):
        sentence = '-Ez egy idézet'

        cleaned = self.preprocessor.clean_sentence(sentence)

        expected = 'Ez egy idézet'
        self.assertEqual(expected, cleaned)
