import unittest

from hu_nmt.data_augmentator.dependency_parsers.nlp_pipeline_factory import NlpPipelineFactory


class StanzaDependencyParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eng_tokenizer = NlpPipelineFactory.get_tokenizer('en')

    def test_sentence_count(self):
        sentence = 'This is a sentence: it\'s still the same. This is another! Is this a list: 1, 2, 3?'
        doc = self.eng_tokenizer.tokenize(sentence)

        count = self.eng_tokenizer.count_sentences(doc)

        self.assertEqual(count, 3)

    def test_word_count(self):
        sentence = 'This is a sentence: it\'s still the same. This is another! Is this a list: 1, 2, 3?'
        doc = self.eng_tokenizer.tokenize(sentence)

        count = self.eng_tokenizer.count_words(doc)

        self.assertEqual(count, 19)


