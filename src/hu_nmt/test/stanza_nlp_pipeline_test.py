import unittest

from hu_nmt.data_augmentator.dependency_parsers.nlp_pipeline_factory import NlpPipelineFactory


class StanzaDependencyParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eng_parser = NlpPipelineFactory.get_dependency_parser('en')
        cls.eng_tokenizer = NlpPipelineFactory.get_tokenizer('en')

    def test_token_count(self):
        sentence = 'This is a sentence: it\'s still the same.'
        tree = self.eng_parser.sentence_to_dep_parse_tree(sentence)

        count = self.eng_parser.count_tokens_from_graph(tree)

        self.assertEqual(count, 9)

    def test_sentence_count(self):
        sentence = 'This is a sentence: it\'s still the same. This is another! Is this a list: 1, 2, 3?'
        doc = self.eng_tokenizer.tokenize(sentence)

        count = self.eng_tokenizer.count_sentences(doc)

        self.assertEqual(count, 3)

    def test_token_count(self):
        sentence = 'This is a sentence: it\'s still the same. This is another! Is this a list: 1, 2, 3?'
        doc = self.eng_tokenizer.tokenize(sentence)

        count = self.eng_tokenizer.count_tokens_from_doc(doc)

        self.assertEqual(count, 19)


