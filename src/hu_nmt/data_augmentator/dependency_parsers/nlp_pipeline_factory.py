from hu_nmt.data_augmentator.base.nlp_pipeline_base import NlpPipelineBase
from hu_nmt.data_augmentator.dependency_parsers.stanza_nlp_pipeline import StanzaNlpPipeline
from hu_nmt.data_augmentator.dependency_parsers.spacy_nlp_pipeline import SpacyNlpPipeline


class NlpPipelineFactory:
    dep_parsers = {
        'en': lambda: StanzaNlpPipeline(lang='en', processors='tokenize, pos, lemma, depparse'),
        'ja': lambda: StanzaNlpPipeline(lang='ja', processors='tokenize, pos, lemma, depparse'),
        'hr': lambda: StanzaNlpPipeline(lang='hr', processors='tokenize, pos, lemma, depparse'),
        'ru': lambda: StanzaNlpPipeline(lang='ru', processors='tokenize, pos, lemma, depparse'),
        'cs': lambda: StanzaNlpPipeline(lang='cs', processors='tokenize, mwt, pos, lemma, depparse'),
        'de': lambda: StanzaNlpPipeline(lang='de', processors='tokenize, mwt, pos, lemma, depparse'),
        'uk': lambda: StanzaNlpPipeline(lang='uk', processors='tokenize, mwt, pos, lemma, depparse'),
        'fr': lambda: StanzaNlpPipeline(lang='fr', processors='tokenize, mwt, pos, lemma, depparse'),
        'hu': lambda: SpacyNlpPipeline(lang='hu'),
    }

    tokenizers = {
        'en': lambda: StanzaNlpPipeline(lang='en', processors='tokenize, pos'),
        'ja': lambda: StanzaNlpPipeline(lang='ja', processors='tokenize, pos'),
        'hr': lambda: StanzaNlpPipeline(lang='hr', processors='tokenize, pos'),
        'ru': lambda: StanzaNlpPipeline(lang='ru', processors='tokenize, pos'),
        'cs': lambda: StanzaNlpPipeline(lang='cs', processors='tokenize, mwt, pos'),
        'de': lambda: StanzaNlpPipeline(lang='de', processors='tokenize, mwt, pos'),
        'uk': lambda: StanzaNlpPipeline(lang='uk', processors='tokenize, mwt, pos'),
        'fr': lambda: StanzaNlpPipeline(lang='fr', processors='tokenize, mwt, pos'),
        'hu': lambda: SpacyNlpPipeline(lang='hu'),
    }
    
    @classmethod
    def get_dependency_parser(cls, lang_code) -> NlpPipelineBase:
        return cls.dep_parsers[lang_code]()

    @classmethod
    def get_tokenizer(cls, lang_code) -> NlpPipelineBase:
        return cls.tokenizers[lang_code]()

