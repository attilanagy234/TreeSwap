from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
from hu_nmt.data_augmentator.dependency_parsers.stanza_dependency_parser import StanzaDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.spacy_dependency_parser import SpacyDependencyParser


class DependencyParserFactory:
    dep_parsers = {
        'en': lambda: StanzaDependencyParser(lang='en', processors='tokenize, pos, lemma, depparse'),
        'ja': lambda: StanzaDependencyParser(lang='ja', processors='tokenize, pos, lemma, depparse'),
        'hr': lambda: StanzaDependencyParser(lang='hr', processors='tokenize, pos, lemma, depparse'),
        'ru': lambda: StanzaDependencyParser(lang='ru', processors='tokenize, pos, lemma, depparse'),
        'cs': lambda: StanzaDependencyParser(lang='cs', processors='tokenize, mwt, pos, lemma, depparse'),
        'de': lambda: StanzaDependencyParser(lang='de', processors='tokenize, mwt, pos, lemma, depparse'),
        'uk': lambda: StanzaDependencyParser(lang='uk', processors='tokenize, mwt, pos, lemma, depparse'),
        'fr': lambda: StanzaDependencyParser(lang='fr', processors='tokenize, mwt, pos, lemma, depparse'),
        'hu': lambda: SpacyDependencyParser(lang='hu'),
    }

    tokenizers = {
        'en': lambda: StanzaDependencyParser(lang='en', processors='tokenize, pos'),
        'ja': lambda: StanzaDependencyParser(lang='ja', processors='tokenize, pos'),
        'hr': lambda: StanzaDependencyParser(lang='hr', processors='tokenize, pos'),
        'ru': lambda: StanzaDependencyParser(lang='ru', processors='tokenize, pos'),
        'cs': lambda: StanzaDependencyParser(lang='cs', processors='tokenize, mwt, pos'),
        'de': lambda: StanzaDependencyParser(lang='de', processors='tokenize, mwt, pos'),
        'uk': lambda: StanzaDependencyParser(lang='uk', processors='tokenize, mwt, pos'),
        'fr': lambda: StanzaDependencyParser(lang='fr', processors='tokenize, mwt, pos'),
        'hu': lambda: SpacyDependencyParser(lang='hu'),
    }
    
    @classmethod
    def get_dependency_parser(cls, lang_code) -> DependencyParserBase:
        return cls.dep_parsers[lang_code]()

    @classmethod
    def get_tokenizer(cls, lang_code) -> DependencyParserBase:
        return cls.tokenizers[lang_code]()

