from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
from hu_nmt.data_augmentator.dependency_parsers.stanza_dependency_parser import StanzaDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.spacy_dependency_parser import SpacyDependencyParser


class DependencyParserFactory:
    dep_parsers = {
        'en': lambda: StanzaDependencyParser(lang='en'),
        'de': lambda: StanzaDependencyParser(lang='de'),
        'hu': lambda: SpacyDependencyParser(lang='hu')
    }

    @classmethod
    def get_dependency_parser(cls, lang_code) -> DependencyParserBase:
        return cls.dep_parsers[lang_code]()

