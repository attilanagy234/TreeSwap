from hu_nmt.data_augmentator.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.utils.data_helpers import get_config_from_yaml

if __name__ == '__main__':
    # Test hungarian dependency parser
    config = get_config_from_yaml('./configs/example_en_config.yaml')
    sentence = 'This is a very colorful rainbow.'
    eng_dep_parser = EnglishDependencyParser()
    dep_graph = eng_dep_parser.sentence_to_dep_parse_tree(sentence)
    eng_dep_graph_wrapper = DependencyGraphWrapper(config, dep_graph)
  #  eng_dep_graph_wrapper.display_graph()
