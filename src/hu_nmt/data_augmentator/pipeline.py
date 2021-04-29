from hu_nmt.data_augmentator.augmentators.depth_based_augmentator import DepthBasedAugmentator
from hu_nmt.data_augmentator.augmentators.depth_based_blanking import DepthBasedBlanking
from hu_nmt.data_augmentator.augmentators.depth_based_dropout import DepthBasedDropout
from hu_nmt.data_augmentator.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.hungarian_dependency_parser import HungarianDependencyParser
from hu_nmt.data_augmentator.utils.data_helpers import get_config_from_yaml
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)

if __name__ == '__main__':
    LANG = 'EN'
    # ------------ Test dependency parsers ------------
    if LANG == 'EN':
        #     Test English dependency parser
        config = get_config_from_yaml('./configs/example_en_config.yaml')
        sentence = 'The black dog is chasing the red cat.'
        eng_dep_parser = EnglishDependencyParser()
        dep_graph = eng_dep_parser.sentence_to_dep_parse_tree(sentence)
        eng_dep_graph_wrapper = DependencyGraphWrapper(config, dep_graph)
        eng_dep_graph_wrapper.display_graph()
        graph = eng_dep_graph_wrapper
    elif LANG == 'HU':
    # Test Hungarian dependency parser
        config = get_config_from_yaml('./configs/example_hu_config.yaml')
        emtsv_output_file_path = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/hun_output.txt'
        hun_dep_parser = HungarianDependencyParser(emtsv_output_file_path)
        dep_graphs = hun_dep_parser.sentence_batch_to_dep_parse_trees()
        for dep_graph in dep_graphs:
            hun_dep_graph_wrapper = DependencyGraphWrapper(config, dep_graph)
            hun_dep_graph_wrapper.display_graph()
            graph = hun_dep_graph_wrapper
    # ------------ Test augmentators ------------
    depth_based_blanker = DepthBasedBlanking(config)
    depth_based_dropout = DepthBasedDropout(config)
    augmented_sentence = depth_based_blanker.augment_sentence_from_dep_graph(graph)
    print(augmented_sentence)
    augmented_sentence = depth_based_dropout.augment_sentence_from_dep_graph(graph)
    print(augmented_sentence)
