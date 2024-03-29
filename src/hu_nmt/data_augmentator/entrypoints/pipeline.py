from hu_nmt.data_augmentator.augmentators.depth_based_blanking import DepthBasedBlanking
from hu_nmt.data_augmentator.augmentators.depth_based_dropout import DepthBasedDropout
from hu_nmt.data_augmentator.dependency_parsers.spacy_nlp_pipeline import SpacyNlpPipeline
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.dependency_parsers.stanza_nlp_pipeline import StanzaNlpPipeline
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)

if __name__ == '__main__':
    LANG = 'DE'
    # ------------ Test dependency parsers ------------
    if LANG == 'EN':
        #     Test English dependency parser
        sentence = 'Peter is eating a cake.'
        eng_dep_parser = StanzaNlpPipeline(lang='en')
        dep_graph = eng_dep_parser.sentence_to_dep_parse_tree(sentence)
        eng_dep_graph_wrapper = DependencyGraphWrapper(dep_graph)
        eng_dep_graph_wrapper.display_graph()
        graph = eng_dep_graph_wrapper
    elif LANG == 'HU':
    # Test Hungarian dependency parser
        sentence = 'A fekete kutya kergeti a piros macskát.'
   #     emtsv_output_file_path = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/hun_output.txt'
        hun_dep_parser = SpacyNlpPipeline(lang='hu')
        dep_graph = hun_dep_parser.sentence_to_dep_parse_tree(sentence)
        hun_dep_graph_wrapper = DependencyGraphWrapper(dep_graph)
        hun_dep_graph_wrapper.display_graph()
        graph = hun_dep_graph_wrapper
    elif LANG == 'DE':
        sentence = 'Ich liebe lange Spaziergänge in den Bergen.'
        de_dep_parser = StanzaNlpPipeline(lang='de')
        dep_graph = de_dep_parser.sentence_to_dep_parse_tree(sentence)
        hun_dep_graph_wrapper = DependencyGraphWrapper(dep_graph)
        hun_dep_graph_wrapper.display_graph()
        graph = hun_dep_graph_wrapper
    # ------------ Test augmentators ------------
    depth_based_blanker = DepthBasedBlanking()
    depth_based_dropout = DepthBasedDropout()
    augmented_sentence = depth_based_blanker.augment(graph)
    print(augmented_sentence)
    augmented_sentence = depth_based_dropout.augment(graph)
    print(augmented_sentence)
