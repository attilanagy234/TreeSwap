from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.hungarian_dependency_parser import HungarianDependencyParser
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper

if __name__ == '__main__':
    eng_data_folder = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/parsed_sentences'
    eng_dep_parser = EnglishDependencyParser()
    eng_wrappers = eng_dep_parser.get_graph_wrappers_from_files(eng_data_folder)
    print(len(eng_wrappers))
    hun_data_folder = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/hu/parsed_sentences'
    hun_dep_parser = HungarianDependencyParser()
    hun_wrappers = hun_dep_parser.get_graph_wrappers_from_files(hun_data_folder)
    print(len(hun_wrappers))
    augmentator = SubjectObjectAugmentator(eng_wrappers, hun_wrappers)

    # eng_sent = 'Peter carried the red cat to the other side of the road.'
    # hun_sent = 'Péter átvitte a piros macskát az út túloldalára.'
    # eng_graph = DependencyGraphWrapper(eng_dep_parser.sentence_to_dep_parse_tree(eng_sent))
    # hun_graph = DependencyGraphWrapper(hun_dep_parser.sentence_to_dep_parse_tree(hun_sent))
    # augmentator.test_graph_pair(hun_graph, eng_graph)
    augmentator.find_augmentable_candidates()
