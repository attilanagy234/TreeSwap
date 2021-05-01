from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.hungarian_dependency_parser import HungarianDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.hungarian_dependency_parser_emtsv_docker import HungarianDependencyParserEmtsv
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
    eng_wrappers[1235].display_graph()
    hun_wrappers[1235].display_graph()
