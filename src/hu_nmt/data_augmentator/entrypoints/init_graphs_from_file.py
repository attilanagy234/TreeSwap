from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.hungarian_dependency_parser import HungarianDependencyParser
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper

if __name__ == '__main__':
    eng_data_folder = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/parsed_sentences'
    eng_dep_parser = EnglishDependencyParser()
    eng_wrappers = eng_dep_parser.get_graph_wrappers_from_files(eng_data_folder)
    print(len(eng_wrappers))

    hun_data_folder = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/hu/test_3000_subsample_hu_output.tsv'
    hun_dep_parser = HungarianDependencyParser(hun_data_folder)
    dep_graphs = hun_dep_parser.sentence_batch_to_dep_parse_trees()
    hun_wrappers = [DependencyGraphWrapper(x) for x in dep_graphs]
    print(len(hun_wrappers))
    for i in range(3):
        eng_wrappers[i].display_graph()
        hun_wrappers[i].display_graph()