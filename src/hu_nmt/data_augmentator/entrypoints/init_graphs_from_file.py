from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser

if __name__ == '__main__':
    data_folder = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/parsed_sentences'
    eng_dep_parser = EnglishDependencyParser()
    wrappers = eng_dep_parser.get_graph_wrappers_from_files(data_folder)
    print(len(wrappers))
    for wrp in wrappers[0:10]:
        wrp.display_graph()
