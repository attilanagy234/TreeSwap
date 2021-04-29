from hu_nmt.data_augmentator.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.utils.data_helpers import get_config_from_yaml

ENG_INPUT_PATH_BASE = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/'
ENG_INPUT_FILE = 'eng_input.txt'
OUTPUT_FOLDER_NAME = 'parsed_sentences'


if __name__ == '__main__':
    # TODO: convert this to easily runnable scripts
    eng_dep_parser = EnglishDependencyParser()
    sentences = []
    with open(ENG_INPUT_PATH_BASE + ENG_INPUT_FILE) as file:
        for line in file:
            sentences.append(line.strip())

    # eng_dep_parser.sentences_to_serialized_dep_graph_files(sentences, f'{ENG_INPUT_PATH_BASE}{OUTPUT_FOLDER_NAME}', 2)

    dep_graphs = eng_dep_parser.read_parsed_dep_trees_from_files(f'{ENG_INPUT_PATH_BASE}{OUTPUT_FOLDER_NAME}')
    config = get_config_from_yaml('./configs/example_en_config.yaml')
    wrappers = [DependencyGraphWrapper(config, x) for x in dep_graphs]
    for wrp in wrappers:
        wrp.display_graph()
