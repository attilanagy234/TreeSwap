from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser

ENG_INPUT_PATH_BASE = '/Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/'
ENG_INPUT_FILE = 'eng_input.txt'
OUTPUT_FOLDER_NAME = 'parsed_sentences'


if __name__ == '__main__':
    eng_dep_parser = EnglishDependencyParser()
    sentences = []
    with open(ENG_INPUT_PATH_BASE + ENG_INPUT_FILE) as file:
        for line in file:
            sentences.append(line.strip())

    print(sentences)
    eng_dep_parser.sentences_to_serialized_dep_graph_files(sentences, f'{ENG_INPUT_PATH_BASE}{OUTPUT_FOLDER_NAME}', 5)

