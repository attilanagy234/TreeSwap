import click
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser

log = get_logger(__name__)


@click.command()
@click.argument('data_input_path')
@click.argument('dep_tree_output_path')
@click.argument('file_batch_size')
def main(data_input_path, dep_tree_output_path, file_batch_size):
    sentences = []
    log.info('Reading sentences...')
    with open(data_input_path) as file:
        for line in file:
            sentences.append(line.strip())
    log.info(f'Preparing {len(sentences)} sentences for dependency parsing...')
    eng_dep_parser = EnglishDependencyParser()
    eng_dep_parser.sentences_to_serialized_dep_graph_files(sentences, dep_tree_output_path, file_batch_size)


if __name__ == '__main__':
    main()

    # During local testing
    # $1 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/augmentation_test_data/train_200000_subsample.en
    # $2 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/parsed_sentences
    # $3 10000
