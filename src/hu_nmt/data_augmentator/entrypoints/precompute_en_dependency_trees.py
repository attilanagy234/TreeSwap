import click

from hu_nmt.data_augmentator.dependency_parsers.nlp_pipeline_factory import NlpPipelineFactory
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)


@click.command()
@click.argument('data_input_path')
@click.argument('dep_tree_output_path')
@click.argument('file_batch_size')
def main(data_input_path, dep_tree_output_path, file_batch_size):
    eng_dep_parser = NlpPipelineFactory.get_dependency_parser('en')
    eng_dep_parser.file_to_serialized_dep_graph_files(data_input_path, dep_tree_output_path, int(file_batch_size))


if __name__ == '__main__':
    main()

    # During local testing
    # $1 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/augmentation_input_data/train_200000_subsample.en
    # $2 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/parsed_sentences
    # $3 10000
