from typing import Iterator

import click

from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
from hu_nmt.data_augmentator.dependency_parsers.spacy_dependency_parser import SpacyDependencyParser
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser

log = get_logger(__name__)


@click.command()
@click.argument('src_tag')
@click.argument('tgt_tag')
@click.argument('src_input_path')
@click.argument('tgt_input_path')
@click.argument('dep_tree_output_path')
@click.argument('file_batch_size')
def main(src_tag, tgt_tag, src_input_path, tgt_input_path, dep_tree_output_path, file_batch_size):
    dep_parsers = {
        'en': EnglishDependencyParser(),
        'hu': SpacyDependencyParser(lang='hu')
    }
    src_dep_parser: DependencyParserBase = dep_parsers[src_tag]
    tgt_dep_parser = dep_parsers[tgt_tag]

    src_sentence_generator: Iterator = None
    tgt_sentence_generator: Iterator = None

    for src_batch, tgt_batch in zip(src_sentence_generator, tgt_sentence_generator):
        for src_sent, tgt_sent in zip(src_batch, tgt_batch):
            pass




    eng_dep_parser = EnglishDependencyParser()
    eng_dep_parser.file_to_serialized_dep_graph_files(data_input_path, dep_tree_output_path, int(file_batch_size))


if __name__ == '__main__':
    main()

    # During local testing
    # $1 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/augmentation_input_data/train_200000_subsample.en
    # $2 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/eng/parsed_sentences
    # $3 10000
