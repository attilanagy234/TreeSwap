import os
import click
from typing import Iterator

from hu_nmt.data_augmentator.dependency_parsers.nlp_pipeline_factory import NlpPipelineFactory
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)


@click.command()
@click.argument('src_lang_code')
@click.argument('tgt_lang_code')
@click.argument('src_input_path')
@click.argument('tgt_input_path')
@click.argument('dep_tree_output_path')
@click.argument('file_batch_size')
def main(src_lang_code, tgt_lang_code, src_input_path, tgt_input_path, dep_tree_output_path, file_batch_size):
    src_dep_parser = NlpPipelineFactory.get_dependency_parser(src_lang_code)
    tgt_dep_parser = NlpPipelineFactory.get_dependency_parser(tgt_lang_code)

    src_sentence_generator: Iterator = src_dep_parser.get_file_line_generator(src_input_path)
    tgt_sentence_generator: Iterator = tgt_dep_parser.get_file_line_generator(tgt_input_path)

    src_dep_parser.sentences_to_serialized_dep_graph_files(src_sentence_generator, os.path.join(dep_tree_output_path, src_lang_code), int(file_batch_size))
    tgt_dep_parser.sentences_to_serialized_dep_graph_files(tgt_sentence_generator, os.path.join(dep_tree_output_path, tgt_lang_code), int(file_batch_size))


if __name__ == '__main__':
    main()

