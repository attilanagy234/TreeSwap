import os
import click
from typing import Iterator

from hu_nmt.data_augmentator.dependency_parsers.dependency_parser_factory import DependencyParserFactory
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
    src_dep_parser = DependencyParserFactory.get_dependency_parser(src_lang_code)
    tgt_dep_parser = DependencyParserFactory.get_dependency_parser(tgt_lang_code)

    src_sentence_generator: Iterator = src_dep_parser._get_file_line_generator(src_input_path)
    tgt_sentence_generator: Iterator = tgt_dep_parser._get_file_line_generator(tgt_input_path)

    have_more_sentences_to_process = True
    src_batch_of_sentences = []
    tgt_batch_of_sentences = []
    while have_more_sentences_to_process:
        try:
            for _ in range(int(file_batch_size)):
                src_batch_of_sentences.append(next(src_sentence_generator))
                tgt_batch_of_sentences.append(next(tgt_sentence_generator))
        except StopIteration:
            have_more_sentences_to_process = False

        src_list_of_dep_rel_lists = []
        tgt_list_of_dep_rel_lists = []
        for file_idx, (src_sent, tgt_sent) in enumerate(zip(src_batch_of_sentences, tgt_batch_of_sentences)):
            src_sents = src_dep_parser.sentence_to_node_relationship_list(src_dep_parser.nlp_pipeline, src_sent)
            tgt_sents = tgt_dep_parser.sentence_to_node_relationship_list(tgt_dep_parser.nlp_pipeline, tgt_sent)
            if src_sents == [] or tgt_sents == []:
                continue
            src_list_of_dep_rel_lists.append(src_sents)
            tgt_list_of_dep_rel_lists.append(tgt_sents)

        src_dep_parser.write_dep_graphs_to_file(os.path.join(dep_tree_output_path, src_lang_code), 1, src_list_of_dep_rel_lists)
        tgt_dep_parser.write_dep_graphs_to_file(os.path.join(dep_tree_output_path, tgt_lang_code), 1, tgt_list_of_dep_rel_lists)


if __name__ == '__main__':
    main()

