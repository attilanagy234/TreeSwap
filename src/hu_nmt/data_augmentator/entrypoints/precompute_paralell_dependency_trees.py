import os
import click
from typing import Iterator

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
    src_dep_parser = dep_parsers[src_tag]
    tgt_dep_parser = dep_parsers[tgt_tag]

    src_sentence_generator: Iterator = src_dep_parser._get_file_line_generator(src_input_path)
    tgt_sentence_generator: Iterator = tgt_dep_parser._get_file_line_generator(tgt_input_path)

    for file_idx, (src_batch, tgt_batch) in enumerate(zip(src_sentence_generator, tgt_sentence_generator)):
        src_list_of_dep_rel_lists = []
        tgt_list_of_dep_rel_lists = []
        for src_sent, tgt_sent in zip(src_batch, tgt_batch):
            src_sents = src_dep_parser.sentence_to_node_relationship_list(src_dep_parser.nlp_pipeline, src_sent)
            tgt_sents = tgt_dep_parser.sentence_to_node_relationship_list(src_dep_parser.nlp_pipeline, tgt_sent)

            if src_sents == [] or tgt_sents == []:
                continue
            src_list_of_dep_rel_lists.append(src_sents)
            tgt_list_of_dep_rel_lists.append(tgt_sents)

        src_dep_parser.write_dep_graphs_to_file(os.path.join(dep_tree_output_path, src_tag), file_idx, src_list_of_dep_rel_lists)
        tgt_dep_parser.write_dep_graphs_to_file(os.path.join(dep_tree_output_path, tgt_tag), file_idx, tgt_list_of_dep_rel_lists)


if __name__ == '__main__':
    main()

