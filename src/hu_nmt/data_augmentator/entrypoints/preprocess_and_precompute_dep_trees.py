import os

import click

from hu_nmt.data_augmentator.preprocessor.preprocessor import Preprocessor
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)


@click.command()
@click.argument('src_lang_code')
@click.argument('tgt_lang_code')
@click.argument('src_input_path')
@click.argument('tgt_input_path')
@click.argument('dep_tree_output_path')
@click.argument('preprocessed_output_path')
@click.argument('config_path')
def main(src_lang_code, tgt_lang_code, src_input_path, tgt_input_path, dep_tree_output_path, preprocessed_output_path,
         config_path):
    src_output_path = os.path.join(preprocessed_output_path, f'preprocessed.{src_lang_code}')
    tgt_output_path = os.path.join(preprocessed_output_path, f'preprocessed.{tgt_lang_code}')

    preprocessor = Preprocessor(src_input_path, tgt_input_path, config_path, src_output_path, tgt_output_path,
                                dep_tree_output_path)
    preprocessor.preprocess_and_precompute()


if __name__ == '__main__':
    main()
