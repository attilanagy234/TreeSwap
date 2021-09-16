import click

from hu_nmt.data_augmentator.preprocessor.preprocessor import Preprocessor
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)


@click.command()
@click.argument('src_input_path')
@click.argument('tgt_input_path')
@click.argument('config_path')
@click.argument('src_output_path')
@click.argument('tgt_output_path')
def main(src_input_path, tgt_input_path, config_path, src_output_path, tgt_output_path):
    preprocessor = Preprocessor(src_input_path, tgt_input_path, config_path, src_output_path, tgt_output_path)
    preprocessor.preprocess()


if __name__ == '__main__':
    main()

    # During local testing
    # $1 /Users/attilanagy/Personal/hu-nmt/data/ftp.mokk.bme.hu/Hunglish2/combined/hunglish2-train.en
    # $2 /Users/attilanagy/Personal/hu-nmt/data/ftp.mokk.bme.hu/Hunglish2/combined/hunglish2-train.hu
    # $3 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/preprocessor/configs/preprocessor_config.yaml
    # $4 /Users/attilanagy/Personal/hu-nmt/src/hu_nmt/data_augmentator/data/augmentation_input_data
