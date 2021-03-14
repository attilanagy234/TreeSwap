import os
import re
import argparse
import sys
import logging
import sentencepiece as spm


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


def create_decoded_translation(model_path: str, data_path: str, output_path: {str, None}, split_line: bool = True):
    new_file_path = output_path if output_path else data_path + '.sp'
    spp = spm.SentencePieceProcessor(model_file=model_path)
    with open(data_path, 'r') as original_file:
        with open(new_file_path, 'w') as decoded_file:
            for line in original_file:
                tokens = line.split() if split_line else line
                decoded_file.write(spp.decode(tokens) + ('\n' if split_line else ''))
    log.info('Created decoded translation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='[REQUIRED] Path of the subword model.')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='[REQUIRED] Path of the prediction to decode.')
    parser.add_argument('-o', '--output_path', type=str, required=False, help='Output path, defaults to <original filename>.sp')

    parser.add_argument('--decode_ids', dest='decode_ids', required=False, action='store_true', help='Source file contains token ids')
    parser.set_defaults(decode_ids=False)

    args = parser.parse_args()

    model_path = args.model
    data_path = args.data_path
    split_line = not args.decode_ids
    output_path = args.output_path

    create_decoded_translation(model_path, data_path, output_path, split_line)

