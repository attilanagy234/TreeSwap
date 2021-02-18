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


def create_decoded_translation(model_path: str, data_path: str):
    new_file_path = data_path + '.sp'
    spp = spm.SentencePieceProcessor(model_file=model_path)
    with open(data_path, 'r') as original_file:
        with open(new_file_path, 'w') as decoded_file:
            for line in original_file:
                tokens = line.split()
                decoded_file.write(spp.decode(tokens) + '\n')
    log.info('Created decoded translation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='[REQUIRED] Path of the subword model.')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='[REQUIRED] Path of the prediction to decode.')

    args = parser.parse_args()

    model_path = args.model
    data_path = args.data_path

    create_decoded_translation(model_path, data_path)

