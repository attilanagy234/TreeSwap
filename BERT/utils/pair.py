import argparse
import logging
import os
import re
import sys

import sentencepiece as spm

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-9s %(message)s'))

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(handler)


def pair_prediction(original_path:str, predicted_path:str, output_path:str=None, n_lines:{None, int}=None):
    with open(original_path, 'r') as f:
        original_lines = f.readlines()

    with open(predicted_path, 'r') as f:
        pred_lines = f.readlines()

    original_length = len(original_lines)
    pred_length = len(pred_lines)
    n_best = pred_length // original_length

    if pred_length % original_length != 0:
        log.error("Files are different length")
        return
    
    log.info(f"Beam n_best detected: {n_best}, # lines: {original_length}")
        
    if output_path is None:
        output_location = sys.stdout
    else:
        output_location = open(output_path, 'w')
    
    def get_lines(lines, n=1):
        i = 0
        for line in lines:
            print(line, end='', file=output_location)
            i += 1
            if i == n:
                i = 0
                yield
            
    
    if n_lines is None:
        n_lines = original_length
        
    original_generator = get_lines(original_lines)
    pred_generator = get_lines(pred_lines, n_best)
    
    for i in range(n_lines):
        next(original_generator)
        next(pred_generator)
        print(file=output_location)
        
    if output_path is not None:
        output_location.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--original_path', type=str, required=True, help='[REQUIRED] Path to decoded original dataset.')
    parser.add_argument('--predicted_path', type=str, required=True, help='[REQUIRED] Path to decoded prediction.')
    parser.add_argument('-o', '--output_path', dest='output_path', type=str, required=False, help='Output path, defaults to sys.stdout')
    parser.set_defaults(output_path=None)
    parser.add_argument('--n_lines', dest='n_lines', type=int, required=False, help='Pick first n_lines only, defaults to full text.')
    parser.set_defaults(n_lines=None)

    args = parser.parse_args()

    pair_prediction(args.original_path, args.predicted_path, args.output_path, args.n_lines)

