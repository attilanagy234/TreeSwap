import sentencepiece as spm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='[REQUIRED] Model to use for')
    parser.add_argument('-d', '--data-path', type=str, required=True, help='[REQUIRED] Path of the data to use bpe on.')
    parser.add_argument('-o', '--out', type=str, required=True, help='[REQUIRED] Output of the encoded text.')

    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model=args.model)
    with open(args.data_path, 'r') as input_file:
        with open(args.out, 'w') as output_file:
            for line in input_file:
                output_file.write(sp.encode(line))
                output_file.write('\n')
