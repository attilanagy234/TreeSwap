import sentencepiece as spm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, required=True, help='[REQUIRED] Path of the data to use bpe on.')
    parser.add_argument('-p', '--model-prefix', type=str, help='[OPTIONAL] Prefix of the model created.')
    parser.add_argument('--vocab-size', type=int, help='[OPTIONAL] Size of the vocabulary to create.')

    args = parser.parse_args()

    if args.model_prefix:
        model_prefix = args.model_prefix
    else:
        model_prefix = 'bpe'

    if args.vocab_size:
        vocab_size = args.vocab_size
    else:
        vocab_size = 30000

    spm.SentencePieceTrainer.train(input=args.data_path, model_prefix=model_prefix, vocab_size=vocab_size, model_type='bpe', character_coverage=1.0)

