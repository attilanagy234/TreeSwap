import argparse

import sentencepiece as spm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, required=True, help='[REQUIRED] Path of the data to use bpe on.')
    parser.add_argument('-p', '--model-prefix', type=str, help='[OPTIONAL] Prefix of the model created.')
    parser.add_argument('--vocab-size', type=int, help='[OPTIONAL] Size of the vocabulary to create.')
    parser.add_argument('--model-type', required=True, type=str)
    parser.add_argument('--character-coverage', type=float,  default=1.0)

    args = parser.parse_args()

    spm.SentencePieceTrainer.train(
        input=args.data_path,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        input_sentence_size=500000,
        shuffle_input_sentence=True,
    )

