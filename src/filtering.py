from hu_nmt.data_augmentator.translate.translate import get_translator
from hu_nmt.data_augmentator.utils.cosine_similarity import get_model, get_word_average_cos_similarity, get_sentence_embedding_cos_similarity
import os
from tqdm import tqdm



class Filter:
    def filter(self, src_path: str, tgt_path: str, output_path: str):
        raise NotImplementedError()


def filter_sentences(model_path, sp_model_path, tgt_lang, src_path, tgt_path, batch_size, output_path):
    translator = get_translator(
        model_path=model_path,
        sp_model_path=sp_model_path,
        batch_size=batch_size)

    tgt_embedding_model = get_model(tgt_lang)

    src_lines, tgt_lines = [], []
    with open(src_path, 'r') as src_file, open(tgt_path, 'r') as tgt_file, open(output_path, 'w') as output_file:
        for src_line, tgt_line in tqdm(zip(src_file, tgt_file)):
            src_lines.append(src_line.strip())
            tgt_lines.append(tgt_line.strip())
            
            if len(src_lines) % batch_size == 0:
                # translate lines
                translated_tgt_lines = translator.translate(src_lines)
            
                # check similarity
                scores = [get_sentence_embedding_cos_similarity(tgt_line, translated_tgt_line, en_embedding_model) for tgt_line, translated_tgt_line in zip(tgt_lines, translated_tgt_lines)]
            
                # save result
                for score, src_line, tgt_line in zip(scores, src_lines, tgt_lines):
                    output_file.write(f'{score[0][0]}\t')
                    output_file.write(f'{src_line}\t')
                    output_file.write(f'{tgt_line}\n')

                # clear batch
                src_lines, tgt_lines = [], []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='[REQUIRED] Config file to use.')
    parser.add_argument('-sp', '--sp_model', type=str, help='[REQUIRED] Config file to use.')
    parser.add_argument('-sp', '--tgt_lang', type=str, help='[REQUIRED] Config file to use.')
    parser.add_argument('-s', '--src', type=str, help='[REQUIRED] Config file to use.')
    parser.add_argument('-t', '--tgt', type=str, help='[REQUIRED] Config file to use.')
    parser.add_argument('--batch_size', type=int, help='[REQUIRED] Config file to use.')
    parser.add_argument('-o', '--output', type=int, help='[REQUIRED] Config file to use.')
    args = parser.parse_args()

    filter_sentences(args.model_path, args.sp_model_path, args.tgt_lang, args.src_path, args.tgt_path, args.bath_size, args.output_path)