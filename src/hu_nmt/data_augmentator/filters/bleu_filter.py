from typing import List, Tuple
from hu_nmt.data_augmentator.filters.basic_filter import BasicFilter
from hu_nmt.data_augmentator.utils.cosine_similarity import get_sentence_embedding_cos_similarity
from sacrebleu.metrics import BLEU



class BleuFilter(BasicFilter):
    def __init__(self, filter_quantile: float, src_model_path: str, tgt_model_path: str, sp_model_path: str, tgt_lang: str, batch_size: int):
        super().__init__(filter_quantile, src_model_path, tgt_model_path, sp_model_path, tgt_lang, batch_size)
        self.bleu_scorer = BLEU()

    def _score_lines(self, src_lines: List[str], tgt_lines: List[str]) -> Tuple[List[float], List[float]]:
        # translate lines
        translated_tgt_lines = self.src_translator.translate(src_lines)
        translated_src_lines = self.tgt_translator.translate(tgt_lines)
    
        # calculate bleu scores
        tgt_scores = [self.bleu_scorer.corpus_score([tgt_line], translated_tgt_line).score for tgt_line, translated_tgt_line in zip(tgt_lines, translated_tgt_lines)]
        src_scores = [self.bleu_scorer.corpus_score([src_line], translated_src_line).score for src_line, translated_src_line in zip(src_lines, translated_src_lines)]

        return src_scores, tgt_scores

if __name__ == '__main__':
    bleu = BLEU()
    refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.']]
    sys =  ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

    res = bleu.corpus_score(sys, refs)
    print(res.score)