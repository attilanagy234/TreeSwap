from typing import List
from hu_nmt.data_augmentator.filters.basic_filter import BasicFilter
from hu_nmt.data_augmentator.utils.cosine_similarity import get_model, get_sentence_embedding_cos_similarity


class CosSimilarityFilter(BasicFilter):
    def __init__(self, filter_quantile: float, src_model_path: str, sp_model_path: str, tgt_lang: str, batch_size: int):
        super().__init__(filter_quantile, src_model_path, sp_model_path, tgt_lang, batch_size)
        self.tgt_embedding_model = get_model(self.tgt_lang)

    def _score_lines(self, src_lines: List[str], tgt_lines: List[str]) -> List[float]:
        # translate lines
        translated_tgt_lines = self.src_translator.translate(src_lines)
    
        # check similarity
        tgt_scores = [get_sentence_embedding_cos_similarity(tgt_line, translated_tgt_line, self.tgt_embedding_model) for tgt_line, translated_tgt_line in zip(tgt_lines, translated_tgt_lines)]

        return tgt_scores
