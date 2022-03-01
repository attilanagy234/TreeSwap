from abc import abstractmethod
from typing import List, Tuple
from hu_nmt.data_augmentator.filters.filter import Filter
from hu_nmt.data_augmentator.translate.translate import get_translator
from hu_nmt.data_augmentator.utils.cosine_similarity import get_model, get_sentence_embedding_cos_similarity
import numpy as np


class BasicFilter(Filter):
    def __init__(self, filter_quantile: float, src_model_path: str, tgt_model_path: str, sp_model_path: str, tgt_lang: str, batch_size: int):
        self.filter_quantile = filter_quantile
        self.src_model_path = src_model_path
        self.tgt_model_path = tgt_model_path
        self.sp_model_path = sp_model_path
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size

        self.src_translator = get_translator(model_path=self.src_model_path,
                                            sp_model_path=self.sp_model_path,
                                            batch_size=self.batch_size)
        self.tgt_translator = get_translator(model_path=self.tgt_model_path,
                                            sp_model_path=self.sp_model_path,
                                            batch_size=self.batch_size)

    @abstractmethod
    def _score_lines(self, src_lines: List[str], tgt_lines: List[str]) -> Tuple[List[float], List[float]]:
        raise NotImplementedError

    def _batch(self, list_to_batch: List):
        for i in range(0, len(list_to_batch), self.batch_size):
            yield list_to_batch[i:i+self.batch_size]

    def _get_scores(self, src_sentences: str, tgt_sentences: str) -> Tuple[List[float], List[float]]:
        all_src_scores, all_tgt_scores = [], []
        for src_sent_batch, tgt_sent_batch in zip(self._batch(src_sentences), self._batch(tgt_sentences)):
            src_scores, tgt_scores = self._score_lines(src_sent_batch, tgt_sent_batch)
            all_src_scores.extend(src_scores)
            all_tgt_scores.extend(tgt_scores)
        return all_src_scores, all_tgt_scores

    def filter(self, src_sentences: List[str], tgt_sentences: List[str]) -> Tuple[List[str], List[str]]:
        src_scores, tgt_scores = self._get_scores(src_sentences, tgt_sentences)
        src_quantile_value = np.quantile(np.array(src_scores), self.filter_quantile)
        tgt_quantile_value = np.quantile(np.array(tgt_scores), self.filter_quantile)

        src_filtered_sentences, tgt_filtered_sentences = [], []
        for src_score, tgt_score, src_sent, tgt_sent in zip(src_scores, tgt_scores, src_sentences, tgt_sentences):
            if src_score > src_quantile_value and tgt_score > tgt_quantile_value:
                src_filtered_sentences.append(src_sent)
                tgt_filtered_sentences.append(tgt_sent)

        return src_filtered_sentences, tgt_filtered_sentences

    def get_pre_filter_data_multiplier(self) -> float:
        return 1 + self.filter_quantile
