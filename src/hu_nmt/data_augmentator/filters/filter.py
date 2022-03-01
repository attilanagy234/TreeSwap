from abc import abstractmethod
from typing import List, Tuple


class Filter:
    @abstractmethod
    def filter(self, src_sentences: List[str], tgt_sentences: List[str]) -> Tuple[List[str], List[str]]:
        raise NotImplementedError()

    @abstractmethod
    def get_pre_filter_data_multiplier(self) -> float:
        raise NotImplementedError()