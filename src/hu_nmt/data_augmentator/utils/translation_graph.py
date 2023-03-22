from dataclasses import dataclass

from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


@dataclass
class TranslationGraph:
    src: DependencyGraphWrapper
    tgt: DependencyGraphWrapper
    similarity: float = 0

    def to_list(self):
        return [self.src, self.tgt]