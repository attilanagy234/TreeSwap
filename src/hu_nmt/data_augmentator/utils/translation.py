from dataclasses import dataclass

from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


@dataclass
class TranslationGraph:
    hun: DependencyGraphWrapper
    eng: DependencyGraphWrapper
