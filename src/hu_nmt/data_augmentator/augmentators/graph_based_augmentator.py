from typing import Tuple, List, Set, Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

from filtering import Filter
from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator
from hu_nmt.data_augmentator.graph_mappers.graph_similarity_base import GraphSimilarityBase
from hu_nmt.data_augmentator.graph_mappers.edge_mapper import EdgeMapper
from hu_nmt.data_augmentator.graph_mappers.ged import GED
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


class GraphBasedAugmentator(SubjectObjectAugmentator):
    similarity: GraphSimilarityBase
    threshold = 0.5

    def __init__(self, src_lang_code, tgt_lang_code, threshold, eng_graphs: Optional[List[DependencyGraphWrapper]],
                 hun_graphs: Optional[List[DependencyGraphWrapper]], augmented_data_ratio: float, random_seed: int,
                 filters: List[Filter], output_path: str, output_format: str, save_original: bool = False,
                 separate_augmentation: bool = False, similarity_type=''):
        super().__init__(eng_graphs, hun_graphs, augmented_data_ratio, random_seed, filters, output_path, output_format,
                         save_original, separate_augmentation)

        GraphBasedAugmentator.threshold = threshold

        if similarity_type == 'ged':
            GraphBasedAugmentator.similarity = GED(src_lang_code, tgt_lang_code)
        elif similarity_type == 'edge_mapper':
            GraphBasedAugmentator.similarity = EdgeMapper()

    @staticmethod
    def sample_item_pairs(items: List, sample_count: int, dep: str = 'both'):
        sampled_index_pairs: Set[Tuple[int, int]] = set()
        pbar = tqdm(total=sample_count)
        size = 0
        while len(sampled_index_pairs) < sample_count:
            (x, y) = np.random.choice(len(items), 2, replace=False)
            if GraphBasedAugmentator._is_similar(items[x].hun, items[y].hun, dep) and \
                    GraphBasedAugmentator._is_similar(items[x].eng, items[y].eng, dep):
                sampled_index_pairs.add((x, y))
                change = len(sampled_index_pairs) - size
                pbar.update(change)
                size = len(sampled_index_pairs)

        return [(items[x], items[y]) for x, y in sampled_index_pairs]

    @staticmethod
    def _is_similar(src_graph: DependencyGraphWrapper, tgt_graph: DependencyGraphWrapper, dep):
        src_subgraph = GraphBasedAugmentator._get_subgraph(src_graph, dep)
        tgt_subgraph = GraphBasedAugmentator._get_subgraph(tgt_graph, dep)
        return GraphBasedAugmentator.similarity.get_similarity_from_graphs(src_subgraph.graph, tgt_subgraph.graph) >= \
            GraphBasedAugmentator.threshold

    @staticmethod
    def _get_subgraph(wrapper: DependencyGraphWrapper, dep: str) -> nx.DiGraph:
        edges_with_type = wrapper.get_edges_with_property('dep', dep)
        edges_with_type = edges_with_type[0]
        top_node_of_tree = edges_with_type.target_node
        node_ids = wrapper.get_subtree_node_ids(top_node_of_tree)
        subgraph = nx.DiGraph(wrapper.graph.subgraph(node_ids))
        return subgraph
