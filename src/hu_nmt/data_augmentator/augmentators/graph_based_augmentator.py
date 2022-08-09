from typing import Tuple, List, Set, Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

from hu_nmt.data_augmentator.filters.filter import Filter
from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator
from hu_nmt.data_augmentator.graph_mappers.edge_mapper import EdgeMapper
from hu_nmt.data_augmentator.utils.logger import get_logger
from hu_nmt.data_augmentator.graph_mappers.ged import GED
from hu_nmt.data_augmentator.graph_mappers.graph_similarity_base import GraphSimilarityBase
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper

log = get_logger(__name__)
log.setLevel('DEBUG')


class GraphBasedAugmentator(SubjectObjectAugmentator):
    similarity: GraphSimilarityBase
    threshold: int

    def __init__(self,
                 src_lang_code: str,
                 tgt_lang_code: str,
                 threshold: int = 0.5,
                 eng_graphs: Optional[List[DependencyGraphWrapper]] = None,
                 hun_graphs: Optional[List[DependencyGraphWrapper]] = None,
                 augmented_data_ratio: float = 0.5,
                 random_seed: int = 123,
                 filters: List[Filter] = None,
                 output_path: str = './augmentations',
                 output_format: str = 'tsv',
                 save_original: bool = False,
                 separate_augmentation: bool = False,
                 similarity_type: str = '',
                 filter_nsub_and_obj_have_same_ancestor: bool = True,
                 filter_same_pos_tag: bool = True,
                 filter_for_noun_tags: bool = False):
        super().__init__(eng_graphs, hun_graphs, augmented_data_ratio, random_seed, filters, output_path, output_format,
                         save_original, separate_augmentation, filter_nsub_and_obj_have_same_ancestor,
                         filter_same_pos_tag, filter_for_noun_tags)

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
        failed = 0
        while len(sampled_index_pairs) < sample_count:
            (x, y) = np.random.choice(len(items), 2, replace=False)
            if GraphBasedAugmentator._is_similar(items[x].hun, items[y].hun, dep) and \
                    GraphBasedAugmentator._is_similar(items[x].eng, items[y].eng, dep):
                sampled_index_pairs.add((x, y))
                change = len(sampled_index_pairs) - size
                pbar.update(change)
                size = len(sampled_index_pairs)
            else:
                failed += 1
                if failed >= sample_count:
                    log.warning(f'Reached {sample_count} failed sentence pairs, decreasing the treshold from '
                                f'{GraphBasedAugmentator.threshold} to {GraphBasedAugmentator.threshold * 0.95}.')
                    GraphBasedAugmentator.threshold *= 0.95
                    failed = 0

        return [(items[x], items[y]) for x, y in sampled_index_pairs]

    @staticmethod
    def _is_similar(src_graph: DependencyGraphWrapper, tgt_graph: DependencyGraphWrapper, dep):
        src_subgraph = GraphBasedAugmentator._get_subgraph(src_graph, dep)
        tgt_subgraph = GraphBasedAugmentator._get_subgraph(tgt_graph, dep)
        return GraphBasedAugmentator.similarity.get_similarity_from_graphs(src_subgraph.graph, tgt_subgraph.graph) >= \
               GraphBasedAugmentator.threshold

    @staticmethod
    # it has only 1 obj or nsubj edge due to previous constraints
    def _get_subgraph(wrapper: DependencyGraphWrapper, dep: str) -> nx.DiGraph:
        edges_with_type = wrapper.get_edges_with_property('dep', dep)
        edges_with_type = edges_with_type[0]
        top_node_of_tree = edges_with_type.target_node
        node_ids = wrapper.get_subtree_node_ids(top_node_of_tree)
        subgraph = nx.DiGraph(wrapper.graph.subgraph(node_ids))
        return subgraph
