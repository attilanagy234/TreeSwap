from typing import Dict

from networkx import graph_edit_distance

from hu_nmt.data_augmentator.graph_mappers.graph_similarity_base import GraphSimilarityBase
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper

"""
GED uses graph edit distance, with normalization
represent a similarity between 0 and 1:
sim = ( d_max - ged(g1, g2) ) / d_max
d_max: the maximum distance (deletes the source graph
and add the target graph)
"""


class GED(GraphSimilarityBase):
    _src_dep_parser = None
    _tgt_dep_parser = None

    def __init__(self, node_cost=1, edge_cost=1, node_subt=2, edge_subt=2, timeout=5):

        self.timeout = timeout
        self.node_subt = node_subt
        self.edge_subt = edge_subt
        self.edge_cost = edge_cost
        self.node_cost = node_cost

    def _node_match(self, n1: Dict[str, str], n2: Dict[str, str]):
        return n1['postag'] == n2['postag']

    def _edge_match(self, e1: Dict[str, str], e2: Dict[str, str]):
        return e1['dep'].split(':')[0].lower() == e2['dep'].split(':')[0].lower()

    def _node_del_or_add(self, n: Dict[str, str]):
        if n['postag'] == 'PUNCT':
            return 0
        else:
            return self.node_cost

    def _edge_del_or_add(self, n: Dict[str, str]):
        if n['dep'] == 'punct':
            return 0
        else:
            return self.edge_cost

    def _node_subst_cost(self, n1: Dict[str, str], n2: Dict[str, str]):
        if self._node_match(n1, n2):
            return 0
        else:
            return self.node_subt

    def _edge_subs_cost(self, e1: Dict[str, str], e2: Dict[str, str]):
        if self._edge_match(e1, e2):
            return 0
        else:
            return self.edge_subt

    def get_ged(self, graph1: DependencyGraphWrapper, graph2: DependencyGraphWrapper):
        return graph_edit_distance(graph1.graph, graph2.graph, self._node_match, self._edge_match,
                                   node_subst_cost=self._node_subst_cost, node_del_cost=self._node_del_or_add,
                                   node_ins_cost=self._node_del_or_add, edge_subst_cost=self._edge_subs_cost,
                                   edge_del_cost=self._edge_del_or_add, edge_ins_cost=self._edge_del_or_add,
                                   roots=(graph1.get_root(), graph2.get_root()), upper_bound=None,
                                   timeout=self.timeout)

    def get_similarity_from_graphs(self, graph1: DependencyGraphWrapper, graph2: DependencyGraphWrapper):
        init_distance = 0

        # if the root pos tags are not the same
        if graph1.graph.nodes[graph1.get_root()]['postag'] != graph2.graph.nodes[graph2.get_root()]['postag']:
            graph1.graph.nodes[graph1.get_root()]['postag'] = graph2.graph.nodes[graph2.get_root()]['postag']
            init_distance += 1

        dist = self.get_ged(graph1, graph2) + init_distance
        # distance of deleting source graph and adding target graph
        # delete: n - 1 edges + n nodes
        # add: n - 1 edges + n nodes
        max_dist = len(graph1.graph.nodes) * 2 - 1 + 2 * len(graph2.graph.nodes) - 1
        return float(max_dist - dist) / float(max_dist)
