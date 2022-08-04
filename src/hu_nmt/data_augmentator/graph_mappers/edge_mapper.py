from collections import defaultdict, Counter
from typing import List

import networkx as nx
import nltk

from hu_nmt.data_augmentator.graph_mappers.graph_similarity_base import GraphSimilarityBase, Edge, Node
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper

"""
EdgeMapper creates a mapping between the edges
and calculates a Jaccard index using the mapping
as the intersection.
"""


class EdgeMapper(GraphSimilarityBase):
    def __init__(self):
        self.dep_weights = defaultdict(lambda: 1)

        self.dep_weights['nsubj'] = 3
        self.dep_weights['obj'] = 3
        self.dep_weights['nmod'] = 2
        self.dep_weights['obl'] = 2
        self.dep_weights['case'] = 2
        self.dep_weights['amod'] = 2
        self.dep_weights['advmod'] = 2

    def map_edges(self, g1: DependencyGraphWrapper, g2: DependencyGraphWrapper):
        g1 = DependencyGraphWrapper(self.add_weight(g1.graph))
        g2 = DependencyGraphWrapper(self.add_weight(g2.graph))
        mapping = {}

        g1_edges = list(sorted(g1.graph.edges(data=True), key=lambda x: -x[2]['weight']))
        g2_edges = list(sorted(g2.graph.edges(data=True), key=lambda x: -x[2]['weight']))

        for i, (s1, d1, data1) in enumerate(g1_edges):
            # edges with the same dependency label
            cands = list(filter(lambda x: x[2]['dep'] == data1['dep'], g2_edges))
            if len(cands) == 1:
                mapping[(s1, d1)] = (cands[0][0], cands[0][1])
                g2_edges.remove(cands[0])
            elif len(cands) > 1:
                # edges with the most similar node labels
                max_cands = self._get_cands_by_nodes((s1, d1, data1), cands, g1.graph, g2.graph)
                if len(max_cands) == 1:
                    mapping[(s1, d1)] = (max_cands[0][0], max_cands[0][1])
                    g2_edges.remove(max_cands[0])
                else:
                    # edges with the most similar root-edge routes
                    min_routes = self._get_cands_by_route((s1, d1, data1), max_cands, g1, g2)
                    if len(min_routes) == 1:
                        mapping[(s1, d1)] = (min_routes[0][0], min_routes[0][1])
                        g2_edges.remove(min_routes[0])
                    else:
                        # edges with the most similar children
                        max_children = self._get_cands_by_children((s1, d1, data1), min_routes, g1.graph, g2.graph)
                        mapping[(s1, d1)] = (max_children[0][0], max_children[0][1])
                        g2_edges.remove(max_children[0])
        return mapping

    def add_weight(self, graph: nx.DiGraph) -> nx.DiGraph:
        for (n1, n2, data) in graph.edges(data=True):
            w = self.dep_weights[data['dep']]
            graph.add_weighted_edges_from([(n1, n2, w)])
        return self.adjust_deps(graph)

    def adjust_deps(self, graph: nx.DiGraph) -> nx.DiGraph:
        for (n1, n2, data) in graph.edges(data=True):
            # use only the main dependency type
            data['dep'] = data['dep'].split(':')[0].lower()
        return graph

    def _get_cands_by_nodes(self, edge: Edge, cands: List[Edge], g1: nx.DiGraph, g2: nx.DiGraph):
        (s1, d1, data1) = edge
        s1_pos = g1.nodes[s1]['postag']
        d1_pos = g1.nodes[d1]['postag']
        max_score = 0
        max_edges = []
        for (s2, d2, data2) in cands:
            score = 0
            if g2.nodes[s2]['postag'] == s1_pos:
                score += 1
            if g2.nodes[d2]['postag'] == d1_pos:
                score += 1
            if max_score == score:
                max_edges.append((s2, d2, data2))
            elif score > max_score:
                max_edges = [(s2, d2, data2)]
                max_score = score
        return max_edges

    # most similar root-edge route based on the Levenshtein-distance
    def _get_cands_by_route(self, edge: Edge, cands: List[Edge], g1: DependencyGraphWrapper,
                            g2: DependencyGraphWrapper) -> List[Edge]:
        (s1, d1, data1) = edge
        node_route1 = nx.shortest_path(g1, g1.get_root(), s1)
        edge_route1 = self._get_edge_route_from_nodes(g1.graph, node_route1)

        min_dist = float('inf')
        min_edges = []

        # find edges with minimal distance
        for (s2, d2, data2) in cands:
            node_route2 = nx.shortest_path(g2, g2.get_root(), s2)
            edge_route2 = self._get_edge_route_from_nodes(g2.graph, node_route2)

            dist = nltk.edit_distance(edge_route1, edge_route2)

            if dist == min_dist:
                min_edges.append((s2, d2, data2))
            elif dist < min_dist:
                min_dist = dist
                min_edges = [(s2, d2, data2)]
        return min_edges

    def _get_edge_route_from_nodes(self, graph: nx.DiGraph, nodes: List[Node]):
        edge_route = []

        for i in range(len(nodes) - 1):
            dep = graph.graph.edges[nodes[i], nodes[i + 1]]['dep']
            edge_route.append(dep)

        return edge_route

    def _get_cands_by_children(self, edge: Edge, cands: List[Edge], g1: nx.DiGraph, g2: nx.DiGraph) -> List[Edge]:
        (s1, t1, data1) = edge

        # check target node's children
        max_edges = self._get_edges_with_max_children(t1, 'target', cands, g1, g2)

        # if more than one has the same children similarity
        # check source node children
        if len(max_edges) > 1:
            max_edges = self._get_edges_with_max_children(s1, 'source', cands, g1, g2)

        return max_edges

    def _get_edges_with_max_children(self, n1: Node, node_type: str, cands: List[Edge], g1: nx.DiGraph,
                                     g2: nx.DiGraph) -> List[Edge]:
        children1 = [e[2]['dep'] for e in g1.out_edges(n1, data=True)]
        counter1 = Counter(children1)
        max_edges = []
        max_children = 0

        for (s2, t2, data2) in cands:
            n2 = s2 if node_type == 'source' else t2
            children2 = [e[2]['dep'] for e in g2.out_edges(n2, data=True)]
            counter2 = Counter(children2)

            intersection = counter1 & counter2
            intersect_count = len(list(intersection.elements()))

            if intersect_count > max_children:
                max_children = intersect_count
                max_edges = [(s2, t2, data2)]
            elif intersect_count == max_children:
                max_edges.append((s2, t2, data2))

        return max_edges

    def get_jaccard_index_from_mapping(self, g1: nx.DiGraph, g2: nx.DiGraph, mapping):
        edges1 = len(g1.edges)
        edges2 = len(g2.edges)
        intersect = len(mapping)
        if edges1 == 0 and edges2 == 0:
            return 1
        return intersect / (edges1 + edges2 - intersect)

    def get_jaccard_index(self, g1: DependencyGraphWrapper, g2: DependencyGraphWrapper):
        mapping = self.map_edges(g1, g2)
        return self.get_jaccard_index_from_mapping(g1.graph, g2.graph, mapping)

    def get_similarity_from_graphs(self, g1: DependencyGraphWrapper, g2: DependencyGraphWrapper):
        return self.get_jaccard_index(g1, g2)
