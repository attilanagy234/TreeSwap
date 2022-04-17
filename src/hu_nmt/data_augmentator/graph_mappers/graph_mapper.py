from hu_nmt.data_augmentator.dependency_parsers.spacy_dependency_parser import SpacyDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser

from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.augmentators.subject_object_augmentator import SubjectObjectAugmentator

import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt


class GraphMapper:
    def __init__(self):
        self.hun_dep_parser = SpacyDependencyParser(lang='hu')
        self.eng_dep_parser = EnglishDependencyParser()
        self.weights = defaultdict(lambda: 1)
        self.weights['nsubj'] = 5
        self.weights['object'] = 5

    def map_subgraphs(self, hun_graph, eng_graph):
        hun_graph, eng_graph = self.add_all_weights(hun_graph, eng_graph)

        m = {}
        best_map = {}
        score = 0

        m[self.get_root(hun_graph)] = self.get_root(eng_graph)

        hu_rem = list(hun_graph.nodes(data=True))
        en_rem = list(eng_graph.nodes(data=True))

        while len(hu_rem) > 0:
            max_cand, score2 = self.get_max_score_candidates(hu_rem, en_rem, m, hun_graph, eng_graph)
            cands = self.max_look_ahead(max_cand, hun_graph, eng_graph)
            if cands is None:
                break
            score += score2
            (n1, n2) = max_cand[0]
            hu_rem = [(n, d) for (n, d) in hu_rem if n != n1]
            en_rem = [(n, d) for (n, d) in en_rem if n != n2]
            m[n1] = n2
        return m, score

    def get_root(self, graph):
        # This should yield the artificial ROOT node on top of the dependency tree
        return [n for n, d in graph.in_degree() if d == 0][0]

    weights = defaultdict(lambda: 1)
    weights['nsubj'] = 5
    weights['object'] = 5

    def add_weight(self, g):
        for (n1, n2, data) in g.edges(data=True):
            w = self.weights[data['dep']]
            g.add_weighted_edges_from([(n1, n2, w)])
        return g

    def add_all_weights(self, hun_graph, eng_graph):
        m_node = {}
        for n in hun_graph.nodes:
            m_node[n] = 1
        nx.set_node_attributes(hun_graph, m_node, name="weight")
        m_node = {}
        for n in eng_graph.nodes:
            m_node[n] = 1
        nx.set_node_attributes(eng_graph, m_node, name="weight")
        hun_graph = self.add_weight(hun_graph)
        eng_graph = self.add_weight(eng_graph)
        return hun_graph, eng_graph

    def get_max_score_candidates(self, hu_nodes, en_nodes, m, hu_graph, en_graph):
        max_score = 0
        max_cands = []
        score = 0
        for (n1, data1) in hu_nodes:
            for (n2, data2) in en_nodes:
                score = 0
                # print(f'Looking {n1}, {n2}')
                if data1['postag'] == data2['postag']:
                    # print('  postag ok')
                    score += data1['weight']
                    score += data2['weight']
                n1_in = hu_graph.in_edges(n1, data=True)
                for (s1, n1, edata) in n1_in:
                    if s1 in m:
                        e = en_graph.get_edge_data(m[s1], n2)
                        if e is not None and e['dep'] == edata['dep']:
                            # print(f'  edge ok: {(s1, n1)} - {(m[s1], n2)}')
                            score += e['weight']
                            score += edata['weight']
                # print(f'  {(n1, n2, score)}')
                if max_score == score:
                    max_cands.append((n1, n2))
                elif score > max_score:
                    max_score = score
                    max_cands = [(n1, n2)]
        return max_cands, max_score

    def get_common_weight(self, edges1, edges2):
        w = 0
        for (n1, n2, data) in edges1:
            for (m1, m2, data2) in edges2:
                if data['dep'].split(':')[0] == data2['dep'].split(':')[0]:
                    w += data['weight']
        return w

    def max_look_ahead(self, cands, hu_graph, en_graph):
        if len(cands) == 0:
            return None
        max_score = 0
        max_cand = None
        for (n1, n2) in cands:
            n1_out = hu_graph.out_edges(n1, data=True)
            n2_out = en_graph.out_edges(n2, data=True)
            n1_in = hu_graph.in_edges(n1, data=True)
            n2_in = en_graph.in_edges(n2, data=True)

            w1 = self.get_common_weight(n1_out, n2_out)
            w2 = self.get_common_weight(n1_in, n2_in)

            if w1 + w2 > max_score:
                max_cand = (n1, n2)
                max_score = w1 + w2
        return max_cand





