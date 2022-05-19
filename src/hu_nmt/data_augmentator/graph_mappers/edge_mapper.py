import networkx as nx
from collections import defaultdict, Counter
import nltk


class EdgeMapper:
    def __init__(self, src_lang_code, tgt_lang_code, use_pos=False):
        self.deps = {}
        self.dep_weights = defaultdict(lambda: 1)
        if src_lang_code == 'hu' and tgt_lang_code == 'en':
            # self.deps['obl'] = 'case'
            self.deps['obl'] = 'obl'
        elif src_lang_code == 'en' and tgt_lang_code == 'hu':
            # self.deps['case'] = 'obl'
            self.deps['case'] = 'case'

        self.dep_weights['nsubj'] = 3
        self.dep_weights['obj'] = 3
        self.dep_weights['nmod'] = 2
        self.dep_weights['obl'] = 2
        self.dep_weights['case'] = 2
        self.dep_weights['amod'] = 2
        self.dep_weights['advmod'] = 2

        self.use_pos = use_pos

    def map_edges(self, g1: nx.DiGraph, g2: nx.DiGraph):
        g1 = self.add_weight(g1)
        g2 = self.add_weight(g2)
        mapping = {}

        g1_edges = list(sorted(g1.edges(data=True), key=lambda x: -x[2]['weight']))
        g2_edges = list(sorted(g2.edges(data=True), key=lambda x: -x[2]['weight']))

        for i, (s1, d1, data1) in enumerate(g1_edges):
            cands = list(filter(lambda x: x[2]['dep'] == self._get_dep_mapping(data1['dep']), g2_edges))
            if len(cands) == 1:
                mapping[(s1, d1)] = (cands[0][0], cands[0][1])
                g2_edges.remove(cands[0])
            elif len(cands) > 1:
                max_cands = self.get_max_cands((s1, d1, data1), cands, g1, g2)
                if len(max_cands) == 1:
                    mapping[(s1, d1)] = (max_cands[0][0], max_cands[0][1])
                    g2_edges.remove(max_cands[0])
                else:
                    min_routes = self._get_max_cand_by_route((s1, d1, data1), max_cands, g1, g2)
                    if len(min_routes) == 1:
                        mapping[(s1, d1)] = (min_routes[0][0], min_routes[0][1])
                        g2_edges.remove(min_routes[0])
                    else:
                        max_children = self._get_max_children((s1, d1, data1), max_cands, g1, g2)
                        if len(max_children) > 1:
                            print('AAAAAAAAAAAAAAAAAA')
                        mapping[(s1, d1)] = (max_children[0][0], max_children[0][1])
                        g2_edges.remove(max_children[0])
        return mapping

    def add_weight(self, g):
        for (n1, n2, data) in g.edges(data=True):
            w = self.dep_weights[data['dep']]
            g.add_weighted_edges_from([(n1, n2, w)])
        return self.adjust_deps(g)

    def adjust_deps(self, g):
        for (n1, n2, data) in g.edges(data=True):
            data['dep'] = data['dep'].split(':')[0].lower()
        return g

    def _get_dep_mapping(self, dep):
        if dep in self.deps:
            return self.deps[dep]
        return dep

    def get_max_cands(self, edge, cands, g1, g2):
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

    def _get_max_cand_by_route(self, edge, cands, g1, g2):
        (s1, d1, data1) = edge
        node_route1 = nx.shortest_path(g1, 'root_0', s1)
        route1 = []

        for i in range(len(node_route1) - 1):
            dep = g1.edges[node_route1[i], node_route1[i + 1]]['dep']
            route1.append(dep)

        min_dist = float('inf')
        min_edges = []

        for (s2, d2, data2) in cands:
            node_route2 = nx.shortest_path(g2, 'root_0', s2)
            route2 = []
            for i in range(len(node_route2) - 1):
                dep = g2.edges[node_route2[i], node_route2[i + 1]]['dep']
                route2.append(dep)
            dist = nltk.edit_distance(route1, route2)

            if dist == min_dist:
                min_edges.append((s2, d2, data2))
            elif dist < min_dist:
                min_dist = dist
                min_edges = [(s2, d2, data2)]
        return min_edges

    def _get_max_children(self, edge, cands, g1, g2):
        (s1, d1, data1) = edge
        children1 = [e[2]['dep'] for e in g1.out_edges(d1, data=True)]
        counter1 = Counter(children1)
        max_edges = []
        max_children = 0

        for (s2, d2, data2) in cands:
            children2 = [e[2]['dep'] for e in g2.out_edges(d2, data=True)]

            counter2 = Counter(children2)
            intersection = counter1 & counter2
            intersect_count = len(list(intersection.elements()))
            if intersect_count > max_children:
                max_children = intersect_count
                max_edges = [(s2, d2, data2)]
            elif intersect_count == max_children:
                max_edges.append((s2, d2, data2))
        return max_edges

    def get_jaccard_index(self, g1, g2, mapping):
        edges1 = len(g1.edges)
        edges2 = len(g2.edges)
        intersect = len(mapping)
        return (intersect) / (edges1 + edges2 - intersect)






