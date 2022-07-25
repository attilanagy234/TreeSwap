from networkx import graph_edit_distance

from hu_nmt.data_augmentator.dependency_parsers.dependency_parser_factory import DependencyParserFactory
from hu_nmt.data_augmentator.graph_mappers.graph_similarity_base import GraphSimilarityBase


class GED(GraphSimilarityBase):

    _src_dep_parser = None
    _tgt_dep_parser = None

    def __init__(self, src_lang_code, tgt_lang_code, node_cost=1, edge_cost=1, node_subt=2, edge_subt=2, timeout=10):
        self.src_lang_code = src_lang_code
        self.tgt_lang_code = tgt_lang_code

        self.timeout = timeout
        self.node_subt = node_subt
        self.edge_subt = edge_subt
        self.edge_cost = edge_cost
        self.node_cost = node_cost

    @property
    def src_dep_parser(self):
        if not self._src_dep_parser:
            self._src_dep_parser = DependencyParserFactory.get_dependency_parser(self.src_lang_code)
        return self._src_dep_parser

    @property
    def tgt_dep_parser(self):
        if not self._tgt_dep_parser:
            self._tgt_dep_parser = DependencyParserFactory.get_dependency_parser(self.tgt_lang_code)
        return self._tgt_dep_parser

    def _node_match(self, n1, n2):
        return n1['postag'] == n2['postag']

    def _edge_match(self, e1, e2):
        return e1['dep'].split(':')[0].lower() == e2['dep'].split(':')[0].lower()

    def _node_del_or_add(self, n):
        if n['postag'] == 'PUNCT':
            return 0
        else:
            return self.node_cost

    def _edge_del_or_add(self, n):
        if n['dep'] == 'punct':
            return 0
        else:
            return self.edge_cost

    def _node_subst_cost(self, n1, n2):
        if self._node_match(n1, n2):
            return 0
        else:
            return self.node_subt

    def _edge_subs_cost(self, e1, e2):
        if self._edge_match(e1, e2):
            return 0
        else:
            return self.edge_subt

    def get_ged(self, graph1, graph2):
        return graph_edit_distance(graph1, graph2, self._node_match, self._edge_match,
                                   node_subst_cost=self._node_subst_cost, node_del_cost=self._node_del_or_add,
                                   node_ins_cost=self._node_del_or_add, edge_subst_cost=self._edge_subs_cost,
                                   edge_del_cost=self._edge_del_or_add, edge_ins_cost=self._edge_del_or_add,
                                   roots=(self._get_root(graph1), self._get_root(graph2)), upper_bound=None,
                                   timeout=self.timeout)

    def _get_root(self, graph):
        node = [n for n, d in graph.in_degree() if d == 0][0]
        return node

    def get_normalized_distance(self, graph1, graph2):
        dist = self.get_ged(graph1, graph2)
        max_dist = len(graph1.nodes) * 2 - 2 + 2 * len(graph2.nodes) - 2
        return float(max_dist - dist) / float(max_dist)

    def get_normalized_distance_from_sentences(self, src_sent, tgt_sent):
        src_graph = self.src_dep_parser.sentence_to_dep_parse_tree(src_sent)
        tgt_graph = self.tgt_dep_parser.sentence_to_dep_parse_tree(tgt_sent)

        return self.get_normalized_distance(src_graph, tgt_graph)

    def get_similarity_from_graphs(self, graph1, graph2):
        init = 0
        if graph1.nodes[self._get_root(graph1)]['postag'] != graph2.nodes[self._get_root(graph2)]['postag']:
            graph1.nodes[self._get_root(graph1)]['postag'] = graph2.nodes[self._get_root(graph2)]['postag']
            init += 1

        dist = self.get_ged(graph1, graph2) + init
        max_dist = len(graph1.nodes) * 2 - 1 + 2 * len(graph2.nodes) - 1
        if dist is None:
            print(f'{graph1.nodes} - {graph2.nodes}')
            print(f'{graph1.edges} - {graph2.edges}')
            print()
            return 0
        return float(max_dist - dist) / float(max_dist)

    def get_similarity_from_sentences(self, src_sent, tgt_sent):
        src_graph = self.src_dep_parser.sentence_to_dep_parse_tree(src_sent)
        tgt_graph = self.tgt_dep_parser.sentence_to_dep_parse_tree(tgt_sent)

        return self.get_normalized_distance(src_graph, tgt_graph)


