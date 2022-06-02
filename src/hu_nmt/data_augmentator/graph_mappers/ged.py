from networkx import graph_edit_distance


class GED:
    def __init__(self, node_cost=1, edge_cost=1, node_subt=2, edge_subt=2, timeout=10):
        self.timeout = timeout
        self.node_subt = node_subt
        self.edge_subt = edge_subt
        self.edge_cost = edge_cost
        self.node_cost = node_cost

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
                                   roots=(graph1.nodes['root_0'], graph2.nodes['root_1']), upper_bound=None,
                                   timeout=self.timeout)

    def get_normalized_distance(self, graph1, graph2):
        dist = self.get_ged(graph1, graph2)
        max_dist = len(graph1.nodes) * 2 - 2 + 2 * len(graph2.nodes) - 2
        return float(max_dist - dist) / float(max_dist)

