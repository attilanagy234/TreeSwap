import unittest

import networkx as nx

from hu_nmt.data_augmentator.graph_mappers.edge_mapper import EdgeMapper
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


class EdgeMapperTest(unittest.TestCase):
    graph2: nx.DiGraph = None
    graph1: nx.DiGraph = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.edge_mapper = EdgeMapper()
        cls.graph1 = nx.DiGraph()
        cls.graph1.add_node('1', postag='tag1')
        cls.graph1.add_node('2', postag='tag2')
        cls.graph1.add_node('3', postag='tag3')
        cls.graph1.add_node('4', postag='tag4')
        cls.graph1.add_node('5', postag='tag5')
        cls.graph1.add_node('6', postag='tag6')
        cls.graph1.add_edge('1', '2', dep='dep1')
        cls.graph1.add_edge('1', '3', dep='dep2')
        cls.graph1.add_edge('2', '4', dep='dep3')
        cls.graph1.add_edge('2', '5', dep='dep4')
        cls.graph1.add_edge('3', '6', dep='dep5')

        cls.graph2 = nx.DiGraph()
        cls.graph2.add_node('a', postag='tag1')
        cls.graph2.add_node('b', postag='tag2')
        cls.graph2.add_node('c', postag='tag3')
        cls.graph2.add_node('d', postag='tag4')
        cls.graph2.add_node('e', postag='tag5')
        cls.graph2.add_node('f', postag='tag2')
        cls.graph2.add_edge('a', 'b', dep='dep1')
        cls.graph2.add_edge('a', 'c', dep='dep1')
        cls.graph2.add_edge('b', 'd', dep='dep3')
        cls.graph2.add_edge('c', 'e', dep='dep4')
        cls.graph2.add_edge('c', 'f', dep='dep3')

        """
            graph1:
                                    1
                                  [tag1]
                                 /      \ 
                             (dep1)    (dep2)
                               /          \ 
                              2            3
                           [tag2]        [tag3]
                           /    \            \ 
                       (dep3)  (dep4)       (dep5)
                         /        \            \ 
                        4          5            6  
                     [tag4]      [tag5]       [tag6]
            graph2:
                                    a
                                  [tag1]
                                 /      \ 
                             (dep1)    (dep1)
                               /          \ 
                              b            c
                           [tag2]        [tag3]
                           /             /    \ 
                        (dep3)       (dep4)  (dep3)
                         /             /        \ 
                        d             e          f  
                     [tag4]        [tag5]      [tag6]
        """

    def test_get_cands_by_node_labels(self):
        edge1_data = self.graph1.edges['1', '2']
        edge1 = ('1', '2', edge1_data)
        cands = list(filter(lambda x: x[2]['dep'] == edge1_data['dep'], self.graph2.edges(data=True)))

        edges = self.edge_mapper._get_cands_by_node_labels(edge1, cands, self.graph1, self.graph2)

        # assert
        expected = [('a', 'b', {'dep': 'dep1'})]
        self.assertListEqual(edges, expected)

    def test_get_cands_by_route(self):
        wrapper1 = DependencyGraphWrapper(self.graph1)
        wrapper2 = DependencyGraphWrapper(self.graph2)

        edge1_data = self.graph1.edges['2', '4']
        edge1 = ('2', '4', edge1_data)
        cands = list(filter(lambda x: x[2]['dep'] == edge1_data['dep'], self.graph2.edges(data=True)))
        edges = self.edge_mapper._get_cands_by_route(edge1, cands, wrapper1, wrapper2)

        # assert
        self.assertIn(('b', 'd', {'dep': 'dep3'}), edges)
        self.assertIn(('c', 'f', {'dep': 'dep3'}), edges)
        self.assertEqual(len(edges), 2)

    def test_get_edge_route_from_nodes(self):
        nodes_route = ['1', '2', '4']

        edge_route = self.edge_mapper._get_edge_route_from_nodes(self.graph1, nodes_route)

        expected = ['dep1', 'dep3']
        self.assertListEqual(edge_route, expected)

    def test_get_cands_by_children(self):
        edge1_data = self.graph1.edges['1', '2']
        edge1 = ('1', '2', edge1_data)
        cands = list(filter(lambda x: x[2]['dep'] == edge1_data['dep'], self.graph2.edges(data=True)))

        edges = self.edge_mapper._get_cands_by_children(edge1, cands, self.graph1, self.graph2)

        # assert
        expected = [('a', 'c', {'dep': 'dep1'})]
        self.assertListEqual(edges, expected)

    def test_map_edges(self):
        wrapper1 = DependencyGraphWrapper(self.graph1)
        wrapper2 = DependencyGraphWrapper(self.graph2)
        mapping = self.edge_mapper.map_edges(wrapper1, wrapper2)

        self.assertEqual(len(mapping), 3)
        hun_edges = [k for k in mapping.keys()]
        eng_edges = [v for k, v in mapping.values()]
        self.assertEqual(len(hun_edges), len(set(hun_edges)))
        self.assertEqual(len(eng_edges), len(set(eng_edges)))

    def test_get_jaccard_index_from_mapping(self):
        wrapper1 = DependencyGraphWrapper(self.graph1)
        wrapper2 = DependencyGraphWrapper(self.graph2)
        mapping = self.edge_mapper.map_edges(wrapper1, wrapper2)

        jaccard = self.edge_mapper.get_jaccard_index_from_mapping(self.graph1, self.graph2, mapping)

        self.assertAlmostEqual(jaccard, 0.4286, 4)
