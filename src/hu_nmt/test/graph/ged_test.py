import unittest

import networkx as nx

from hu_nmt.data_augmentator.graph_mappers.ged import GED
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


class GEDTest(unittest.TestCase):
    graph2: nx.DiGraph = None
    graph1: nx.DiGraph = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.ged = GED()
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

    def test_get_similarity_from_graphs(self):
        wrapper1 = DependencyGraphWrapper(self.graph1)
        wrapper2 = DependencyGraphWrapper(self.graph2)

        sim = self.ged.get_similarity_from_graphs(wrapper1, wrapper2)

        # assert
        self.assertAlmostEqual(sim, 0.6364, 4)

    def test_get_similarity_from_graphs_root_postag_mismatch(self):
        graph1_new_postag = nx.DiGraph(self.graph1)
        graph1_new_postag.nodes['1']['postag'] = 'newtag'

        wrapper1 = DependencyGraphWrapper(graph1_new_postag)
        wrapper2 = DependencyGraphWrapper(self.graph2)

        sim = self.ged.get_similarity_from_graphs(wrapper1, wrapper2)

        # assert
        self.assertAlmostEqual(sim, 0.5909, 4)


