import pathlib
import unittest

from hu_nmt.data_augmentator.base.nlp_pipeline_base import NlpPipelineBase
from hu_nmt.data_augmentator.graph_mappers.edge_mapper import EdgeMapper
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper


class EdgeMapperTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.edge_mapper = EdgeMapper()

        test_resource_dir = pathlib.Path(__file__).parent.parent.resolve() / 'resources' / 'graph_test'
        graphs = next(NlpPipelineBase.read_parsed_dep_trees_from_files(data_dir=test_resource_dir, per_file=True))

        cls.hun_graph = DependencyGraphWrapper(cls.edge_mapper.adjust_deps(graphs[0]))
        cls.eng_graph = DependencyGraphWrapper(cls.edge_mapper.adjust_deps(graphs[1]))

    def test_get_cands_by_node_labels(self):
        s = 'elmenekül_11'
        t = 'Megyéből_13'
        edge1_data = self.hun_graph.graph.edges[s, t]
        edge1 = (s, t, edge1_data)
        cands = list(filter(lambda x: x[2]['dep'] == edge1_data['dep'], self.eng_graph.graph.edges(data=True)))

        edges = self.edge_mapper._get_cands_by_node_labels(edge1, cands, self.hun_graph.graph, self.eng_graph.graph)

        # assert
        expected = [('flying_6', 'shire_9', {'dep': 'obl'})]
        self.assertListEqual(edges, expected)

    def test_get_cands_by_route(self):
        s = 'Megyéből_13'
        t = 'a_12'
        edge1_data = self.hun_graph.graph.edges[s, t]
        edge1 = (s, t, edge1_data)
        cands = list(filter(lambda x: x[2]['dep'] == edge1_data['dep'], self.eng_graph.graph.edges(data=True)))

        edges = self.edge_mapper._get_cands_by_route(edge1, cands, self.hun_graph, self.eng_graph)

        # assert
        expected = [('comforts_22', 'the_20', {'dep': 'det'})]
        self.assertListEqual(edges, expected)

    def test_get_edge_route_from_nodes(self):
        nodes_route = ['realized_4', 'mean_11', 'partings_14', 'painful_13', 'more_12']

        edge_route = self.edge_mapper._get_edge_route_from_nodes(self.eng_graph.graph, nodes_route)

        expected = ['ccomp', 'obj', 'amod', 'advmod']
        self.assertListEqual(edge_route, expected)

    def test_get_cands_by_children(self):
        s = 'ráeszmélt_2'
        t = 'Hirtelen_1'
        edge1_data = self.hun_graph.graph.edges[s, t]
        edge1 = (s, t, edge1_data)
        cands = list(filter(lambda x: x[2]['dep'] == edge1_data['dep'], self.eng_graph.graph.edges(data=True)))

        edges = self.edge_mapper._get_cands_by_children(edge1, cands, self.hun_graph.graph, self.eng_graph.graph)

        # assert
        expected = [('realized_4', 'suddenly_3', {'dep': 'advmod'})]
        self.assertListEqual(edges, expected)

    def test_get_edges_with_max_children_source(self):
        s = 'ráeszmélt_2'
        t = 'Hirtelen_1'
        edge1_data = self.hun_graph.graph.edges[s, t]
        cands = list(filter(lambda x: x[2]['dep'] == edge1_data['dep'], self.eng_graph.graph.edges(data=True)))

        edges = self.edge_mapper._get_edges_with_max_children(s, 'source', cands, self.hun_graph.graph,
                                                              self.eng_graph.graph)

        # assert
        expected = [('realized_4', 'suddenly_3', {'dep': 'advmod'})]
        self.assertListEqual(edges, expected)

    def test_get_edges_with_max_children_target(self):
        s = 'elmenekül_11'
        t = 'Megyéből_13'
        edge1_data = self.hun_graph.graph.edges[s, t]
        cands = list(filter(lambda x: x[2]['dep'] == edge1_data['dep'], self.eng_graph.graph.edges(data=True)))

        edges = self.edge_mapper._get_edges_with_max_children(t, 'target', cands, self.hun_graph.graph,
                                                              self.eng_graph.graph)

        # assert
        self.assertIn(('flying_6', 'shire_9', {'dep': 'obl'}), edges)
        self.assertIn(('saying_17', 'comforts_22', {'dep': 'obl'}), edges)
        self.assertEqual(len(edges), 2)

    def test_map_edges(self):
        mapping = self.edge_mapper.map_edges(self.hun_graph, self.eng_graph)

        self.assertEqual(len(mapping), 17)
        hun_edges = [k for k in mapping.keys()]
        eng_edges = [v for k, v in mapping.values()]
        self.assertEqual(len(hun_edges), len(set(hun_edges)))
        self.assertEqual(len(eng_edges), len(set(eng_edges)))

    def test_get_jaccard_index_from_mapping(self):
        mapping = self.edge_mapper.map_edges(self.hun_graph, self.eng_graph)

        jaccard = self.edge_mapper.get_jaccard_index_from_mapping(self.hun_graph.graph, self.eng_graph.graph, mapping)

        self.assertAlmostEqual(jaccard, 0.5152, 4)
