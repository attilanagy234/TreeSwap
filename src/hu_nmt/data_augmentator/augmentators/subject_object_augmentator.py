from hu_nmt.data_augmentator.base.augmentator_base import AugmentatorBase
from hu_nmt.data_augmentator.wrapper.dependency_graph_wrapper import DependencyGraphWrapper
from tqdm import tqdm


class SubjectObjectAugmentator(AugmentatorBase):

    def __init__(self, eng_graphs, hun_graphs):
        super().__init__()
        if len(eng_graphs) != len(hun_graphs):
            raise ValueError('Length of sentences must be equal for both langugages')
        self._eng_graphs = eng_graphs
        self._hun_graphs = hun_graphs
        self._augmentation_candidate_sentence_pairs = []

    def augment(self, eng_dep_graph, hun_dep_graph):
        raise NotImplementedError()

    def find_augmentable_candidates(self):
        for hun_graph, eng_graph in tqdm(zip(self._hun_graphs, self._eng_graphs)):
            if self.test_graph_pair(hun_graph, eng_graph):
                self._augmentation_candidate_sentence_pairs.append((hun_graph, eng_graph))

        print(len(self._augmentation_candidate_sentence_pairs))
        for sentence_pair in self._augmentation_candidate_sentence_pairs[0:3]:
            sentence_pair[0].display_graph()
            sentence_pair[1].display_graph()
            print(self.reconstruct_sentence_from_node_ids(sentence_pair[0].graph.nodes))
            print(self.reconstruct_sentence_from_node_ids(sentence_pair[1].graph.nodes))

    def test_graph_pair(self, hun_graph: DependencyGraphWrapper, eng_graph: DependencyGraphWrapper) -> bool:
        """
        Tests if a sentence (graph) pair is eligible for augmentation
        """

        hun_nsubj_edges = hun_graph.get_edges_with_property('dep', 'nsubj')
        eng_nsubj_edges = eng_graph.get_edges_with_property('dep', 'nsubj')

        hun_obj_edges = hun_graph.get_edges_with_property('dep', 'obj')
        eng_obj_edges = eng_graph.get_edges_with_property('dep', 'obj')

        # Should contain one nsubj and one obj in both languages
        if len(hun_nsubj_edges) != 1 or len(eng_nsubj_edges) != 1 or len(hun_obj_edges) != 1 or len(eng_obj_edges) != 1:
            return False
        else:
            hun_nsubj_edge = hun_nsubj_edges[0]
            eng_nsubj_edge = eng_nsubj_edges[0]
            hun_obj_edge = hun_obj_edges[0]
            eng_obj_edge = eng_obj_edges[0]

        # nsubj and obj edges have the same ancestor (predicate)
        if hun_nsubj_edge.source_node != hun_obj_edge.source_node or eng_nsubj_edge.source_node != eng_obj_edge.source_node:
            return False
        object_hun = hun_obj_edge.target_node
        object_eng = eng_obj_edge.target_node
        hun_obj_subgraph = hun_graph.get_subtree_node_ids(object_hun)
        eng_obj_subgraph = eng_graph.get_subtree_node_ids(object_eng)

        # Object subtree is consecutive
        if not self.is_consecutive_subsequence(hun_obj_subgraph):
            return False
        if not self.is_consecutive_subsequence(eng_obj_subgraph):
            return False
        return True

    @staticmethod
    def is_consecutive_subsequence(node_ids):
        def check(lst):
            lst = sorted(lst)
            if lst:
                return lst == list(range(lst[0], lst[-1] + 1))
            else:
                return True
        """
        Params:
            node_ids (list of Strings): list of node ids
        Returns:
            Boolean value whether the words corresponding to nodes
             are a consecutive subsequence in the original sentence
        """
        return check([int(x.split('-')[-1]) for x in node_ids])


