

class SubjectObjectAugmentator:
    def __init__(self, eng_graphs, hun_graphs):
        self._eng_graph = eng_graphs
        self._hun_graphs = hun_graphs
        self._augmentation_candidate_sentence_pairs = []


    def augment(self, eng_dep_graph, hun_dep_graph):
        raise NotImplementedError()