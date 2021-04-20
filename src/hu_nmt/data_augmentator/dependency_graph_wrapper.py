import networkx as nx
from io import BytesIO
from IPython.display import Image, display
from hu_nmt.data_augmentator.dependency_parsers.english_dependency_parser import EnglishDependencyParser
from hu_nmt.data_augmentator.dependency_parsers.hungarian_dependency_parser import HungarianDependencyParser


class DependencyGraphWrapper:
    def __init__(self, config, sentence):
        self._config = config
        if self._config.data.lang == 'en':
            self._dep_parser = EnglishDependencyParser()
        if self._config.data.lang == 'hu':
            self._dep_parser = HungarianDependencyParser()
        self._graph = self._dep_parser.sentence_to_dep_parse_tree(sentence)

    def get_root(self):
        # since dep parsing always yields a tree, it should always have one element
        return [n for n, d in self._graph.in_degree() if d == 0][0]

    def get_distances_from_root(self):
        return nx.shortest_path_length(self._graph, self.get_root())

    def display_graph(self):
        agraph = nx.nx_agraph.to_agraph(self._graph) # graphviz agraph format
        agraph.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
        imgbuf = BytesIO()
        agraph.draw(imgbuf, format='png', prog='dot')
        img = Image(imgbuf.getvalue())
        display(img)

    def get_node_with_property(self, attribute_key, attribute_value):
        return [x for x, y in self._graph.nodes(data=True) if y[attribute_key] == attribute_value]

