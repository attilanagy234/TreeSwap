from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
import pandas as pd
import networkx as nx

ROOT_KEY = 'root-0'


class HungarianDependencyParser(DependencyParserBase):
    """
    This can only be used in a batched format atm,
    because it reads from a file generated by the emtsv dep parser,
    which is run from Docker
    """
    def __init__(self, emtsv_output_file_path):
        super().__init__()
        self._emtsv_output_file_path = emtsv_output_file_path

    def sentence_to_dep_parse_tree(self, sentence):
        raise NotImplementedError()

    def sentence_batch_to_dep_parse_trees(self):
        """
        Args:
             sentence_batch_dfs (list of DataFrames): parsed from the emtsv output file\
        Returns:
             dep_graphs (list of networkx graphs): dependency trees
        """
        sentence_batch_dfs = self.parse_emtsv_output()
        dep_graphs = []
        for df in sentence_batch_dfs:
            dep_graphs.append(self.emtsv_dataframe_to_nxgraph(df))

        return dep_graphs

    def parse_emtsv_output(self):
        first_line = True
        sentences = []
        token_buffer = []
        with open(self._emtsv_output_file_path, 'r') as file:
            for line in file:
                if first_line:
                    headers = [x.strip() for x in line.split('\t')]
                    first_line = False
                    continue
                if line == '\n':
                    sentences.append(token_buffer)
                    token_buffer = []
                else:
                    token_buffer.append([x.strip() for x in line.split('\t')])

        # Using dataframes here will have worse performance, but easier to use. Might need to optimize
        # Use lists+maps or reindex ids to be globally unique in case of multiple sentences and use one dataframe
        dataframes = [pd.DataFrame(sentence, columns=headers) for sentence in sentences]
        return dataframes

    @staticmethod
    def emtsv_dataframe_to_nxgraph(df):
        dep_graph = nx.DiGraph()
        # Add ROOT node
        dep_graph.add_node(ROOT_KEY)
        for row in df.itertuples(index=True, name='Pandas'):
            target_key = f'{row.form.lower()}-{row.id}'
            target_postag = row.upostag
            target_deprel = row.deprel
            if row.head != '0':
                head = df.iloc[int(row.head) - 1]
                source_key = f'{head.form.lower()}-{head.id}'
                source_postag = head.upostag
            else:
                source_key = ROOT_KEY
                source_postag = None
            dep_graph.add_node(source_key, postag=source_postag)
            dep_graph.add_node(target_key, postag=target_postag)
            dep_graph.add_edge(source_key, target_key, dep=target_deprel)

        return dep_graph
