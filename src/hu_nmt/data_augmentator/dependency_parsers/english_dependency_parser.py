import stanza
import pandas as pd
import networkx as nx
from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase


class EnglishDependencyParser(DependencyParserBase):

    def __init__(self):
        super().__init__()
        self.nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

    def sentence_to_dep_parse_tree(self, sent):
        """
        Args:
            sent: space separated string of the input sentence
        Returns:
            A directed (networkx) graph representation of the dependency tree
        """
        doc = self.nlp_pipeline(sent)
        # We most likely will only pass single sentences.
        for sentence in doc.sentences:
            words_dict = [word.to_dict() for word in sentence.words]
            df = pd.DataFrame.from_records(words_dict)
        dep_graph = nx.DiGraph()
        # Add ROOT node
        root_key = 'root-0'
        dep_graph.add_node(root_key)
        for row in df.itertuples(index=True, name='Pandas'):
            target_key = f'{row.text.lower()}-{row.id}'
            target_postag = row.upos
            target_deprel = row.deprel
            if row.head != 0:
                head = df.iloc[int(row.head) - 1]
                source_key = f'{head.text.lower()}-{head.id}'
                source_postag = head.upos
            else:
                source_key = root_key
                source_postag = None
            dep_graph.add_node(source_key, postag=source_postag)
            dep_graph.add_node(target_key, postag=target_postag)
            dep_graph.add_edge(source_key, target_key, dep=target_deprel)
        return dep_graph
