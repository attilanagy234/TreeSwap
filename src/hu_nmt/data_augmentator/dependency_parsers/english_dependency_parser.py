import stanza
import pandas as pd
import networkx as nx
from tqdm import tqdm

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
            target_lemma = row.lemma
            # TODO: add lemma as node attribute
            target_deprel = row.deprel
            if row.head != 0:
                head = df.iloc[int(row.head) - 1]
                source_key = f'{head.text.lower()}-{head.id}'
                source_postag = head.upos
                source_lemma = head.lemma
            else:
                source_key = root_key
                source_postag = None
                source_lemma = None
            dep_graph.add_node(source_key, postag=source_postag, lemma=source_lemma)
            dep_graph.add_node(target_key, postag=target_postag, lemma=target_lemma)
            dep_graph.add_edge(source_key, target_key, dep=target_deprel)
        return dep_graph

    # def sentences_to_serialized_dep_graph(self, sentences, output_dir):
    #     """
    #     Args:
    #         sentences: list of sentences to process
    #         output_dir: location of tsv files containing the dep parsed sentences
    #     """
    #     file_idx = 0
    #     open_new_file = True
    #
    #     for progress_idx, record in tqdm(enumerate(sentences)):
    #         if open_new_file:
    #             file = open(f'{output_dir}/{file_idx}.tsv', 'w')
    #             open_new_file = False
    #
    #         doc = self.nlp_pipeline(record)
    #         for sent in doc.sentences:
    #             words_dict = [word.to_dict() for word in sent.words]
    #             df = pd.DataFrame.from_records(words_dict)
    #         # Add ROOT node
    #         root_key = 'root-0'
    #         for row in df.itertuples(index=True, name='Pandas'):
    #             target_key = f'{row.text.lower()}-{row.id}'
    #             target_postag = row.upos
    #             # TODO: add lemma as node attribute
    #             target_deprel = row.deprel
    #             if row.head != 0:
    #                 head = df.iloc[int(row.head) - 1]
    #                 source_key = f'{head.text.lower()}-{head.id}'
    #                 source_postag = head.upos
    #             else:
    #                 source_key = root_key
    #                 source_postag = None
    #
    #
    #             graph_record = ''
    #             file.write(graph_record)
    #
    #         if progress_idx % 50000 == 0:
    #             file.close()
    #             file_idx += 1
    #             open_new_file = True
    #
    #
    #             dep_graph.add_node(source_key, postag=source_postag)
    #             dep_graph.add_node(target_key, postag=target_postag)
    #             dep_graph.add_edge(source_key, target_key, dep=target_deprel)