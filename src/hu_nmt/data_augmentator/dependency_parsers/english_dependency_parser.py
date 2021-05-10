import stanza
import pandas as pd
import networkx as nx
from tqdm import tqdm
from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)

ROOT_KEY = 'root_0'


class EnglishDependencyParser(DependencyParserBase):

    def __init__(self):
        super().__init__()
        self.nlp_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

    @staticmethod
    def extract_info_from_row(row, df):
        target_key = f'{row.text.lower()}_{row.id}'
        target_postag = row.upos
        target_lemma = row.lemma
        target_deprel = row.deprel
        if row.head != 0:
            head = df.iloc[int(row.head) - 1]
            source_key = f'{head.text.lower()}_{head.id}'
            source_postag = head.upos
            source_lemma = head.lemma
        else:
            source_key = ROOT_KEY
            source_postag = None
            source_lemma = None
        return target_key, target_postag, target_lemma, target_deprel, source_key, source_postag, source_lemma

    def sentence_to_dep_parse_tree(self, sent):
        """
        Args:
            sent: space separated string of the input sentence
        Returns:
            A directed (networkx) graph representation of the dependency tree
        """
        doc = self.nlp_pipeline(sent)
        # We most likely will only pass single sentences.
        if len(doc.sentences) != 1:
            log.info(f'Sample has multiple sentences: {doc.sentences}')
        for sentence in doc.sentences:
            words_dict = [word.to_dict() for word in sentence.words]
            df = pd.DataFrame.from_records(words_dict)
        dep_graph = nx.DiGraph()
        # Add ROOT node
        dep_graph.add_node(ROOT_KEY)
        for row in df.itertuples(index=True, name='Pandas'):
            target_key, target_postag, target_lemma, target_deprel, \
            source_key, source_postag, source_lemma = self.extract_info_from_row(row, df)
            dep_graph.add_node(source_key, postag=source_postag, lemma=source_lemma)
            dep_graph.add_node(target_key, postag=target_postag, lemma=target_lemma)
            dep_graph.add_edge(source_key, target_key, dep=target_deprel)
        return dep_graph

    def sentences_to_serialized_dep_graph_files(self, sentences, output_dir, file_batch_size):
        """
        Args:
            sentences: list of sentences to process
            output_dir: location of tsv files containing the dep parsed sentences
            file_batch_size: amount of sentences to be parsed into a single file
        """
        file_idx = 1
        open_new_file = True
        file_batch_size = int(file_batch_size)

        for progress_idx, record in tqdm(enumerate(sentences)):
            if open_new_file:
                file = open(f'{output_dir}/{file_idx}.tsv', 'w+')
                open_new_file = False

            doc = self.nlp_pipeline(record)
            for sent in doc.sentences:
                words_dict = [word.to_dict() for word in sent.words]
                df = pd.DataFrame.from_records(words_dict)
            for row in df.itertuples(index=True, name='Pandas'):
                target_key, target_postag, target_lemma, target_deprel, \
                source_key, source_postag, source_lemma = self.extract_info_from_row(row, df)

                graph_record = f'{target_key}\t{target_postag}\t{target_lemma}' \
                               f'\t{target_deprel}\t{source_key}\t{source_postag}\t{source_lemma}\n'
                file.write(graph_record)

            file.write('\n')  # Separate sentences with a new line

            if (progress_idx + 1) % file_batch_size == 0:
                file.close()
                file_idx += 1
                open_new_file = True
