import hu_core_ud_lg
import networkx as nx
import spacy
from tqdm import tqdm
from hu_nmt.data_augmentator.base.depedency_parser_base import DependencyParserBase
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)

ROOT_KEY = 'root_0'


class SpacyDependencyParser(DependencyParserBase):
    def __init__(self, lang):
        super().__init__()
        if lang == 'hu':
            self.nlp_pipeline = hu_core_ud_lg.load()
        elif lang == 'de':
            self.nlp_pipeline = spacy.load("de_core_news_sm")
        elif lang == 'fr':
            self.nlp_pipeline = spacy.load("fr_core_news_sm")
        else:
            raise ValueError(f'Language {lang} is not supported by the SpacyDependencyParser.')

    def sentence_to_dep_parse_tree(self, sent):
        """
        Args:
            sent: space separated string of the input sentence
        Returns:
            A directed (networkx) graph representation of the dependency tree
        """
        doc = self.nlp_pipeline(sent)
        dep_graph = nx.DiGraph()
        for sent in doc.sents:
            for token in sent:
                target_key = f'{token}_{token.i+1}'
                target_postag = token.pos_
                target_lemma = token.lemma_
                target_deprel = token.dep_
                if target_deprel == 'ROOT':
                    source_key = ROOT_KEY
                    source_postag = None
                    source_lemma = None
                else:
                    source_key = f'{token.head}_{token.head.i+1}'
                    source_postag = token.head.pos_
                    source_lemma = token.head.lemma_

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
            for sent in doc.sents:
                for token in sent:
                    target_key = f'{token}_{token.i + 1}'
                    target_postag = token.pos_
                    target_lemma = token.lemma_
                    target_deprel = token.dep_
                    if target_deprel == 'ROOT':
                        source_key = ROOT_KEY
                        source_postag = None
                        source_lemma = None
                    else:
                        source_key = f'{token.head}_{token.head.i + 1}'
                        source_postag = token.head.pos_
                        source_lemma = token.head.lemma_

                    graph_record = f'{target_key}\t{target_postag}\t{target_lemma}' \
                                   f'\t{target_deprel}\t{source_key}\t{source_postag}\t{source_lemma}\n'
                    file.write(graph_record)

            file.write('\n')  # Separate sentences with a new line

            if (progress_idx + 1) % file_batch_size == 0:
                file.close()
                file_idx += 1
                open_new_file = True
