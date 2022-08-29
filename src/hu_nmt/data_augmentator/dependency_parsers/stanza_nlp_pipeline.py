import os
import pathlib
from functools import partial
from typing import List

import networkx as nx
import stanza
from stanza.pipeline.core import DownloadMethod

from hu_nmt.data_augmentator.base.nlp_pipeline_base import NlpPipelineBase, NodeRelationship, \
    SentenceProcessUnit, SentenceProcessBatch
from hu_nmt.data_augmentator.utils.logger import get_logger

log = get_logger(__name__)

ROOT_KEY = 'root_0'


class StanzaNlpPipeline(NlpPipelineBase):
    def __init__(self, lang, processors):
        nlp_pipeline_constructor = partial(stanza.Pipeline, lang=lang, processors=processors,
                                           download_method=DownloadMethod.REUSE_RESOURCES)
        if not pathlib.Path(f'{os.getenv("HOME")}/stanza_resources/{lang}').resolve().exists():
            stanza.download(lang=lang, processors=processors)

        use_multiprocessing = False
        if os.getenv('USE_MULTIPROCESSING', False) == 'True':
            use_multiprocessing = True

        super().__init__(nlp_pipeline_constructor, use_multiprocessing=use_multiprocessing)

    @staticmethod
    def sentence_to_node_relationship_list(nlp_pipeline, sent: str) -> List[NodeRelationship]:
        doc = nlp_pipeline(sent)
        # We most likely will only pass single sentences.
        if len(doc.sentences) != 1:
            log.debug(f'Sample has multiple sentences: {[s.text for s in doc.sentences]}')
            return []
        sent = doc.sentences[0]

        node_relationship_list = []
        word_dicts = [word.to_dict() for word in sent.words]
        for word in sent.words:
            token = word.to_dict()
            target_key = f'{token["text"]}_{token["id"]}'
            target_postag = token['upos']
            target_lemma = token.get('lemma', target_key)
            target_deprel = token['deprel']
            if token['head'] == 0:
                source_key = ROOT_KEY
                source_postag = None
                source_lemma = None
            else:
                head = word_dicts[int(token['head']) - 1]
                source_key = f'{head["text"]}_{head["id"]}'
                source_postag = head['upos']
                source_lemma = head.get('lemma', source_key)

            node_relationship_list.append(NodeRelationship(target_key, target_postag, target_lemma, target_deprel,
                                                           source_key, source_postag, source_lemma))

        return node_relationship_list

    def node_relationship_list_to_dep_parse_tree(self, dep_rel_list: List[NodeRelationship]) -> nx.DiGraph:
        dep_graph = nx.DiGraph()
        # Add ROOT node
        dep_graph.add_node(ROOT_KEY, postag='root')
        for node_rel in dep_rel_list:
            dep_graph.add_node(node_rel.source_key, postag=node_rel.source_postag, lemma=node_rel.source_lemma)
            dep_graph.add_node(node_rel.target_key, postag=node_rel.target_postag, lemma=node_rel.target_lemma)
            dep_graph.add_edge(node_rel.source_key, node_rel.target_key, dep=node_rel.target_deprel)
        return dep_graph

    def count_sentences(self, doc) -> int:
        return len(doc.sentences)

    #def count_tokens(self, doc) -> int:
    #    return len([token for sent in doc.sentences for token in sent.words if token.pos != 'PUNCT'])

    def count_tokens(self, graph) -> int:
        return len([node for node, data in graph.nodes(data=True) if data['postag'] != 'PUNCT']) - 1

    @staticmethod
    def _sentence_process_unit_to_node_relationship_list(process_unit: SentenceProcessUnit) -> List[NodeRelationship]:
        return StanzaNlpPipeline.sentence_to_node_relationship_list(process_unit.pipeline, process_unit.sentence)

    @staticmethod
    def _sentence_process_batch_to_node_relationship_list(process_batch: SentenceProcessBatch) \
            -> List[List[NodeRelationship]]:
        log.info('Creating pipeline in process')
        pipeline = process_batch.pipeline_constructor()
        log.info('Processing sentences in process')
        return [StanzaNlpPipeline.sentence_to_node_relationship_list(pipeline, sentence)
                for sentence in process_batch.sentences]
