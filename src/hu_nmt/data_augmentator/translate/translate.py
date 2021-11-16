from argparse import Namespace
from typing import List
from unittest.mock import MagicMock

import onmt.opts as opts
import sentencepiece as spm
import torch
from onmt.translate.translator import build_translator
from onmt.utils.logging import init_logger
from onmt.utils.parse import ArgumentParser


def _get_parser():
    parser = ArgumentParser(description='translate.py')
    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


class TranslatorWrapper:
    def __init__(self, opt: Namespace, sp_model_path: str):
        ArgumentParser.validate_translate_opts(opt)
        self.opt = opt
        self.logger = init_logger(opt.log_file)

        self.logger.info(f'Using model: {opt.models}')
        self.logger.info(f'Using sentencepiece model: {sp_model_path}')

        self.spp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.translator = build_translator(opt, logger=self.logger, report_score=True, out_file=MagicMock())

    @staticmethod
    def _batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def translate(self, sentences: List[str]) -> List[str]:
        src_shards = self._batch(sentences, self.opt.batch_size)

        results = []
        for i, src_shard in enumerate(src_shards):
            self.logger.info("Translating shard %d." % i)
            src_encoded_shard = self.spp.encode(src_shard, out_type=str)
            scores, encoded_sentences = self.translator.translate(src=src_encoded_shard, batch_size=self.opt.batch_size, batch_type=self.opt.batch_type)
            results.extend([self.spp.decode(sent_list[0].split(' ')) for sent_list in encoded_sentences])
            
        return results


def get_translator(model_path: str, sp_model_path: str, batch_size: int = 1) -> TranslatorWrapper:
    parser = _get_parser()

    # the '-src' flag is required but we won't be using it
    opt = parser.parse_args([
        '--model', model_path,
        '--src', '',
        '--batch_size', f'{batch_size}',
        '--batch_type', 'sents',
        '--gpu', '0' if torch.cuda.is_available() else '-1'])

    return TranslatorWrapper(opt, sp_model_path)
