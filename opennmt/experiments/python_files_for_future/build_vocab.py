import os

import hydra
import onmt
import sentencepiece as spm
import torch
from omegaconf import DictConfig
from onmt.bin.build_vocab import build_vocab_main
from onmt.opts import dynamic_prepare_opts
from onmt.utils.logging import init_logger
from onmt.utils.misc import set_random_seed
from onmt.utils.parse import ArgumentParser


def create_sentencepiece_models(cfg: DictConfig):
    src_spm_model_name = cfg.src_subword_model[:-len('.model')]
    tgt_spm_model_name = cfg.tgt_subword_model[:-len('.model')]
    spm.SentencePieceTrainer.train(input=cfg.data.hunglish.path_src, model_prefix=src_spm_model_name, vocab_size=cfg.src_vocab_size, model_type='bpe', character_coverage=1.0)
    spm.SentencePieceTrainer.train(input=cfg.data.hunglish.path_tgt, model_prefix=tgt_spm_model_name, vocab_size=cfg.tgt_vocab_size, model_type='bpe', character_coverage=1.0)


def create_vocabs(cfg: DictConfig):
    parser = ArgumentParser(description='build_vocab.py')
    dynamic_prepare_opts(parser, build_vocab_only=True)
    base_args = (["-config", ".hydra/config.yaml", "-n_sample", "-1"])
    opts, unknown = parser.parse_known_args(base_args)

    build_vocab_main(opts)


@hydra.main(config_name='config')
def main(cfg: DictConfig):
    init_logger()

    is_cuda = torch.cuda.is_available()
    set_random_seed(500, is_cuda)

    create_sentencepiece_models(cfg)
    create_vocabs(cfg)


if __name__ == '__main__':
    main()
