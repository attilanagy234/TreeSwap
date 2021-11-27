import argparse
from typing import DefaultDict

import onmt
import sentencepiece
import torch
import torch.nn as nn
import yaml
from bert.models import BertEncoder
from bert.transforms import EncoderBertTransform, DummyTransform
from configargparse import ArgumentError
from onmt.encoders.transformer import TransformerEncoder
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.inputters.inputter import (IterOnDevice, _build_fields_vocab,
                                     _load_vocab, get_fields)
from onmt.opts import dynamic_prepare_opts
from onmt.transforms.tokenize import SentencePieceTransform
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import set_random_seed
from onmt.utils.parse import ArgumentParser
from transformers import AutoTokenizer
from utils.training_utils import create_model, load_checkpoint

if __name__ == "__main__":
    parser = ArgumentParser()
    onmt.opts.train_opts(parser)

    group = parser.add_argument_group('Custom')
    group.add('--encoder_freezed', type=bool, default=False)
    group.add('--custom_encoder_type', type=str)
    group.add('--custom_decoder_type', type=str)
    group.add('--translate_model', type=str)
    group.add('--output_file', type=str)

    base_args = [
        "-config",
        "config.yaml",
    ]  # "../opennmt/experiments-bert/runs/bert/config.yaml"
    opts, unknown = parser.parse_known_args(base_args)

    init_logger(log_file=opts.log_file)

    logger.info("opts:", opts)

    is_cuda = torch.cuda.is_available()
    set_random_seed(opts.seed, is_cuda)

    model, vocab_fields = create_model(opts)

    logger.info(model)
    model.count_parameters(log=logger.info)

    optimizer = onmt.utils.optimizers.Optimizer.from_opt(model, opts)

    load_checkpoint(opts.translate_model, model, optimizer)

    data_config = yaml.load(opts.data)
    opts.data = data_config
    src_test = data_config["test"]["path_src"]
    src_test = data_config["test"]["path_tgt"]

    # build the ParallelCorpus
    test = ParallelCorpus("test", src_test, src_test)

    encoder_bert_transform = EncoderBertTransform(opts)
    encoder_bert_transform.warm_up()

    spt = SentencePieceTransform(opts)
    spt.warm_up()

    onmt.transforms.register_transform(EncoderBertTransform)

    dataset_transforms = {
        "sentencepiece": spt,
        "encoderberttransform": encoder_bert_transform,
    }

    gpu_rank = 0 if is_cuda else -1  # TODO: read from config

    test_iter = DynamicDatasetIter.from_opts(
        corpora={"test": test},
        transforms=dataset_transforms,
        fields=vocab_fields,
        opts=opts,
        is_train=False,
    )
    test_iter = IterOnDevice(test_iter, gpu_rank)

    # TODO: predict
    inference = onmt.translate.translator.Inference.from_opt(
        model=model,
        opt=opts,
        model_opt=opts,
        fields=vocab_fields,
        out_file=opts.output_file
    )

    

