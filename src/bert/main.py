import argparse
from collections import Counter, defaultdict
from typing import DefaultDict

import onmt
import sentencepiece
import torch
import torch.nn as nn
import yaml
from BERT.utils.training import setup_vocab
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
from utils.training_utils import setup_vocab

if __name__ == "__main__":
    parser = ArgumentParser()
    onmt.opts.train_opts(parser)

    group = parser.add_argument_group('Custom')
    group.add('--encoder_freezed', type=bool, default=False)
    group.add('--custom_encoder_type', type=str)
    group.add('--custom_decoder_type', type=str)

    base_args = [
        "-config",
        "config.yaml",
    ]  # "../opennmt/experiments-bert/runs/bert/config.yaml"
    opts, unknown = parser.parse_known_args(base_args)

    init_logger(log_file=opts.log_file)

    logger.info("opts:", opts)

    is_cuda = torch.cuda.is_available()
    set_random_seed(opts.seed, is_cuda)

    # build_vocab_main(opts)

    src_vocab, src_padding, tgt_vocab, tgt_padding = setup_vocab(opts)

    if opts.custom_encoder_type == "transformer":
        encoder_embeddings = onmt.modules.Embeddings(
            opts.src_word_vec_size, len(src_vocab), word_padding_idx=src_padding
        )
        encoder = TransformerEncoder.from_opt(
            opts, encoder_embeddings
        )
    elif opts.custom_encoder_type == "custom_bert":
        encoder = BertEncoder.from_opt(opts)
    else:
        raise ArgumentError(f"Unknown encoder type: {opts.custom_encoder_type}")

    if opts.decoder_type == "transformer":
        decoder_embeddings = onmt.modules.Embeddings(
            opts.tgt_word_vec_size, len(tgt_vocab), word_padding_idx=tgt_padding
        )
        decoder = onmt.decoders.transformer.TransformerDecoder.from_opt(
            opts, decoder_embeddings
        )
    elif opts.decoder_type == "custom_bert":
        raise NotImplementedError(f"Not implemented decoder type: {opts.decoder_type}")
    else:
        raise ArgumentError(f"Unknown decoder type: {opts.decoder_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = onmt.models.model.NMTModel(encoder, decoder)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(
        nn.Linear(opts.dec_rnn_size, len(tgt_vocab)), nn.LogSoftmax(dim=-1)
    ).to(device)

    # TODO: load model

    model.to(device)

    logger.info(model)
    model.count_parameters(log=logger.info)

    # loss = onmt.utils.loss.NMTLossCompute(
    #     criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
    #     generator=model.generator)

    optimizer = onmt.utils.optimizers.Optimizer.from_opt(model, opts)

    data_config = yaml.load(opts.data)
    opts.data = data_config
    src_train = data_config["corpus"]["path_src"]
    tgt_train = data_config["corpus"]["path_tgt"]
    src_val = data_config["valid"]["path_src"]
    tgt_val = data_config["valid"]["path_tgt"]

    # build the ParallelCorpus
    corpus = ParallelCorpus("corpus", src_train, tgt_train)
    valid = ParallelCorpus("valid", src_val, tgt_val)

    encoder_bert_transform = EncoderBertTransform(opts)
    encoder_bert_transform.warm_up()

    dummyTransform = DummyTransform(opts)
    dummyTransform.warm_up()

    spt = SentencePieceTransform(opts)
    spt.warm_up()

    gpu_rank = 0 if is_cuda else -1  # TODO: read from config

    # if opts.custom_encoder_type == "transformer":
    #     import sentencepiece
    #     dataset_transforms = {"sentencepiece": sentencepiece}
    # elif opts.custom_encoder_type == "custom_bert":
    #     dataset_transforms = {"sentencepiece": bertTransform}
    # else:
    #     raise ArgumentError(f"Unknown encoder type: {opts.custom_encoder_type}")

    onmt.transforms.register_transform(EncoderBertTransform)

    dataset_transforms = {
        "sentencepiece": spt,
        "encoderberttransform": encoder_bert_transform,
        "dummy": dummyTransform
    }

    train_iter = DynamicDatasetIter.from_opts(
        corpora={"corpus": corpus},
        transforms=dataset_transforms,
        fields=vocab_fields,
        opts=opts,
        is_train=True,
    )
    train_iter = iter(IterOnDevice(train_iter, gpu_rank))
    valid_iter = DynamicDatasetIter.from_opts(
        corpora={"valid": valid},
        transforms=dataset_transforms,
        fields=vocab_fields,
        opts=opts,
        is_train=False,
    )
    valid_iter = IterOnDevice(valid_iter, gpu_rank)

    report_manager = onmt.utils.build_report_manager(opts, gpu_rank)
    model_saver = onmt.models.build_model_saver(
        model_opt=opts, opt=opts, model=model, fields=fields, optim=optimizer
    )

    trainer = onmt.trainer.build_trainer(
        opts,
        device_id=gpu_rank,
        model=model,
        fields=fields,
        optim=optimizer,
        model_saver=model_saver,
    )

    trainer.train(
        train_iter=train_iter,
        train_steps=opts.train_steps,
        valid_iter=valid_iter,
        valid_steps=opts.valid_steps,
        save_checkpoint_steps=opts.save_checkpoint_steps,
    )
