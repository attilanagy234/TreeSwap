import yaml
import torch.nn as nn
from collections import defaultdict, Counter
import onmt
from onmt.inputters.inputter import (
    _load_vocab,
    _build_fields_vocab,
    get_fields,
    IterOnDevice,
)
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.utils.misc import set_random_seed
from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts
from transformers import AutoTokenizer
import torch

from onmt.utils.logging import init_logger, logger

from bert.models import BertEncoder
from bert.transforms import BertTransform
import argparse


if __name__ == "__main__":
    parser = ArgumentParser()
    onmt.opts.train_opts(parser)

    opts, unknown = parser.parse_known_args()

    base_args = ["-config", "config.yaml"]  # "../opennmt/experiments-bert/runs/bert/config.yaml"
    opts, unknown = parser.parse_known_args(base_args)

    init_logger(log_file=opts.log_file)

    is_cuda = torch.cuda.is_available()
    set_random_seed(opts.seed, is_cuda)

    # build_vocab_main(opts)

    src_vocab_path = opts.src_vocab
    tgt_vocab_path = opts.tgt_vocab

    counters = defaultdict(Counter)
    eng_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    _src_vocab = [[k, v] for k, v in eng_tokenizer.get_vocab().items()]
    _src_vocab_size = len(_src_vocab)
    for k, _ in _src_vocab:
        counters["src"][k] = 1000  # Set dummy count

    # load target vocab
    _tgt_vocab, _tgt_vocab_size = _load_vocab(tgt_vocab_path, "tgt", counters)

    # initialize fields
    src_nfeats, tgt_nfeats = 0, 0  # do not support word features for now
    fields = get_fields("text", src_nfeats, tgt_nfeats)

    share_vocab = False
    vocab_size_multiple = 1
    src_vocab_size = _src_vocab_size
    tgt_vocab_size = opts.tgt_vocab_size
    src_words_min_frequency = 0
    tgt_words_min_frequency = opts.tgt_words_min_frequency

    vocab_fields = _build_fields_vocab(
        fields,
        counters,
        "text",
        share_vocab,
        vocab_size_multiple,
        src_vocab_size,
        src_words_min_frequency,
        tgt_vocab_size,
        tgt_words_min_frequency,
    )

    # src_text_field = vocab_fields["src"].base_field
    # src_vocab = src_text_field.vocab
    # src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields["tgt"].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    encoder = BertEncoder()
    decoder_embeddings = onmt.modules.Embeddings(
        opts.tgt_word_vec_size, len(tgt_vocab), word_padding_idx=tgt_padding
    )
    decoder = onmt.decoders.transformer.TransformerDecoder.from_opt(
        opts, decoder_embeddings
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = onmt.models.model.NMTModel(encoder, decoder)
    model.to(device)

    logger.info(model)
    model.count_parameters(log=logger.info)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(
        nn.Linear(opts.dec_rnn_size, len(tgt_vocab)), nn.LogSoftmax(dim=-1)
    ).to(device)

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

    bertTransform = BertTransform(opts, src_tokenizer=eng_tokenizer)
    bertTransform.warm_up()

    gpu_rank = 0 if is_cuda else -1  # TODO: read from config

    train_iter = DynamicDatasetIter.from_opts(
        corpora={"corpus": corpus},
        transforms={"berttokenizer": bertTransform},
        fields=vocab_fields,
        opts=opts,
        is_train=True,
    )
    train_iter = iter(IterOnDevice(train_iter, gpu_rank))
    valid_iter = DynamicDatasetIter.from_opts(
        corpora={"valid": valid},
        transforms={"berttokenizer": bertTransform},
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