import os
from argparse import Namespace
from collections import Counter, defaultdict

import hydra
import onmt
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.inputters.inputter import (IterOnDevice, _build_fields_vocab,
                                     _load_vocab, get_fields)
from onmt.translate import GNMTGlobalScorer, TranslationBuilder, Translator
from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import set_random_seed

init_logger()
is_cuda = torch.cuda.is_available()
set_random_seed(500, is_cuda)


def build_fields(cfg):
    # initialize the frequency counter
    counters = defaultdict(Counter)

    # load source vocab
    _src_vocab, _src_vocab_size = _load_vocab(
        cfg.src_vocab,
        'src',
        counters)

    # load target vocab
    _tgt_vocab, _tgt_vocab_size = _load_vocab(
        cfg.tgt_vocab,
        'tgt',
        counters)

    # initialize fields
    src_nfeats, tgt_nfeats = 0, 0 # do not support word features for now
    fields = get_fields('text', src_nfeats, tgt_nfeats)

    # build fields vocab
    vocab_fields = _build_fields_vocab(
        fields, counters, 'text', cfg.share_vocab,
        cfg.vocab_size_multiple,
        cfg.src_vocab_size, cfg.src_words_min_frequency,
        cfg.tgt_vocab_size, cfg.tgt_words_min_frequency)

    return vocab_fields


def create_model_loss_optimizer(cfg, vocab_fields):
    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    # set embedding sizes
    enc_emb_size = 500
    dec_emb_size = 500
    if hasattr(cfg, 'src_word_vec_size'):
        enc_emb_size = cfg.src_emb_size
    if hasattr(cfg, 'tgt_word_vec_size'):
        dec_emb_size = cfg.tgt_emb_size
    if hasattr(cfg, 'word_vec_size'):
        enc_emb_size = cfg.word_vec_size
        dec_emb_size = cfg.word_vec_size

    # set decoder sizes
    enc_rnn_size = 500
    dec_rnn_size = 500
    if hasattr(cfg, 'enc_rnn_size'):
        enc_rnn_size = cfg.enc_rnn_size
    if hasattr(cfg, 'dec_rnn_size'):
        dec_rnn_size = cfg.dec_rnn_size
    if hasattr(cfg, 'rnn_size'):
        enc_rnn_size = cfg.rnn_size
        dec_rnn_size = cfg.rnn_size

    # Specify the core model.

    encoder_embeddings = onmt.modules.Embeddings(enc_emb_size, len(src_vocab), word_padding_idx=src_padding)
    encoder = onmt.encoders.RNNEncoder(hidden_size=enc_rnn_size, num_layers=1,
                                       rnn_type="LSTM", bidirectional=True,
                                       embeddings=encoder_embeddings)

    decoder_embeddings = onmt.modules.Embeddings(dec_emb_size, len(tgt_vocab), word_padding_idx=tgt_padding)
    decoder = onmt.decoders.decoder.InputFeedRNNDecoder(hidden_size=dec_rnn_size,
                                                        num_layers=1, bidirectional_encoder=True, 
                                                        rnn_type="LSTM", embeddings=decoder_embeddings)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = onmt.models.model.NMTModel(encoder, decoder)
    model.to(device)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(
        nn.Linear(dec_rnn_size, len(tgt_vocab)),
        nn.LogSoftmax(dim=-1)).to(device)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
        generator=model.generator)

    lr = cfg.learning_rate
    torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optim = onmt.utils.optimizers.Optimizer(
        torch_optimizer, learning_rate=lr, max_grad_norm=cfg.max_grad_norm)

    return model, loss, optim


def create_data_iterators(cfg, vocab_fields):
    src_train = cfg.data.hunglish.path_src
    tgt_train = cfg.data.hunglish.path_tgt
    src_val = cfg.data.valid.path_src
    tgt_val = cfg.data.valid.path_tgt

    # build the ParallelCorpus
    corpus = ParallelCorpus("corpus", src_train, tgt_train)
    valid = ParallelCorpus("valid", src_val, tgt_val)

    # build the training iterator
    train_iter = DynamicDatasetIter(
        corpora={"corpus": corpus},
        corpora_info={"corpus": {"weight": 1}},
        transforms={"transforms": cfg.data.hunglish.transforms},
        fields=vocab_fields,
        is_train=True,
        batch_type="sents",
        batch_size=cfg.batch_size,
        batch_size_multiple=1,
        data_type="text")

    valid_iter = DynamicDatasetIter(
        corpora={"valid": valid},
        corpora_info={"valid": {"weight": 1}},
        transforms={"transforms": cfg.data.valid.transforms},
        fields=vocab_fields,
        is_train=False,
        batch_type="sents",
        batch_size=cfg.valid_batch_size,
        batch_size_multiple=1,
        data_type="text")

    # make sure the iteration happens on GPU 0 (-1 for CPU, N for GPU N)
    train_iter = iter(IterOnDevice(train_iter, 0))
    valid_iter = IterOnDevice(valid_iter, 0)

    return train_iter, valid_iter


@hydra.main(config_name='config')
def main(cfg):
    os.mkdir('run')
    vocab_fields = build_fields(cfg)

    model, loss, optim = create_model_loss_optimizer(cfg, vocab_fields)
    train_iter, valid_iter = create_data_iterators(cfg, vocab_fields)

    report_manager = onmt.utils.ReportMgr(report_every=cfg.report_every, start_time=None, tensorboard_writer=None)

    trainer = onmt.Trainer(model=model,
                           train_loss=loss,
                           valid_loss=loss,
                           optim=optim,
                           report_manager=report_manager,
                           dropout=cfg.dropout)

    trainer.train(train_iter=train_iter,
                  train_steps=cfg.train_steps,
                  valid_iter=valid_iter,
                  valid_steps=cfg.valid_steps)


if __name__ == "__main__":
    main()
