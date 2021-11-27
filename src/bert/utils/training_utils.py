import argparse
from collections import Counter, defaultdict
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


def setup_vocab(opts):
    src_vocab_path = opts.src_vocab
    tgt_vocab_path = opts.tgt_vocab

    from collections import defaultdict
    counters = defaultdict(Counter)

    if opts.custom_encoder_type == "transformer":
        # load source vocab
        _src_vocab, src_vocab_size = _load_vocab(src_vocab_path, "src", counters)

        src_words_min_frequency = opts.src_words_min_frequency
    elif opts.custom_encoder_type == "custom_bert":
        eng_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        _src_vocab = [[k, v] for k, v in sorted(eng_tokenizer.get_vocab().items(), key=lambda x: x[1])]
        src_vocab_size = len(_src_vocab)
        for k, v in _src_vocab:
            counters["src"][k] = src_vocab_size - v

        src_words_min_frequency = 0
    else:
        raise ArgumentError(f"Unknown encoder type: {opts.custom_encoder_type}")

    if opts.custom_decoder_type == "transformer":
        # load target vocab
        _tgt_vocab, tgt_vocab_size = _load_vocab(tgt_vocab_path, "tgt", counters)

        tgt_words_min_frequency = opts.tgt_words_min_frequency
    elif opts.custom_decoder_type == "custom_bert":
        raise NotImplementedError("Tokenizer for BERT decoder not implemented yet.")
    else:
        raise ArgumentError(f"Unknown encoder type: {opts.custom_decoder_type}")

    # initialize fields
    src_nfeats, tgt_nfeats = 0, 0  # do not support word features for now
    fields = get_fields("text", src_nfeats, tgt_nfeats)

    share_vocab = opts.share_vocab
    vocab_size_multiple = 1

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

    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab

    if opts.custom_encoder_type == "custom_bert":
        src_text_field.pad_token = "[PAD]"
        src_text_field.unk_token = "[UNK]"
        src_vocab.UNK = "[UNK]"

        from collections import defaultdict
        src_vocab.stoi = defaultdict(lambda: 100)
        for k, v in _src_vocab:
            src_vocab.stoi[k] = v
        src_vocab.itos = [k for k, _ in _src_vocab]

        src_vocab.unk_index = src_vocab.stoi["[UNK]"]

    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields["tgt"].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    return src_vocab, src_padding, tgt_vocab, tgt_padding, vocab_fields


def create_encoder(opts, src_vocab, src_padding):
    encoder = None

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

    return encoder

def create_decoder(opts, tgt_vocab, tgt_padding):
    decoder = None

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

    return decoder


def create_model(opts):
    src_vocab, src_padding, tgt_vocab, tgt_padding, vocab_fields = setup_vocab(opts)

    encoder = create_encoder(opts, src_vocab, src_padding)
    decoder = create_decoder(opts, tgt_vocab, tgt_padding)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = onmt.models.model.NMTModel(encoder, decoder)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(
        nn.Linear(opts.dec_rnn_size, len(tgt_vocab)), nn.LogSoftmax(dim=-1)
    )

    model.to(device)

    return model, vocab_fields

def load_checkpoint(checkpoint, model, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optim"])
