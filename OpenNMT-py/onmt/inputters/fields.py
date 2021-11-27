"""Module for build dynamic fields."""
from collections import Counter, defaultdict
import torch
from onmt.utils.logging import logger
from onmt.utils.misc import check_path
from onmt.inputters.inputter import get_fields, _load_vocab, \
    _build_fields_vocab
from transformers import AutoTokenizer
from configargparse import ArgumentError


def _get_dynamic_fields(opts):
    # NOTE: not support tgt feats yet
    tgt_feats = None
    with_align = hasattr(opts, 'lambda_align') and opts.lambda_align > 0.0
    fields = get_fields('text', opts.src_feats_vocab, tgt_feats,
                        dynamic_dict=opts.copy_attn,
                        src_truncate=opts.src_seq_length_trunc,
                        tgt_truncate=opts.tgt_seq_length_trunc,
                        with_align=with_align,
                        data_task=opts.data_task)

    return fields


def build_dynamic_fields(opts, src_specials=None, tgt_specials=None):
    src_vocab_path = opts.src_vocab
    tgt_vocab_path = opts.tgt_vocab

    from collections import defaultdict
    counters = defaultdict(Counter)

    if opts.encoder_type == "transformer":
        # load source vocab
        _src_vocab, src_vocab_size = _load_vocab(src_vocab_path, "src", counters)

        src_words_min_frequency = opts.src_words_min_frequency
    elif opts.encoder_type == "bert":
        eng_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        _src_vocab = [[k, v] for k, v in sorted(eng_tokenizer.get_vocab().items(), key=lambda x: x[1])]
        src_vocab_size = len(_src_vocab)
        for k, v in _src_vocab:
            counters["src"][k] = src_vocab_size - v

        src_words_min_frequency = 0
    else:
        raise ArgumentError(f"Unknown encoder type: {opts.encoder_type}")

    if opts.decoder_type == "transformer":
        # load target vocab
        _tgt_vocab, tgt_vocab_size = _load_vocab(tgt_vocab_path, "tgt", counters)

        tgt_words_min_frequency = opts.tgt_words_min_frequency
    elif opts.decoder_type == "bert":
        raise NotImplementedError("Tokenizer for BERT decoder not implemented yet.")
    else:
        raise ArgumentError(f"Unknown encoder type: {opts.decoder_type}")

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

    if opts.encoder_type == "bert":
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

    #return src_vocab, src_padding, tgt_vocab, tgt_padding, vocab_fields
    return vocab_fields

    # """Build fields for dynamic, including load & build vocab."""
    # fields = _get_dynamic_fields(opts)

    # counters = defaultdict(Counter)
    # logger.info("Loading vocab from text file...")

    # _src_vocab, _src_vocab_size = _load_vocab(
    #     opts.src_vocab, 'src', counters,
    #     min_freq=opts.src_words_min_frequency)

    # if opts.src_feats_vocab:
    #     for feat_name, filepath in opts.src_feats_vocab.items():
    #         _, _ = _load_vocab(
    #             filepath, feat_name, counters,
    #             min_freq=0)

    # if opts.tgt_vocab:
    #     _tgt_vocab, _tgt_vocab_size = _load_vocab(
    #         opts.tgt_vocab, 'tgt', counters,
    #         min_freq=opts.tgt_words_min_frequency)
    # elif opts.share_vocab:
    #     logger.info("Sharing src vocab to tgt...")
    #     counters['tgt'] = counters['src']
    # else:
    #     raise ValueError("-tgt_vocab should be specified if not share_vocab.")

    # logger.info("Building fields with vocab in counters...")
    # fields = _build_fields_vocab(
    #     fields, counters, 'text', opts.share_vocab,
    #     opts.vocab_size_multiple,
    #     opts.src_vocab_size, opts.src_words_min_frequency,
    #     opts.tgt_vocab_size, opts.tgt_words_min_frequency,
    #     src_specials=src_specials, tgt_specials=tgt_specials)

    return fields


def get_vocabs(fields):
    """Get a dict contain src & tgt vocab extracted from fields."""
    src_vocab = fields['src'].base_field.vocab
    tgt_vocab = fields['tgt'].base_field.vocab
    vocabs = {'src': src_vocab, 'tgt': tgt_vocab}
    return vocabs


def save_fields(fields, save_data, overwrite=True):
    """Dump `fields` object."""
    fields_path = "{}.vocab.pt".format(save_data)
    check_path(fields_path, exist_ok=overwrite, log=logger.warning)
    logger.info(f"Saving fields to {fields_path}...")
    torch.save(fields, fields_path)


def load_fields(save_data, checkpoint=None):
    """Load dumped fields object from `save_data` or `checkpoint` if any."""
    if checkpoint is not None:
        logger.info("Loading fields from checkpoint...")
        fields = checkpoint['vocab']
    else:
        fields_path = "{}.vocab.pt".format(save_data)
        logger.info(f"Loading fields from {fields_path}...")
        fields = torch.load(fields_path)
    return fields
