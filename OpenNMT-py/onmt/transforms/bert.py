from onmt.transforms import Transform, register_transform
from onmt.transforms.tokenize import TokenizerTransform
from transformers import AutoTokenizer


@register_transform(name="encoder_bert_transform")
class EncoderBertTransform(TokenizerTransform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)

        self.src_tokenizer = None

        self.src_seq_length = opts.src_seq_length
        self.tgt_seq_length = opts.tgt_seq_length

    def _parse_opts(self):
        super()._parse_opts()
        self.dropout = {"src": self.src_subword_alpha, "tgt": self.tgt_subword_alpha}

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import random

        random.seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)

        # # Source side
        # from subword_nmt.apply_bpe import BPE, read_vocabulary

        # # Load vocabulary file if provided and set threshold
        # src_vocabulary = None
        # if self.src_subword_vocab != "" and self.src_vocab_threshold > 0:
        #     with open(self.src_subword_vocab, encoding="utf-8") as _sv:
        #         src_vocabulary = read_vocabulary(_sv, self.src_vocab_threshold)

        # # self.src_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # with open(self.src_subword_model, encoding="utf-8") as src_codes:
        #     load_src_model = BPE(codes=src_codes, vocab=src_vocabulary)
        load_src_model = AutoTokenizer.from_pretrained("bert-base-cased")

        # Target side
        import sentencepiece as spm

        load_tgt_model = spm.SentencePieceProcessor()
        load_tgt_model.Load(self.tgt_subword_model)
        if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
            load_tgt_model.LoadVocabulary(
                self.tgt_subword_vocab, self.tgt_vocab_threshold
            )

        # Combined
        self.load_models = {"src": load_src_model, "tgt": load_tgt_model}

    def _tokenize_spm(self, tokens, side="src", is_train=False):
        if side == "tgt":
            sp_model = self.load_models[side]
            sentence = " ".join(tokens)
            if is_train is False or self.tgt_subword_nbest in [0, 1]:
                segmented = sp_model.encode(sentence, out_type=str)
            else:
                segmented = sp_model.encode(
                    sentence,
                    out_type=str,
                    enable_sampling=True,
                    alpha=self.tgt_subword_alpha,
                    nbest_size=self.tgt_subword_nbest,
                )
            return segmented
        else:  # src - bert
            segmented = ["[CLS]", *self.load_models["src"].tokenize(
                tokens, is_split_into_words=True,
            ), "[SEP]"]
            return segmented

    def _tokenize_bpe(self, tokens, side="src", is_train=False):
        """Do bpe subword tokenize."""
        bpe_model = self.load_models[side]
        dropout = self.dropout[side] if is_train else 0.0
        segmented = bpe_model.segment_tokens(tokens, dropout=dropout)
        return segmented

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply sentencepiece subword encode to src & tgt."""
        src_out = self._tokenize_spm(example["src"], "src", is_train)
        tgt_out = self._tokenize_spm(example["tgt"], "tgt", is_train)
        # if stats is not None:
        #     n_words = len(example["src"]) + len(example["tgt"])
        #     n_subwords = len(src_out) + len(tgt_out)
        #     stats.subword(n_subwords, n_words)
        example["src"], example["tgt"] = src_out, tgt_out
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return "{}={}, {}={}".format(
            "src_seq_length", self.src_seq_length, "tgt_seq_length", self.tgt_seq_length
        )
