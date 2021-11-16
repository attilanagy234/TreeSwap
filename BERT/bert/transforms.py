from onmt.transforms import Transform, register_transform
from onmt.transforms.tokenize import TokenizerTransform
from transformers import AutoTokenizer


# @register_transform(name='berttokenizer')
class _EncoderBertTransform(TokenizerTransform):
    """SentencePiece subword transform class."""

    def __init__(self, opts):
        """Initialize necessary options for sentencepiece."""
        super().__init__(opts)

        self.src_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased") # src_tokenizer

    def _set_seed(self, seed):
        """set seed to ensure reproducibility."""
        import sentencepiece as spm
        spm.set_random_generator_seed(seed)

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        import sentencepiece as spm

        load_tgt_model = spm.SentencePieceProcessor()
        load_tgt_model.Load(self.tgt_subword_model)
        if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
            load_tgt_model.LoadVocabulary(
                self.tgt_subword_vocab, self.tgt_vocab_threshold)
        self.load_models = {
            'src': self.src_tokenizer,
            'tgt': load_tgt_model
        }

    def _tokenize(self, tokens, side='src', is_train=False):
        if side == "tgt":
            sp_model = self.load_models[side]
            sentence = ' '.join(tokens)
            if is_train is False or self.tgt_subword_nbest in [0, 1]:
                segmented = sp_model.encode(sentence, out_type=str)
            else:
                segmented = sp_model.encode(
                    sentence, out_type=str, enable_sampling=True,
                    alpha=self.tgt_subword_alpha, nbest_size=self.tgt_subword_nbest)
            return segmented
        else:  # src - bert
            segmented = self.load_models['src'].tokenize(tokens, is_split_into_words=True)
            return segmented

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply sentencepiece subword encode to src & tgt."""
        src_out = self._tokenize(example['src'], 'src', is_train)
        tgt_out = self._tokenize(example['tgt'], 'tgt', is_train)
        if stats is not None:
            n_words = len(example['src']) + len(example['tgt'])
            n_subwords = len(src_out) + len(tgt_out)
            stats.subword(n_subwords, n_words)
        example['src'], example['tgt'] = src_out, tgt_out
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        kwargs_str = super()._repr_args()
        additional_str = 'src_subword_nbest={}, tgt_subword_nbest={}'.format(
            self.src_subword_nbest, self.tgt_subword_nbest
        )
        return kwargs_str + ', ' + additional_str


@register_transform(name='berttokenizer')
class EncoderBertTransform(TokenizerTransform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)

        self.src_tokenizer = None

        self.src_seq_length = opts.src_seq_length
        self.tgt_seq_length = opts.tgt_seq_length

    def warm_up(self, vocabs=None):
        """Load subword models."""
        super().warm_up(None)
        import sentencepiece as spm

        load_tgt_model = spm.SentencePieceProcessor()
        load_tgt_model.Load(self.tgt_subword_model)
        if self.tgt_subword_vocab != "" and self.tgt_vocab_threshold > 0:
            load_tgt_model.LoadVocabulary(
                self.tgt_subword_vocab, self.tgt_vocab_threshold)

        self.src_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        self.load_models = {
            'src': self.src_tokenizer,
            'tgt': load_tgt_model
        }

    def _tokenize(self, tokens, side='src', is_train=False):
        if side == "tgt":
            sp_model = self.load_models[side]
            sentence = ' '.join(tokens)
            if is_train is False or self.tgt_subword_nbest in [0, 1]:
                segmented = sp_model.encode(sentence, out_type=str)
            else:
                segmented = sp_model.encode(
                    sentence, out_type=str, enable_sampling=True,
                    alpha=self.tgt_subword_alpha, nbest_size=self.tgt_subword_nbest)
            return segmented
        else:  # src - bert
            segmented = self.load_models['src'].tokenize(tokens, is_split_into_words=True)
            return segmented

    def apply(self, example, is_train=False, stats=None, **kwargs):
        """Apply sentencepiece subword encode to src & tgt."""
        src_out = self._tokenize(example['src'], 'src', is_train)
        tgt_out = self._tokenize(example['tgt'], 'tgt', is_train)
        if stats is not None:
            n_words = len(example['src']) + len(example['tgt'])
            n_subwords = len(src_out) + len(tgt_out)
            stats.subword(n_subwords, n_words)
        example['src'], example['tgt'] = src_out, tgt_out
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format(
            'src_seq_length', self.src_seq_length,
            'tgt_seq_length', self.tgt_seq_length
        )

@register_transform(name='dummytransform')
class DummyTransform(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)

        self.src_seq_length = opts.src_seq_length
        self.tgt_seq_length = opts.tgt_seq_length

    def apply(self, example, is_train=False, stats=None, **kwargs):
        print("Dummy")
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format(
            'src_seq_length', self.src_seq_length,
            'tgt_seq_length', self.tgt_seq_length
        )
