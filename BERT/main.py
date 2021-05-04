import yaml
import torch
import torch.nn as nn
from argparse import Namespace
from collections import defaultdict, Counter
import onmt
from onmt.inputters.inputter import _load_vocab, _build_fields_vocab, get_fields, IterOnDevice
from onmt.inputters.corpus import ParallelCorpus
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
from onmt.transforms import register_transform
from onmt.transforms.tokenize import TokenizerTransform
from onmt.translate import GNMTGlobalScorer, Translator, TranslationBuilder
from onmt.utils.misc import set_random_seed
from onmt.utils.parse import ArgumentParser
from onmt.opts import dynamic_prepare_opts
from onmt.bin.build_vocab import build_vocab_main
from transformers import AutoModel, AutoTokenizer
import torch

from onmt.utils.logging import init_logger, logger
init_logger()

if __name__ == "__main__":
    is_cuda = torch.cuda.is_available()
    set_random_seed(1111, is_cuda)

    # with open("config.yaml", "r") as f:
    #     config = yaml.safe_load(f)

    parser = ArgumentParser()
    dynamic_prepare_opts(parser, build_vocab_only=True)

    base_args = (["-config", "config.yaml"])
    opts, unknown = parser.parse_known_args(base_args)

    # build_vocab_main(opts)
    #
    src_vocab_path = opts.src_vocab
    tgt_vocab_path = opts.tgt_vocab

    counters = defaultdict(Counter)
    # load source vocab
    eng_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    _src_vocab = [[k, v] for k, v in eng_tokenizer.get_vocab().items()]
    _src_vocab_size = len(_src_vocab)
    for k, _ in _src_vocab:
        counters['src'][k] = 1000
    # _src_vocab, _src_vocab_size = _load_vocab(
    #     src_vocab_path,
    #     'src',
    #     counters)

    # load target vocab
    _tgt_vocab, _tgt_vocab_size = _load_vocab(
        tgt_vocab_path,
        'tgt',
        counters)

    # initialize fields
    src_nfeats, tgt_nfeats = 0, 0  # do not support word features for now
    fields = get_fields(
        'text', src_nfeats, tgt_nfeats)


    share_vocab = False
    vocab_size_multiple = 1
    src_vocab_size = _src_vocab_size #config['src_vocab_size']
    tgt_vocab_size = 128 #config['tgt_vocab_size']
    src_words_min_frequency = 1
    tgt_words_min_frequency = 1

    vocab_fields = _build_fields_vocab(
        fields, counters, 'text', share_vocab,
        vocab_size_multiple,
        src_vocab_size, src_words_min_frequency,
        tgt_vocab_size, tgt_words_min_frequency)


    src_text_field = vocab_fields["src"].base_field
    src_vocab = src_text_field.vocab
    src_padding = src_vocab.stoi[src_text_field.pad_token]

    tgt_text_field = vocab_fields['tgt'].base_field
    tgt_vocab = tgt_text_field.vocab
    tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]

    class BertEncoder(onmt.encoders.EncoderBase):
        """
        Pretrained Bert as encoder.
        """

        def __init__(self):
            super().__init__()

            self.base = AutoModel.from_pretrained('bert-base-cased')

        @classmethod
        def from_opt(cls, opt, embeddings=None):
            raise NotImplementedError

        @staticmethod
        def lengths_to_mask(lengths):
            maxlen = lengths.max()
            mask = torch.arange(maxlen)[None, :].to(lengths.device) < lengths[:, None]
            return mask.T

        def forward(self, src, lengths=None):
            """
            Args:
                src (LongTensor):
                   padded sequences of sparse indices ``(src_len, batch, nfeat)``
                lengths (LongTensor): length of each sequence ``(batch,)``


            Returns:
                (FloatTensor, FloatTensor, FloatTensor):

                * final encoder state, used to initialize decoder
                * memory bank for attention, ``(src_len, batch, hidden)``
                * lengths
            """

            self._check_args(src, lengths)
            attention_mask = self.lengths_to_mask(lengths)
            encoded = self.base(src.squeeze(), attention_mask=attention_mask)[0]

            final_state = (torch.zeros((1, encoded.shape[1], encoded.shape[2])).to(encoded.device),
                           torch.zeros((1, encoded.shape[1], encoded.shape[2])).to(encoded.device))

            return final_state, encoded, lengths

    emb_size = 100
    rnn_size = 768
    # Specify the core model.



    encoder = BertEncoder()

    decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab),
                                                 word_padding_idx=tgt_padding)
    decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
        hidden_size=rnn_size, num_layers=1, bidirectional_encoder=False,
        rnn_type="LSTM", embeddings=decoder_embeddings)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = onmt.models.model.NMTModel(encoder, decoder)
    model.to(device)

    # Specify the tgt word generator and loss computation module
    model.generator = nn.Sequential(
        nn.Linear(rnn_size, len(tgt_vocab)),
        nn.LogSoftmax(dim=-1)).to(device)

    loss = onmt.utils.loss.NMTLossCompute(
        criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
        generator=model.generator)

    lr = 1
    torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optim = onmt.utils.optimizers.Optimizer(
        torch_optimizer, learning_rate=lr, max_grad_norm=2)

    src_train = "E://Data/hu-nmt/combined-en-hu/hunglish2-tiny-train.en"  # opts.data['hunglish']['path_src']
    tgt_train = "E://Data/hu-nmt/combined-en-hu/hunglish2-tiny-train.hu"  # opts.data['hunglish']['path_tgt']
    src_val = "E://Data/hu-nmt/combined-en-hu/hunglish2-tiny-valid.en"   # opts.data['valid']['path_src']
    tgt_val = "E://Data/hu-nmt/combined-en-hu/hunglish2-tiny-train.hu"  # opts.data['valid']['path_tgt']

    # build the ParallelCorpus
    corpus = ParallelCorpus("corpus", src_train, tgt_train)
    valid = ParallelCorpus("valid", src_val, tgt_val)


    @register_transform(name='berttokenizer')
    class BertTransform(TokenizerTransform):
        """SentencePiece subword transform class."""

        def __init__(self, opts):
            """Initialize necessary options for sentencepiece."""
            super().__init__(opts)

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
                'src': eng_tokenizer,
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

    # class BertTransform(Transform):
    #     """Do Bert pretrained tokinization"""
    #
    #     def __init__(self, opts):
    #         super().__init__(opts)
    #         self.opts = opts
    #         self.eng_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    #         self.hun_tokenizer = None
    #
    #     def apply(self, example, is_train=False, stats=None, **kwargs):
    #         """Apply transform to `example`.
    #
    #         Args:
    #             example (dict): a dict of field value, ex. src, tgt;
    #             is_train (bool): Indicate if src/tgt is training data;
    #             stats (TransformStatistics): a statistic object.
    #         """
    #         example['src'] = self.eng_tokenizer.encode(example['src'], is_split_into_words=True)
    #         # res = {"src": "Fuck yeah", "tgt": "Bassza meg"}
    #         return example

    bertTransform = BertTransform(opts)
    bertTransform.warm_up()

    # build the training iterator
    train_iter = DynamicDatasetIter(
        corpora={"corpus": corpus},
        corpora_info={"corpus": {"weight": 1, "transforms": ["berttokenizer"]}},
        transforms={"berttokenizer": bertTransform},
        fields=vocab_fields,
        is_train=True,
        batch_type="sents",
        batch_size=8,
        batch_size_multiple=1,
        data_type="text")
    train_iter = iter(IterOnDevice(train_iter, 0))
    valid_iter = DynamicDatasetIter(
        corpora={"valid": valid},
        corpora_info={"valid": {"weight": 1, "transforms": ["berttokenizer"]}},
        transforms={"berttokenizer": bertTransform},
        fields=vocab_fields,
        is_train=False,
        batch_type="sents",
        batch_size=8,
        batch_size_multiple=1,
        data_type="text")

    valid_iter = IterOnDevice(valid_iter, 0)

    report_manager = onmt.utils.ReportMgr(
        report_every=1, start_time=None, tensorboard_writer=None)

    trainer = onmt.Trainer(model=model,
                           train_loss=loss,
                           valid_loss=loss,
                           optim=optim,
                           report_manager=report_manager,
                           dropout=[0.1])

    trainer.train(train_iter=train_iter,
                  train_steps=20,
                  valid_iter=valid_iter,
                  valid_steps=5)

