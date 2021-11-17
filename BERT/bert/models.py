from typing import Any

import onmt
import torch
from transformers import AutoModel
import logging

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

class BertEncoder(onmt.encoders.EncoderBase):
    """
    Pretrained Bert as encoder.
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, encoder_freezed=True):
        super().__init__()
        self.encoder_freezed = encoder_freezed

        self.base = AutoModel.from_pretrained("bert-base-cased")

        if self.encoder_freezed:
            logging.info("Freezing BERT.")
            for p in self.base.parameters():
                p.requires_grad = False
        else:
            logging.info("BERT stays unfreezed.")

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        return BertEncoder(opt.encoder_freezed)

    @staticmethod
    def lengths_to_mask(lengths):
        maxlen = lengths.max()
        mask = torch.arange(maxlen)[None, :].to(lengths.device) < lengths[:, None]
        return mask.T

    def forward(self, src, lengths=None):
        """
        Returns:
            (FloatTensor, FloatTensor, FloatTensor):

            * final encoder state, used to initialize decoder
            * memory bank for attention, ``(src_len, batch, hidden)``
            * lengths
        """

        self._check_args(src, lengths)
        attention_mask = self.lengths_to_mask(lengths).transpose(0, 1)

        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # sent = src[:, 0, 0].cpu().numpy()
        #print(tokenizer.decode(src[:, 0, 0].cpu().numpy(), skip_special_tokens=True))

        if self.encoder_freezed:
            with torch.no_grad():
                encoded = self.base(
                    src.squeeze(-1).transpose(0, 1), attention_mask=attention_mask
                )[0].transpose(0, 1)
        else:
            encoded = self.base(
                src.squeeze(-1).transpose(0, 1), attention_mask=attention_mask
            )[0].transpose(0, 1)

        # Dummy final_state to remain compatible to original interface
        final_state = (
            torch.zeros((1, encoded.shape[1], encoded.shape[2])).to(encoded.device),
            torch.zeros((1, encoded.shape[1], encoded.shape[2])).to(encoded.device),
        )

        return final_state, encoded, lengths
