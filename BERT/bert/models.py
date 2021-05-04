from typing import Any

import onmt
from transformers import AutoModel
import torch


class BertEncoder(onmt.encoders.EncoderBase):
    """
    Pretrained Bert as encoder.
    """

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self):
        super().__init__()

        self.base = AutoModel.from_pretrained('bert-base-cased')
        for p in self.base.parameters():
            p.requires_grad = False

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
        Returns:
            (FloatTensor, FloatTensor, FloatTensor):

            * final encoder state, used to initialize decoder
            * memory bank for attention, ``(src_len, batch, hidden)``
            * lengths
        """

        self._check_args(src, lengths)
        attention_mask = self.lengths_to_mask(lengths)

        with torch.no_grad():
            encoded = self.base(src.squeeze(), attention_mask=attention_mask)[0]

        # Dummy final_state to remain compatible to original interface
        final_state = (torch.zeros((1, encoded.shape[1], encoded.shape[2])).to(encoded.device),
                       torch.zeros((1, encoded.shape[1], encoded.shape[2])).to(encoded.device))

        return final_state, encoded, lengths
