"""Stand-in for transformers, so the tests don't need the multi-GB ProtT5 weights."""
from types import SimpleNamespace

import torch
from torch import nn


class T5Tokenizer:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls()

    def batch_encode_plus(self, sequences, add_special_tokens=True, padding=True):
        (sequence,) = sequences
        length = len(sequence.split()) + 1
        return {"input_ids": [[1] * length], "attention_mask": [[1] * length]}


class T5EncoderModel(nn.Module):
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls()

    def forward(self, input_ids, attention_mask):
        assert not self.training, "ProtT5 must run in eval mode"
        return SimpleNamespace(last_hidden_state=torch.ones(*input_ids.shape, 1024))
