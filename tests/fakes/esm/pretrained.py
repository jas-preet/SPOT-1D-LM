import torch
from torch import nn


# The residues ESM-1b can encode. Any other letter raises KeyError in the real batch converter.
ALPHABET = "LAGVSERTIDPKQNFYMHWCXBUZO.-"


class FakeEsm(nn.Module):
    def forward(self, tokens, repr_layers=(), need_head_weights=False, return_contacts=False):
        assert not self.training, "ESM must run in eval mode"
        assert not return_contacts, "contact maps are unused and costly"
        return {"representations": {33: torch.ones(tokens.shape[0], tokens.shape[1], 1280)}}


class FakeAlphabet:
    def get_batch_converter(self):
        def convert(data):
            (label, seq), = data
            for residue in seq:
                if residue not in ALPHABET:
                    raise KeyError(residue)
            return [label], [seq], torch.zeros(1, len(seq) + 2, dtype=torch.long)

        return convert


def esm1b_t33_650M_UR50S():
    return FakeEsm(), FakeAlphabet()
