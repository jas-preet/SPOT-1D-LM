import argparse
import os
import re
import sys

import numpy as np
import torch
from tqdm import tqdm

from dataset.data_functions import (INPUTS_DIR, check_device, embedding_path, load_proteins,
                                    protein_name)

MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
# ProtT5 has no token for these. Left in, adjacent ones merge into a single token.
NON_STANDARD_RESIDUES = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ProtTrans (ProtT5-XL) embeddings.")
    parser.add_argument('--file_list', required=True, type=str,
                        help='text file listing one FASTA file path per line')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to run ProtT5 on, such as cpu or cuda:0 (default: %(default)s)')
    return parser.parse_args()


def load_model(device):
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(MODEL_NAME)
    return tokenizer, model.to(device).eval()


def main():
    args = parse_args()
    check_device(args.device)
    proteins = load_proteins(args.file_list)

    os.makedirs(INPUTS_DIR, exist_ok=True)
    tokenizer, model = load_model(args.device)

    for prot_path, seq in tqdm(proteins):
        spaced_seq = NON_STANDARD_RESIDUES.sub("X", seq).replace('', " ")
        ids = tokenizer.batch_encode_plus([spaced_seq], add_special_tokens=True, padding=True)

        if len(ids['input_ids'][0]) != len(seq) + 1:
            sys.exit(f'error: {prot_path}: ProtT5 tokens do not line up with the residues')

        input_ids = torch.tensor(ids['input_ids']).to(args.device)
        attention_mask = torch.tensor(ids['attention_mask']).to(args.device)
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        # The final position holds the end-of-sequence token
        features = embedding.last_hidden_state[0, :len(seq)].cpu().numpy()
        np.save(embedding_path(protein_name(prot_path), "pt"), features)
    print(" ProtTrans embeddings generation completed ... ")


if __name__ == '__main__':
    main()
