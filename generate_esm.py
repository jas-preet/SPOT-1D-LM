import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from dataset.data_functions import (INPUTS_DIR, check_device, embedding_path, load_proteins,
                                    protein_name)

# ESM-1b has 1024 positions, two of which hold the start and end tokens
ESM_MAX_RESIDUES = 1022


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ESM-1b embeddings.")
    parser.add_argument('--file_list', required=True, type=str,
                        help='text file listing one FASTA file path per line')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to run ESM-1b on, such as cpu or cuda:0 (default: %(default)s)')
    return parser.parse_args()


def load_model(device):
    import esm

    # The ESM-1b checkpoint stores its settings as a Namespace, which PyTorch refuses by default
    with torch.serialization.safe_globals([argparse.Namespace]):
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    return model.to(device).eval(), alphabet.get_batch_converter()


def main():
    args = parse_args()
    check_device(args.device)
    proteins = load_proteins(args.file_list)
    too_long = [path for path, seq in proteins if len(seq) > ESM_MAX_RESIDUES]
    if too_long:
        sys.exit(f'error: ESM-1b accepts up to {ESM_MAX_RESIDUES} residues, these are longer:\n  '
                 + '\n  '.join(too_long))

    os.makedirs(INPUTS_DIR, exist_ok=True)
    model, batch_converter = load_model(args.device)

    for prot_path, seq in tqdm(proteins):
        prot_name = protein_name(prot_path)
        _, _, batch_tokens = batch_converter([(prot_name, seq)])
        with torch.no_grad():
            results = model(batch_tokens.to(args.device), repr_layers=[33])
        embedding = results["representations"][33][0, 1: len(seq) + 1].cpu().numpy()
        np.save(embedding_path(prot_name, "esm"), embedding)
    print(" ESM-1b embeddings generation completed ... ")


if __name__ == '__main__':
    main()
