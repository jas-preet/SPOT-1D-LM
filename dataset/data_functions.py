import os
import sys
from pathlib import Path

import numpy as np
import torch

INPUTS_DIR = Path(__file__).resolve().parent.parent / 'inputs'


def read_list(file_name):
    """
    read a text file to get the list of elements, ignoring blank lines
    :param file_name: complete path to a file (string)
    :return: list of elements in the text file
    """
    with open(file_name, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def read_fasta_file(fname):
    """
    reads the single sequence from a fasta file
    :param fname: filename (string)
    :return: upper-case protein sequence, without whitespace or a trailing stop "*" (string)
    :raises ValueError: if the file does not hold exactly one sequence
    """
    with open(fname, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines or not lines[0].startswith('>'):
        raise ValueError(f"{fname}: expected a FASTA header line starting with '>'")
    if any(line.startswith('>') for line in lines[1:]):
        raise ValueError(f"{fname}: expected one sequence per file, found several")

    sequence = ''.join(''.join(lines[1:]).split()).upper().rstrip('*')
    if not sequence:
        raise ValueError(f"{fname}: no sequence after the header")
    return sequence


def protein_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def embedding_path(name, kind):
    """
    :param name: protein name (string)
    :param kind: "esm" or "pt" (string)
    :return: where that protein's embedding is stored (Path)
    """
    return INPUTS_DIR / f"{name}_{kind}.npy"


def load_proteins(file_list):
    """
    reads every protein in a file list, exiting with a message if any cannot be used
    :param file_list: path of a text file with one fasta path per line (string)
    :return: list of (fasta path, protein sequence)
    """
    try:
        paths = read_list(file_list)
        proteins = [(path, read_fasta_file(path)) for path in paths]
    except (OSError, ValueError) as error:
        sys.exit(f'error: {error}')
    if not proteins:
        sys.exit(f'error: {file_list} does not list any FASTA files')
    return proteins


def check_device(device):
    try:
        parsed = torch.device(device)
    except RuntimeError:
        sys.exit(f'error: unknown device {device}')
    if parsed.type == 'cuda' and not torch.cuda.is_available():
        sys.exit(f'error: device {device} was requested but CUDA is not available')


def one_hot(seq):
    """
    converts a sequence to one hot encoding
    :param seq: amino acid sequence (string)
    :return: one hot encoding of the amino acid (array)[L,20]
    """
    prot_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in prot_seq])
    return feat
