import numpy as np

def read_list(file_name):
    """
    read a text file to get the list of elements
    :param file_name: complete path to a file (string)
    :return: list of elements in the text file
    """
    with open(file_name, 'r') as f:
        text = f.read().splitlines()
    return text


def read_fasta_file(fname):
    """
    reads the sequence from the fasta file
    :param fname: filename (string)
    :return: protein sequence  (string)
    """
    with open(fname, 'r') as f:
        AA = ''.join(f.read().splitlines()[1:])
    return AA


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