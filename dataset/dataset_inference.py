import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from dataset.data_functions import one_hot, read_list, read_fasta_file

# def one_hot(seq):
#     RNN_seq = seq
#     BASES = 'ARNDCQEGHILKMFPSTWYV'
#     bases = np.array([base for base in BASES])
#     feat = np.concatenate(
#         [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
#          in RNN_seq])
#
#     return feat
#
#
# def read_list(file_name):
#     '''
#     returns list of proteins from file
#     '''
#     with open(file_name, 'r') as f:
#         text = f.read().splitlines()
#     return text
#
#
# def read_fasta_file(fname):
#     """
#     reads the sequence from the fasta file
#     :param fname: filename (string)
#     :return: protein sequence  (string)
#     """
#     with open(fname, 'r') as f:
#         AA = ''.join(f.read().splitlines()[1:])
#     return AA
#

class Proteins_Dataset(Dataset):
    def __init__(self, list):
        self.protein_list = read_list(list)

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        prot_path = self.protein_list[idx]
        protein = prot_path.split('/')[-1].split('.')[0]

        seq = read_fasta_file(prot_path)
        one_hot_enc = one_hot(seq)
        embedding1 = np.load(os.path.join("inputs/", protein + "_esm.npy"))
        embedding2 = np.load(os.path.join("inputs/", protein + "_pt.npy"))

        features = np.concatenate((one_hot_enc, embedding1, embedding2), axis=1)

        protein_len = len(seq)

        return features, protein_len, protein, seq


def text_collate_fn(data):
    """
    collate function for data read from text file
    """

    # sort data by caption length
    data.sort(key=lambda x: x[1], reverse=True)
    features, protein_len, protein, seq = zip(*data)
    features = [torch.FloatTensor(x) for x in features]

    # Pad features
    padded_features = nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)

    return padded_features, protein_len, protein, seq
