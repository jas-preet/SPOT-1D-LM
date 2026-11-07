import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dataset.data_functions import embedding_path, one_hot, protein_name, read_fasta_file


class Proteins_Dataset(Dataset):
    def __init__(self, protein_list):
        self.protein_list = protein_list

    def __len__(self):
        return len(self.protein_list)

    def __getitem__(self, idx):
        prot_path = self.protein_list[idx]
        protein = protein_name(prot_path)

        seq = read_fasta_file(prot_path)
        one_hot_enc = one_hot(seq)
        embedding1 = np.load(embedding_path(protein, "esm"))
        embedding2 = np.load(embedding_path(protein, "pt"))

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
