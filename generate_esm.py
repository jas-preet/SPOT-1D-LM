import time
import os.path
import torch
import argparse
import numpy as np
from tqdm import tqdm
from dataset.data_functions import read_list, read_fasta_file

parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--device', default='cpu', type=str,help=' define the device you want the ')
args = parser.parse_args()

model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
batch_converter = alphabet.get_batch_converter()
model = model.to(args.device)

prot_list = read_list(args.file_list)


for prot_path in tqdm(prot_list):

    prot_name = prot_path.split('/')[-1].split('.')[0]
    save_path = "inputs/" + prot_name + "_esm.npy"
    seq = read_fasta_file(prot_path)
    data = [(prot_name, seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(args.device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, (prot_n, seq) in enumerate(data):
        if args.device == "cpu":
            save_arr = token_representations[i, 1: len(seq) + 1].numpy()
        else:
            save_arr = token_representations[i, 1: len(seq) + 1].cpu().numpy()
        np.save(save_path, save_arr)
print(" ESM-1b embeddings generation completed ... ")