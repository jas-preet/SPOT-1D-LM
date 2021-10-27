import time
start_time = time.time()
import re
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer
from dataset.data_functions import read_list, read_fasta_file

parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--device', default='cpu', type=str,help=' define the device you want the ')
args = parser.parse_args()


tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

device = torch.device(args.device)

model = model.to(args.device)
model = model.eval()

prot_list = read_list(args.file_list)


for prot_path in tqdm(prot_list):
    seq = read_fasta_file(prot_path)
    prot_name = prot_path.split('/')[-1].split('.')[0]


    seq_temp = seq.replace('', " ")
    sequences_Example = [seq_temp]
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, padding=True)

    input_ids = torch.tensor(ids['input_ids']).to(args.device)
    attention_mask = torch.tensor(ids['attention_mask']).to(args.device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    if args.device == "cpu":
        embedding = embedding.last_hidden_state.numpy()
    else:
        embedding = embedding.last_hidden_state.cpu().numpy()

    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    np.save("inputs/" + prot_name + "_pt.npy", features[0])
print(" ProtTrans embeddings generation completed ... ")
