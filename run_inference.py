import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.data_functions import check_device, embedding_path, load_proteins, protein_name
from dataset.dataset_inference import Proteins_Dataset, text_collate_fn
from main import main_class, main_reg, write_csv
from models.bilstm import Network as Network1
from models.bilstm_reg import Network as Network4
from models.ms_res_lstm import Network as Network3
from models.ms_res_lstm_reg import Network as Network6
from models.ms_resnet import Network as Network2
from models.ms_resnet_reg import Network as Network5

REPO_DIR = Path(__file__).resolve().parent

CLASS_ENSEMBLE = ((Network1, 'model1.pt'), (Network2, 'model2.pt'), (Network3, 'model3.pt'))
REG_ENSEMBLE = ((Network4, 'model4.pt'), (Network5, 'model5.pt'), (Network6, 'model6.pt'))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict protein structural properties from ESM-1b and ProtTrans embeddings.")
    parser.add_argument('--file_list', required=True, type=str,
                        help='text file listing one FASTA file path per line')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device to run the models on, such as cpu or cuda:0 (default: %(default)s)')
    parser.add_argument('--save_path', default='results/', type=str,
                        help='output directory, created if needed (default: %(default)s)')
    parser.add_argument('--checkpoints_dir', default=REPO_DIR / 'checkpoints', type=Path,
                        help='directory holding model1.pt to model6.pt '
                             '(default: checkpoints/ next to this script)')
    return parser.parse_args()


def check_embeddings(proteins):
    for path, seq in proteins:
        for kind in ('esm', 'pt'):
            embedding = embedding_path(protein_name(path), kind)
            if not embedding.is_file():
                sys.exit(f'error: missing embedding {embedding}, '
                         'generate it with run_SPOT-1D-LM.sh')
            rows = np.load(embedding, mmap_mode='r').shape[0]
            if rows != len(seq):
                sys.exit(f'error: {embedding} has {rows} rows but {path} has {len(seq)} residues, '
                         'regenerate it with run_SPOT-1D-LM.sh')


def check_inputs(args):
    """Report anything that would stop the run before spending time on the models."""
    check_device(args.device)

    missing = [str(args.checkpoints_dir / filename)
               for _, filename in CLASS_ENSEMBLE + REG_ENSEMBLE
               if not (args.checkpoints_dir / filename).is_file()]
    if missing:
        sys.exit('error: missing model checkpoints (see "Installation" in the README):\n  '
                 + '\n  '.join(missing))

    proteins = load_proteins(args.file_list)
    check_embeddings(proteins)
    try:
        os.makedirs(args.save_path, exist_ok=True)
    except OSError as error:
        sys.exit(f'error: {error}')
    return [path for path, _ in proteins]


def load_ensemble(ensemble, checkpoints_dir, device):
    models = []
    for network, filename in ensemble:
        model = network()
        model.load_state_dict(torch.load(checkpoints_dir / filename, map_location='cpu',
                                         weights_only=True))
        models.append(model.to(device))
    return models


def main():
    args = parse_args()
    path_list = check_inputs(args)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # batch_size must stay 1: this implementation has only been tested with single proteins
    loader = DataLoader(Proteins_Dataset(path_list), batch_size=1, collate_fn=text_collate_fn)

    # Each ensemble is released before the next is loaded to keep peak memory down
    class_out = main_class(loader, *load_ensemble(CLASS_ENSEMBLE, args.checkpoints_dir, args.device),
                           args.device)
    reg_out = main_reg(loader, *load_ensemble(REG_ENSEMBLE, args.checkpoints_dir, args.device),
                       args.device)
    write_csv(class_out, reg_out, args.save_path)


if __name__ == '__main__':
    main()
