import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

SS3_CLASSES = 'CEH'
SS8_CLASSES = 'CSTHGIEB'

CSV_COLUMNS = ["AA", "SS3", "SS8", "ASA", "HseU", "HseD", "CN", "Psi", "Phi", "Theta", "Tau",
               "P3C", "P3E", "P3H", "P8C", "P8S", "P8T", "P8H", "P8G", "P8I", "P8E", "P8B"]

# Maximum accessible surface area per residue, used to convert relative ASA to absolute.
# Anything else (X, B, Z, U, O, J, ...) falls back to UNKNOWN_RESIDUE_ASA.
MAX_ASA = dict(zip(
    "ACDEFGHIKLMNPQRSTVWY",
    (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
     185, 160, 145, 180, 225, 115, 140, 155, 255, 230),
))
UNKNOWN_RESIDUE_ASA = 1

# Output channels of the regression models. Each angle is a (sin, cos) pair.
PSI, PHI, THETA, TAU = 0, 2, 4, 6
HSE_UP, HSE_DOWN, CONTACT_NUMBER, REL_ASA = 8, 9, 10, -1

# The models predict these quantities scaled to [0, 1].
HSE_UP_SCALE, HSE_DOWN_SCALE, CONTACT_NUMBER_SCALE = 50, 65, 85


def get_unnorm_asa(rel_asa, seqs):
    """
    :param rel_asa: predicted relative ASA, padded to the longest sequence (tensor)[B,L]
    :param seqs: protein sequences in the batch, longest first
    :return: absolute ASA (array)[B,L]
    """
    rel_asa = rel_asa.cpu().detach().numpy()
    asa_max = np.full(rel_asa.shape, UNKNOWN_RESIDUE_ASA, dtype=np.float32)
    for i, seq in enumerate(seqs):
        asa_max[i, :len(seq)] = [MAX_ASA.get(residue.upper(), UNKNOWN_RESIDUE_ASA) for residue in seq]
    return np.multiply(rel_asa, asa_max)


def get_angle_degree(preds):
    preds = preds * 2 - 1
    preds_sin = preds[:, :, 0]
    preds_cos = preds[:, :, 1]
    preds_angle_rad = np.arctan2(preds_sin, preds_cos)
    preds_angle = np.degrees(preds_angle_rad)
    return preds_angle


def predict(models, feats, length):
    return [model(feats, length) for model in models]


def median_angle(preds, channel):
    """Ensemble an angle by taking the median of its sin and cos, then convert to degrees."""
    stacked = torch.stack([pred[:, :, channel:channel + 2] for pred in preds], dim=3)
    median, _ = torch.median(stacked, dim=-1)
    return get_angle_degree(median.cpu().numpy())


def mean_channel(preds, channel):
    pred1, pred2, pred3 = preds
    return (pred1[:, :, channel] + pred2[:, :, channel] + pred3[:, :, channel]) / 3


def append_trimmed(target, values, length):
    """Remove the padding from each protein in the batch and add it to the target list."""
    for i, prot_len in enumerate(length):
        target.append(values[i, :prot_len, None])


@torch.no_grad()
def main_reg(data_loader, model1, model2, model3, device):
    models = (model1, model2, model3)
    psi_list = []
    phi_list = []
    theta_list = []
    tau_list = []
    hseu_list = []
    hsed_list = []
    cn_list = []
    asa_list = []

    for model in models:
        model.eval()

    for feats, length, _, seqs in tqdm(data_loader):
        feats = feats.to(device, dtype=torch.float)
        preds = predict(models, feats, length)

        append_trimmed(psi_list, median_angle(preds, PSI), length)
        append_trimmed(phi_list, median_angle(preds, PHI), length)
        append_trimmed(theta_list, median_angle(preds, THETA), length)
        append_trimmed(tau_list, median_angle(preds, TAU), length)

        append_trimmed(hseu_list, (mean_channel(preds, HSE_UP) * HSE_UP_SCALE).cpu().numpy(), length)
        append_trimmed(hsed_list, (mean_channel(preds, HSE_DOWN) * HSE_DOWN_SCALE).cpu().numpy(), length)
        append_trimmed(cn_list, (mean_channel(preds, CONTACT_NUMBER) * CONTACT_NUMBER_SCALE).cpu().numpy(), length)
        append_trimmed(asa_list, get_unnorm_asa(mean_channel(preds, REL_ASA), seqs), length)

    return psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list


def decode_classes(scores, classes):
    indices = np.argmax(scores, axis=1)
    return np.array([classes[index] for index in indices])[:, None]


@torch.no_grad()
def main_class(data_loader, model1, model2, model3, device):
    models = (model1, model2, model3)
    ss3_pred_list = []
    ss8_pred_list = []
    ss3_prob_list = []
    ss8_prob_list = []
    names_list = []
    seq_list = []

    for model in models:
        model.eval()

    for feats, length, names, seqs in tqdm(data_loader):
        feats = feats.to(device, dtype=torch.float)

        pred1, pred2, pred3 = predict(models, feats, length)
        pred = (pred1 + pred2 + pred3) / 3

        ss3_scores = pred[:, :, 0:3]
        ss8_scores = pred[:, :, 3:]
        ss3_probs = F.softmax(ss3_scores, dim=2).cpu().numpy()
        ss8_probs = F.softmax(ss8_scores, dim=2).cpu().numpy()
        ss3_scores = ss3_scores.cpu().numpy()
        ss8_scores = ss8_scores.cpu().numpy()

        for i, prot_len in enumerate(length):
            ss3_pred_list.append(decode_classes(ss3_scores[i, :prot_len], SS3_CLASSES))
            ss8_pred_list.append(decode_classes(ss8_scores[i, :prot_len], SS8_CLASSES))
            ss3_prob_list.append(ss3_probs[i, :prot_len])
            ss8_prob_list.append(ss8_probs[i, :prot_len])
            names_list.append(names[i])
            seq_list.append(np.array(list(seqs[i]))[:, None])

    return names_list, seq_list, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list


def write_csv(class_out, reg_out, save_dir):
    names, seqs, ss3_preds, ss8_preds, ss3_probs, ss8_probs = class_out
    psis, phis, thetas, taus, hseus, hseds, cns, asas = reg_out

    for name, *columns in zip(names, seqs, ss3_preds, ss8_preds, asas, hseus, hseds, cns,
                              psis, phis, thetas, taus, ss3_probs, ss8_probs):
        data = np.concatenate(columns, axis=1)
        save_path = os.path.join(save_dir, name + ".csv")
        pd.DataFrame(data).to_csv(save_path, header=CSV_COLUMNS)
    print(f'Results saved to {save_dir}')


if __name__ == '__main__':
    print("Please run run_SPOT-1D-LM.sh instead")
