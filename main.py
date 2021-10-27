import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_unnorm_asa_new(rel_asa, seq):
    """
    :param asa_pred: The predicted relative ASA
    :param seq_list: Sequence of the protein
    :return: absolute ASA_PRED
    """
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY-X"

    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
               185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1, 1)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))

    max_seq_len = len(seq[0])
    array_list = []
    for i, single_seq in enumerate(list(seq)):
        rel_asa_current = rel_asa[i, :]
        seq_len_diff = max_seq_len - len(single_seq)
        single_seq = single_seq + ("X" * seq_len_diff)
        asa_max = np.array([dict_rnam1_ASA[i] for i in single_seq]).astype(np.float32)
        abs_asa = np.multiply(rel_asa_current.cpu().detach().numpy(), asa_max)
        array_list.append(abs_asa)

    final_array = np.array(array_list)
    return final_array


def get_angle_degree(preds):

    preds = preds * 2 - 1
    preds_sin = preds[:, :, 0]
    preds_cos = preds[:, :, 1]
    preds_angle_rad = np.arctan2(preds_sin, preds_cos)
    preds_angle = np.degrees(preds_angle_rad)
    return preds_angle

ss_conv_3_8_dict = {'X': 'X', 'C': 'C', 'S': 'C', 'T': 'C', 'H': 'H', 'G': 'H', 'I': 'H', 'E': 'E', 'B': 'E'}
SS3_CLASSES = 'CEH'
SS8_CLASSES = 'CSTHGIEB'

def main_reg(data_loader, model1, model2, model3, device):
    psi_list = []
    phi_list = []
    theta_list = []
    tau_list = []
    hseu_list = []
    hsed_list = []
    cn_list = []
    asa_list = []

    model1.eval()
    model2.eval()
    model3.eval()

    for i, data in enumerate(tqdm(data_loader)):
        feats, length, name, seq = data

        feats = feats.to(device, dtype=torch.float)

        pred1 = model1(feats, length)
        pred2 = model2(feats, length)
        pred3 = model3(feats, length)

        psi_pred1 = pred1[:, :, 0:2].unsqueeze(3)
        psi_pred2 = pred2[:, :, 0:2].unsqueeze(3)
        psi_pred3 = pred3[:, :, 0:2].unsqueeze(3)
        psi_pred_cat = torch.cat((psi_pred1, psi_pred2, psi_pred3), 3)
        psi_pred, _ = torch.median(psi_pred_cat, dim=-1)
        psi_deg = get_angle_degree(psi_pred.cpu().detach().numpy())
        for i, len_prot in enumerate(list(length)):
            psi_list.append(psi_deg[i, :int(len_prot), None])

        phi_pred1 = pred1[:, :, 2:4].unsqueeze(3)
        phi_pred2 = pred2[:, :, 2:4].unsqueeze(3)
        phi_pred3 = pred3[:, :, 2:4].unsqueeze(3)
        phi_pred_cat = torch.cat((phi_pred1, phi_pred2, phi_pred3), 3)
        phi_pred, _ = torch.median(phi_pred_cat, dim=-1)
        phi_deg = get_angle_degree(phi_pred.cpu().detach().numpy())
        for i, len_prot in enumerate(list(length)):
            phi_list.append(phi_deg[i, :int(len_prot), None])

        theta_pred1 = pred1[:, :, 4:6].unsqueeze(3)
        theta_pred2 = pred2[:, :, 4:6].unsqueeze(3)
        theta_pred3 = pred3[:, :, 4:6].unsqueeze(3)
        theta_pred_cat = torch.cat((theta_pred1, theta_pred2, theta_pred3), 3)
        theta_pred, _ = torch.median(theta_pred_cat, dim=-1)
        theta_deg = get_angle_degree(theta_pred.cpu().detach().numpy())
        for i, len_prot in enumerate(list(length)):
            theta_list.append(theta_deg[i, :int(len_prot), None])

        tau_pred1 = pred1[:, :, 6:8].unsqueeze(3)
        tau_pred2 = pred2[:, :, 6:8].unsqueeze(3)
        tau_pred3 = pred3[:, :, 6:8].unsqueeze(3)
        tau_pred_cat = torch.cat((tau_pred1, tau_pred2, tau_pred3), 3)
        tau_pred, _ = torch.median(tau_pred_cat, dim=-1)
        tau_deg = get_angle_degree(tau_pred.cpu().detach().numpy())
        # tau_list = [tau_deg[i, :int(len_prot), None] for i, len_prot in enumerate(list(length))]
        for i, len_prot in enumerate(list(length)):
            tau_list.append(tau_deg[i, :int(len_prot), None])

        hseu_pred1 = pred1[:, :, 8]
        hseu_pred2 = pred2[:, :, 8]
        hseu_pred3 = pred3[:, :, 8]
        hseu_pred = ((hseu_pred1 + hseu_pred2 + hseu_pred3) / 3) * 50
        hseu_pred = hseu_pred.cpu().detach().numpy()
        for i, len_prot in enumerate(list(length)):
            hseu_list.append(hseu_pred[i, :int(len_prot), None])

        hsed_pred1 = pred1[:, :, 9]
        hsed_pred2 = pred2[:, :, 9]
        hsed_pred3 = pred3[:, :, 9]
        hsed_pred = ((hsed_pred1 + hsed_pred2 + hsed_pred3) / 3) * 65
        hsed_pred = hsed_pred.cpu().detach().numpy()
        for i, len_prot in enumerate(list(length)):
            hsed_list.append(hsed_pred[i, :int(len_prot), None])

        cn_pred1 = pred1[:, :, 10]
        cn_pred2 = pred2[:, :, 10]
        cn_pred3 = pred3[:, :, 10]
        cn_pred = ((cn_pred1 + cn_pred2 + cn_pred3) / 3) * 85
        cn_pred = cn_pred.cpu().detach().numpy()
        for i, len_prot in enumerate(list(length)):
            cn_list.append(cn_pred[i, :int(len_prot), None])

        asa_pred1 = pred1[:, :, -1]
        asa_pred2 = pred2[:, :, -1]
        asa_pred3 = pred3[:, :, -1]
        asa_pred = (asa_pred1 + asa_pred2 + asa_pred3) / 3
        asa_pred = get_unnorm_asa_new(asa_pred, seq)
        asa_pred = asa_pred
        for i, len_prot in enumerate(list(length)):
            asa_list.append(asa_pred[i, :int(len_prot), None])

    return psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list


def main_class(data_loader, model1, model2, model3, device):
    ss3_pred_list = []
    ss8_pred_list = []
    ss3_prob_list = []
    ss8_prob_list = []
    names_list = []
    seq_list = []

    model1.eval()
    model2.eval()
    model3.eval()

    for i, data in enumerate(tqdm(data_loader)):
        feats, length, name, seq = data

        feats = feats.to(device, dtype=torch.float)

        pred1 = model1(feats, length)
        pred2 = model2(feats, length)
        pred3 = model3(feats, length)

        pred = (pred1 + pred2 + pred3) / 3

        pred = pred.view(-1, 11)

        ss3_pred = pred[:, 0:3]
        ss8_pred = pred[:, 3:]

        name = list(name)
        for i, prot_len in enumerate(list(length)):
            prot_len_int = int(prot_len)
            ss3_pred_single = ss3_pred[:prot_len_int, :]
            ss3_pred_single = ss3_pred_single.cpu().detach().numpy()
            ss3_indices = np.argmax(ss3_pred_single, axis=1)
            ss3_pred_aa = np.array([SS3_CLASSES[aa] for aa in ss3_indices])[:, None]
            ss3_pred_list.append(ss3_pred_aa)
            ss3_prob_list.append(ss3_pred_single)

            ss8_pred_single = ss8_pred[:prot_len_int, :]
            ss8_pred_single = ss8_pred_single.cpu().detach().numpy()
            ss8_indices = np.argmax(ss8_pred_single, axis=1)
            ss8_pred_aa = np.array([SS8_CLASSES[aa] for aa in ss8_indices])[:, None]
            ss8_pred_list.append(ss8_pred_aa)
            ss8_prob_list.append(ss8_pred_single)
            names_list.append(name[i])
        for seq in list(seq):
            seq_list.append(np.array([i for i in seq])[:, None])

    return names_list, seq_list, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list


def write_csv(class_out, reg_out, save_dir):
    names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out
    psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list = reg_out

    for seq, ss3, ss8, asa, hseu, hsed, cn, psi, phi, theta, tau, ss3_prob, ss8_prob, name in zip(seq, ss3_pred_list,
                                                                                                  ss8_pred_list,
                                                                                                  asa_list, hseu_list,
                                                                                                  hsed_list, cn_list,
                                                                                                  psi_list, phi_list,
                                                                                                  theta_list, tau_list,
                                                                                                  ss3_prob_list,
                                                                                                  ss8_prob_list, names):
        data = np.concatenate((seq, ss3, ss8, asa, hseu, hsed, cn, psi, phi, theta, tau, ss3_prob, ss8_prob), axis=1)

        save_path = os.path.join(save_dir, name + ".csv")
        pd.DataFrame(data).to_csv(save_path,
                                  header=["AA", "SS3", "SS8", "ASA", "HseU", "HseD", "CN", "Psi", "Phi", "Theta",
                                          "Tau", "P3C", "P3E", "P3H", "P8C", "P8S", "P8T", "P8H", "P8G",
                                          "P8I", "P8E", "P8B"])
    return print(f'please find the results saved at {save_dir} with .csv extention')


if __name__ == '__main__':
    print("Please run the run_SPOT-1D-LM.sh instead")
