import time
import torch
import argparse
from torch.utils.data import DataLoader
# from config import PATH, LIST, VAL_LIST, TEST_LIST, TEST2_LIST, TEST3_LIST, TEST4_LIST, IGNORE_LABEL, DEVICE

from dataset.dataset_inference import Proteins_Dataset, text_collate_fn


from models.bilstm import Network
from models.ms_resnet import Network as Network2
from models.ms_res_lstm import Network as Network3
from models.bilstm_reg import Network as Network4
from models.ms_resnet_reg import Network as Network5
from models.ms_res_lstm_reg import Network as Network6

from main import main_reg, main_class, write_csv

parser = argparse.ArgumentParser()
parser.add_argument('--file_list', default='', type=str, help='file list path ')
parser.add_argument('--device', default='cpu', type=str,help=' define the device you want the ')
parser.add_argument('--save_path', default='results/', type=str,help=' define the device you want the ')
args = parser.parse_args()



test_set = Proteins_Dataset(args.file_list)  ## spot-1d test set
print("test_dataset Loaded with ", len(test_set), "proteins")
# this implementation has only been tested for batch size 1 only.
test_loader = DataLoader(test_set, batch_size=1, collate_fn=text_collate_fn, num_workers=16)

### final
path1 = "/home/jaspreet/jaspreet_data/results/project3/classification/bilstm2_both/"
path2 = "/home/jaspreet/jaspreet_data/results/project3/classification/ms_resnet_both/"
path3 = "/home/jaspreet/jaspreet_data/results/project3/classification/ms_res_lstm_both/"
path4 = "/home/jaspreet/jaspreet_data/results/project3/regression/bilstm_all_again/"
path5 = "/home/jaspreet/jaspreet_data/results/project3/regression/ms_resnet_all/"
path6 = "/home/jaspreet/jaspreet_data/results/project3/regression/ms_res_lstm_all_again/"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model1 = Network()
model2 = Network2()
model3 = Network3()
model4 = Network4()
model5 = Network5()
model6 = Network6()

model1.load_state_dict(torch.load(path1 + "best_checkpoint.pt"))
model2.load_state_dict(torch.load(path2 + "best_checkpoint.pt"))
model3.load_state_dict(torch.load(path3 + "best_checkpoint.pt"))
model4.load_state_dict(torch.load(path4 + "best_checkpoint.pt"))
model5.load_state_dict(torch.load(path5 + "best_checkpoint.pt"))
model6.load_state_dict(torch.load(path6 + "best_checkpoint.pt"))

model1 = model1.to(args.device)
model2 = model2.to(args.device)
model3 = model3.to(args.device)
model4 = model4.to(args.device)
model5 = model5.to(args.device)
model6 = model6.to(args.device)


class_out = main_class(test_loader, model1, model2, model3, args.device)
names, seq, ss3_pred_list, ss8_pred_list, ss3_prob_list, ss8_prob_list = class_out
reg_out = main_reg(test_loader, model4, model5, model6, args.device)
psi_list, phi_list, theta_list, tau_list, hseu_list, hsed_list, cn_list, asa_list = reg_out
print(len(ss3_pred_list), len(psi_list))
write_csv(class_out, reg_out, args.save_path)

