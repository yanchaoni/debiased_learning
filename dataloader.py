import Constants
import numpy as np
import torch
from torch.utils.data import Dataset


class SNLIDataset(Dataset):
    def __init__(self, data_list, device=Constants.DEVICE):
        self.s1_list, self.s2_list, self.target_list = zip(*data_list)
        self.device = device
        assert (len(self.s1_list) == len(self.target_list))

    def __len__(self):
        return len(self.target_list)
        
    def __getitem__(self, key):    
        s1_idx = self.s1_list[key][:Constants.MAXLEN[0]]
        s2_idx = self.s2_list[key][:Constants.MAXLEN[1]]        
        label = self.target_list[key]
        return [(s1_idx, s2_idx), (len(s1_idx), len(s2_idx)), label, self.device]

def sort_unsort(length_list):
    ind_dec_order = np.argsort(length_list)[::-1].copy()
    ind_unsort = np.argsort(ind_dec_order)
    return ind_dec_order, ind_unsort

def collate_func(batch):
    device = batch[0][3]
    s1_data_list, s2_data_list = [], []
    label_list = []
    s1_length_list, s2_length_list = [], []
    for datum in batch:
        label_list.append(datum[2])
        s1_length_list.append(datum[1][0])
        s2_length_list.append(datum[1][1])        
    # padding
    for datum in batch:
        padded_s1 = np.pad(np.array(datum[0][0]), 
                                pad_width=((0,Constants.MAXLEN[0]-datum[1][0])), 
                                mode="constant", constant_values=0)
        padded_s2 = np.pad(np.array(datum[0][1]), 
                                pad_width=((0,Constants.MAXLEN[1]-datum[1][1])), 
                                mode="constant", constant_values=0)
        s1_data_list.append(padded_s1)
        s2_data_list.append(padded_s2)
    s1_ind_sort, s1_ind_unsort = sort_unsort(s1_length_list)
    s2_ind_sort, s2_ind_unsort = sort_unsort(s2_length_list)
    
    return [(torch.from_numpy(np.array(s1_data_list)).to(device), torch.from_numpy(np.array(s2_data_list)).to(device)), 
            (torch.LongTensor(s1_length_list).to(device), torch.LongTensor(s2_length_list).to(device)), 
            (torch.LongTensor(s1_ind_sort).to(device), torch.LongTensor(s2_ind_sort).to(device)),
            (torch.LongTensor(s1_ind_unsort).to(device), torch.LongTensor(s2_ind_unsort).to(device)),
            torch.LongTensor(label_list).to(device)]