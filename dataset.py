import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import *


def process_tox21_smiles_data(data_dir):
    char_dict = dict()
    i = 0
    for c in CHAR_LIST:
        char_dict[c] = i
        i += 1
    char_list1 = list()
    char_list2 = list()
    char_dict1 = dict()
    char_dict2 = dict()
    for key in CHAR_LIST:
        if len(key) == 1:
            char_list1 += [key]
            char_dict1[key] = char_dict[key]
        elif len(key) == 2:
            char_list2 += [key]
            char_dict2[key] = char_dict[key]
        else:
            print("strange ", key)
    df = pd.read_csv(data_dir)
    target = df[TOX21_TASKS].values
    smiles_list = df.smiles.values
    Xdata = []
    Ldata = []
    Pdata = []
    for i, smi in enumerate(smiles_list):
        # print(type(line[0]))
        # line[0]: NR-AR, line[13]: smiles
        smiles_len = len(smi)
        if smiles_len > MAX_SEQ_LEN:
            continue
        label = target[i]
        label[np.isnan(label)] = 6  # nan set 6!
        Pdata.append(label)
        X_d = np.zeros((MAX_SEQ_LEN, CHAR_LEN))
        j = 0
        istring = 0
        check = True
        while check:
            char2 = smi[j: j + 2]
            char1 = smi[j]
            if char2 in char_list2:
                index = char_dict2[char2]
                j += 2
                if j >= smiles_len:
                    check = False
            elif char1 in char_list1:
                index = char_dict1[char1]
                j += 1
                if j >= smiles_len:
                    check = False
            else:
                print(char1, char2, "error")
                sys.exit()
            X_d[istring, index] = 1
            istring += 1
        for k in range(istring, MAX_SEQ_LEN):
            X_d[k, 0] = 1
        Xdata.append(X_d)
        Ldata.append(istring)
    weights = []
    for i, task in enumerate(TOX21_TASKS):
        negative_df = df[df[task] == 0][["smiles", task]]
        positive_df = df[df[task] == 1][["smiles", task]]
        neg_len = len(negative_df)
        pos_len = len(positive_df)
        weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
    X_data = np.asarray(Xdata, dtype="long")
    L_data = np.asarray(Ldata, dtype="long")
    P_data = np.asarray(Pdata, dtype="long")
    Weights = np.asarray(weights, dtype="float")
    print(X_data.shape, L_data.shape, P_data.shape, Weights.shape)
    np.save('data/X_data.npy', X_data)
    np.save('data/L_data.npy', L_data)
    np.save('data/P_data.npy', P_data)
    np.save('data/Weights.npy', Weights)
    return


class UserDataset(Dataset):
    def __init__(self, data_dir, name):
        Xdata = torch.tensor(np.load(data_dir + f'X_data.npy'), dtype=torch.long)
        Ldata = torch.tensor(np.load(data_dir + f'L_data.npy'), dtype=torch.long)
        Pdata = torch.tensor(np.load(data_dir + f'P_data.npy'), dtype=torch.long)
        len = Xdata.shape[0]
        train_num = int(0.7 * len)
        if name == 'train':
            self.Xdata = Xdata[:train_num]
            self.Ldata = Ldata[:train_num]
            self.Pdata = Pdata[:train_num]
            self.weights = torch.tensor(np.load(data_dir + 'Weights.npy'), dtype=torch.float)
            self.len = self.Xdata.shape[0]
        elif name == 'test':
            self.Xdata = Xdata[train_num:]
            self.Ldata = Ldata[train_num:]
            self.Pdata = Pdata[train_num:]
            self.len = self.Xdata.shape[0]
        elif name == 'unlabeled':
            self.Xdata = Xdata
            self.Ldata = Ldata
            self.Pdata = Pdata
            self.len = self.Xdata.shape[0]

    def __getitem__(self, index):
        return self.Xdata[index], self.Ldata[index], self.Pdata[index]

    def __len__(self):
        return self.len


def load_tox21_dataset(data_dir):
    # process_tox21_smiles_data(data_dir)
    train_data = UserDataset('data/', 'train')
    test_data = UserDataset('data/', 'test')
    unlabeled_data = UserDataset('data/', 'unlabeled')
    return train_data, test_data, unlabeled_data


if __name__ == '__main__':
    pass
    # process_tox21_smiles_data('data/tox21.csv')
    # train_data = UserDataset('data/', 'train')
    # test_data = UserDataset('data/', 'test')
    # unlabeled_data = UserDataset('data/', 'unlabeled')
