import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import *


def process_tox21_smiles_data(data_dir):
    char_dict = dict()
    i = 0
    for c in TOX21_CHAR_LIST:
        char_dict[c] = i
        i += 1
    char_list1 = list()
    char_list2 = list()
    char_dict1 = dict()
    char_dict2 = dict()
    for key in TOX21_CHAR_LIST:
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
    # pos_id = []
    # neg_id = []
    for i, smi in enumerate(smiles_list):
        # print(type(line[0]))
        # line[0]: NR-AR, line[13]: smiles
        smiles_len = len(smi)
        label = target[i]
        if smiles_len > MAX_TOX21_LEN:
            continue
        # label[np.isnan(label)] = 6
        # pos_id.append([1 if item == 1 else 0 for item in label])
        # neg_id.append([1 if item == 0 else 0 for item in label])
        Pdata.append(label)
        X_d = np.zeros((MAX_TOX21_LEN, len(TOX21_CHAR_LIST)))
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
        for k in range(istring, MAX_TOX21_LEN):
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
    # pos_data = np.asarray(pos_id, dtype="long")
    # neg_data = np.asarray(neg_id, dtype="long")
    Weights = np.asarray(weights, dtype="float")
    print(X_data.shape, L_data.shape, P_data.shape, Weights.shape)
    np.save('data/X_data.npy', X_data)
    np.save('data/L_data.npy', L_data)
    np.save('data/P_data.npy', P_data)
    np.save('data/Weights.npy', Weights)
    # np.save('data/pos_index.npy', pos_data)
    # np.save('data/neg_index.npy', neg_data)
    return


def process_qm9_smiles_data(data_dir):
    """
    max smiles len: 42
    total smiles: 133885
    """
    char_dict = dict()
    i = 0
    for c in QM9_CHAR_LIST:
        char_dict[c] = i
        i += 1
    char_list1 = list()
    char_list2 = list()
    char_dict1 = dict()
    char_dict2 = dict()
    for key in QM9_CHAR_LIST:
        if len(key) == 1:
            char_list1 += [key]
            char_dict1[key] = char_dict[key]
        elif len(key) == 2:
            char_list2 += [key]
            char_dict2[key] = char_dict[key]
        else:
            print("strange ", key)
    df = pd.read_csv(data_dir)
    target = df[QM9_TASKS].values
    pmax = np.array(target).max(axis=0)
    pmin = np.array(target).min(axis=0)
    mean = np.array(target).mean(axis=0)
    std = np.array(target).std(axis=0)
    smiles_list = df.smiles.values
    Xdata = []
    Ldata = []
    Pdata = []
    for i, smi in enumerate(smiles_list):
        smiles_len = len(smi)
        # labels = (target[i] - pmin) / (pmax - pmin)  # normalization
        labels = target[i] - mean / std
        Pdata.append(labels)
        # X_d = np.zeros((MAX_QM9_LEN, len(QM9_CHAR_LIST)))  # one-hot
        X_d = []
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
            # X_d[istring, index] = 1
            X_d.append(index)
            istring += 1
        Ldata.append(istring)
        for k in range(istring, MAX_QM9_LEN):
            # X_d[k, 0] = 1
            X_d.append(0)
        Xdata.append(X_d)
    X_data = np.asarray(Xdata, dtype="long")
    L_data = np.asarray(Ldata, dtype="long")
    P_data = np.asarray(Pdata, dtype="float")
    print(X_data.shape, L_data.shape, P_data.shape)
    # X_data(133885, 42)  P_data(133885, 12)
    np.save('data/qm9/X_data.npy', X_data)
    np.save('data/qm9/L_data.npy', L_data)
    np.save('data/qm9/P_data.npy', P_data)


class QM9Dataset(Dataset):
    def __init__(self, data_dir, name, task_no):
        Xdata = torch.tensor(np.load(data_dir + f'X_data.npy'), dtype=torch.long)
        Ldata = torch.tensor(np.load(data_dir + f'L_data.npy'), dtype=torch.long)
        Pdata = torch.tensor(np.load(data_dir + f'P_data.npy'), dtype=torch.float)
        data_len = Xdata.shape[0]
        index = list(range(data_len))
        # random.shuffle(index)
        train_num = int(0.8 * data_len)
        if name == 'train':
            Pdata = Pdata[:, task_no]
            self.Xdata = Xdata[index[:train_num]]
            self.Ldata = Ldata[index[:train_num]]
            self.Pdata = Pdata[index[:train_num]]
            self.len = self.Xdata.shape[0]
        elif name == 'test':
            Pdata = Pdata[:, task_no]
            self.Xdata = Xdata[index[train_num:]]
            self.Ldata = Ldata[index[train_num:]]
            self.Pdata = Pdata[index[train_num:]]
            self.len = self.Xdata.shape[0]
        elif name == 'all':
            self.Xdata = Xdata
            self.Ldata = Ldata
            self.Pdata = Pdata
            self.len = self.Xdata.shape[0]

    def __getitem__(self, index):
        return self.Xdata[index], self.Ldata[index], self.Pdata[index]

    def __len__(self):
        return self.len


class UserDataset(Dataset):
    def __init__(self, data_dir, name, task_name=None):
        Xdata = torch.tensor(np.load(data_dir + f'X_data.npy'), dtype=torch.long)
        Ldata = torch.tensor(np.load(data_dir + f'L_data.npy'), dtype=torch.long)
        Pdata = torch.tensor(np.load(data_dir + f'P_data.npy'), dtype=torch.long)
        if task_name is not None:
            task_num = TOX21_TASKS.index(task_name)
            pos_id = []
            neg_id = []
            for i in range(Pdata.shape[0]):
                if Pdata[i, task_num] == 1:
                    pos_id.append(i)
                if Pdata[i, task_num] == 0:
                    neg_id.append(i)
            random.shuffle(pos_id)
            random.shuffle(neg_id)
            train_pos_num = int(0.8 * len(pos_id))
            train_neg_num = int(0.8 * len(neg_id))
        train_num = int(0.8 * len(Xdata))
        if name == 'train':
            if task_name is None:
                self.Xdata = Xdata[:train_num]
                self.Ldata = Ldata[:train_num]
                self.Pdata = Pdata[:train_num]
            else:
                self.Xdata = Xdata[pos_id[:train_pos_num] + neg_id[:train_neg_num]]
                self.Ldata = Ldata[pos_id[:train_pos_num] + neg_id[:train_neg_num]]
                self.Pdata = Pdata[pos_id[:train_pos_num] + neg_id[:train_neg_num], task_num]
            self.weights = torch.tensor(np.load(data_dir + 'Weights.npy'), dtype=torch.float)
            self.len = self.Xdata.shape[0]
        elif name == 'test':
            if task_name is None:
                self.Xdata = Xdata[train_num:]
                self.Ldata = Ldata[train_num:]
                self.Pdata = Pdata[train_num:]
            else:
                self.Xdata = Xdata[pos_id[train_pos_num:] + neg_id[train_neg_num:]]
                self.Ldata = Ldata[pos_id[train_pos_num:] + neg_id[train_neg_num:]]
                self.Pdata = Pdata[pos_id[train_pos_num:] + neg_id[train_neg_num:], task_num]
            self.len = self.Xdata.shape[0]
        elif name == 'unlabeled':
            self.Xdata = Xdata
            self.Ldata = Ldata
            if task_name is None:
                self.Pdata = Pdata
            else:
                self.Pdata = Pdata[:, task_num]
            self.len = self.Xdata.shape[0]

    def __getitem__(self, index):
        return self.Xdata[index], self.Ldata[index], self.Pdata[index]

    def __len__(self):
        return self.len


def load_alltasks_tox21_dataset(data_dir):
    # process_tox21_smiles_data(data_dir)
    train_data = UserDataset('data/', 'train')
    test_data = UserDataset('data/', 'test')
    unlabeled_data = UserDataset('data/', 'unlabeled')
    return train_data, test_data, train_data


def load_tox21_dataset(data_dir, task_name):
    # process_tox21_smiles_data(data_dir)
    train_data = UserDataset('data/', 'train', task_name)
    test_data = UserDataset('data/', 'test', task_name)
    unlabeled_data = UserDataset('data/', 'unlabeled', task_name)
    return train_data, test_data, unlabeled_data


def load_qm9_dataset(data_dir, task_no):
    # process_qm9_smiles_data(data_dir)
    train_data = QM9Dataset('data/qm9/', 'train', task_no)
    test_data = QM9Dataset('data/qm9/', 'test', task_no)
    all_data = QM9Dataset('data/qm9/', 'all', task_no)
    # unlabeled_data = QM9Dataset('data/qm9/', 'unlabeled', task_no)
    return train_data, test_data, all_data


if __name__ == '__main__':
    process_qm9_smiles_data('data/qm9.csv')
    # process_tox21_smiles_data('data/tox21.csv')
    # train_data = UserDataset('data/', 'train')
    # test_data = UserDataset('data/', 'test')

    # unlabeled_data = UserDataset('data/', 'unlabeled', 'NR-Aromatase')
    # neg, pos, na = 0, 0, 0
    # for _, _, label in unlabeled_data:
    #
    #     if label == 0: neg += 1
    #     elif label == 1: pos += 1
    #     else:
    #         na += 1
    # print(neg, pos, na)
