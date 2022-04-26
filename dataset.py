import os.path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]  # ex: if x is allowable_set[0] return [True, False, False, ..., False]


def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} is not in allowable set{allowable_set}")
    return [x == s for s in allowable_set]


def atom_attr(mol, explicit_H=True, use_chirality=True):
    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        results = onehot_encoding_unk(atom.GetSymbol(), ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other']) + \
            onehot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            onehot_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
        if not explicit_H:
            results += onehot_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results += onehot_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results += [0, 0] + [atom.HasProp('_ChiralityPossible')]
        feat.append(results)
    return np.array(feat)



def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats += onehot_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])
    return np.array(index), np.array(feat)


class MultiDataset(InMemoryDataset):

    def __init__(self, root, dataset, tasks, transform=None, pre_transform=None, pre_filter=None):
        self.tasks = tasks
        self.dataset = dataset
        self.weight = 0
        super(MultiDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.dataset}.csv']

    @property
    def processed_file_names(self):
        return [f'{self.dataset}.pt']

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles_list = df.smiles.values
        print("number of all smiles: ", len(smiles_list))
        remained_smiles = []
        canonical_smiles = []
        for smiles in smiles_list:
            try:
                remained_smiles.append(smiles)
                canonical_smiles.append(MolToSmiles(MolFromSmiles(smiles), isomericSmiles=True))  # isomericSmiles: include information about stereochemistry in the SMILES.
            except:
                print("not successfully processed smiles: ", smiles)
                pass
        print("number of successfully precessed smiles: ", len(canonical_smiles))

        df = df[df['smiles'].isin(remained_smiles)].reset_index()  # When we reset the index, the old index is added as a column, and a new sequential index is used.
        target = df[self.tasks].values  # tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        smiles_list = df.smiles.values
        data_list = []

        for i, smiles in enumerate(tqdm(smiles_list)):
            mol = MolFromSmiles(smiles)
            data = self.mol2graph(mol)
            if data is not None:
                label = target[i]
                label[np.isnan(label)] = 6 # why set nan label to 6 ???
                data.y = torch.LongTensor([label])
                data_list.append(data)

        if self.pre_filter is not None:  # A function that takes in an torch_geometric.data.Data object and returns a boolean value, indicating whether the data object should be included in the final dataset
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:  # A function/transform that takes in an torch_geometric.data.Data object and returns a transformed version
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def mol2graph(self, mol):
        if mol is None:
            return None
        node_attr = atom_attr(mol)
        edge_index, edge_attr = bond_attr(mol)
        pyg_data = Data(x=torch.FloatTensor(node_attr), edge_index=torch.LongTensor(edge_index).t(), edge_attr=torch.FloatTensor(edge_attr), y=None)
        return pyg_data

def load_dataset_random(path, dataset, seed, tasks=None):
    save_path = path + f'processed/train_test_{dataset}_seed_{seed}.ckpt'
    if os.path.isfile(save_path):
        train, test = torch.load(save_path)
        return train, test
    pyg_dataset = MultiDataset(root=path, dataset=dataset, tasks=tasks)
    df = pd.read_csv(os.path.join(path, f'raw/{dataset}.csv'))
    smiles_list = df.smiles.values
    print("number of all smiles: ", len(smiles_list))
    remained_smiles = []
    canonical_smiles = []
    for smiles in smiles_list:
        try:
            remained_smiles.append(smiles)
            canonical_smiles.append(MolToSmiles(MolFromSmiles(smiles),
                                                isomericSmiles=True))  # isomericSmiles: include information about stereochemistry in the SMILES.
        except:
            print("not successfully processed smiles: ", smiles)
            pass
    print("number of successfully precessed smiles: ", len(canonical_smiles))

    df = df[df['smiles'].isin(remained_smiles)].reset_index()
    if dataset == 'tox21':
        train_size = int(0.8 * len(pyg_dataset))
        test_size = len(pyg_dataset) - train_size
        pyg_dataset = pyg_dataset.shuffle()
        train, test = pyg_dataset[:train_size], pyg_dataset[train_size:]

        # # in each classfication work, add a weight base on the number of positive and negative data
        # weights = []
        # for i, task in enumerate(tasks):
        #     negative_df = df[df[task] == 0][['smiles', task]]
        #     positive_df = df[df[task] == 1][["smiles", task]]
        #     neg_len = len(negative_df)
        #     pos_len = len(positive_df)
        #     weights.append([(neg_len + pos_len) / neg_len, (neg_len + pos_len) / pos_len])
        # train.weight = weights

        torch.save([train, test], save_path)  # save to disk for loading next time without data processing
        return load_dataset_random(path, dataset, seed, tasks)
