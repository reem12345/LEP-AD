from collections import OrderedDict
import json,pickle
import networkx as nx
import numpy as np
import torch
import pandas as pd
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import numpy as np
import torchvision.transforms as T
from torch_geometric.transforms import ToDense
RDLogger.DisableLog('rdApp.*')


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
    
# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def smile_to_graph(smile, return_smile = False):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    atom_names = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        h = atom.GetTotalNumHs()  # counts implicit+explicit H
    
        # --- labeling ---
        # special case: aromatic N with 1 H -> [nH]
        if sym == "N" and atom.GetIsAromatic() and h == 1:
            label = "[nH]"
        else:
            # your original rule: SymbolH / SymbolH2 / Symbol for 0 H
            if h == 0:
                label = sym
            elif h == 1:
                label = f"{sym}H"
            else:
                label = f"{sym}H{h}"
    
        atom_names.append(label)
    
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_edge_weight=[]
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
        mol_edge_weight.append([1])
    if return_smile: 
        return smile, c_size, features, atom_names, edge_index, mol_edge_weight
    else:
        return c_size, features, edge_index,mol_edge_weight

# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform= None,
                 pre_transform=None, smile_graph=None, target_key=None, target_rep=None, return_smile=False):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_rep, return_smile)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_rep, return_smile):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_len = len(xd)
        for i in range(data_len):
            entity1 = xd[i]
            labels = y[i]
            
            if entity1 in smile_graph.keys():
                c_size, features, edge_index,edge_weight = smile_graph[entity1]
            else:
                c_size, features, edge_index,edge_weight = smile_to_graph(entity1)

            if return_smile: 
                smile, c_size, features, atom_names, edge_index, mol_edge_weight = smile_to_graph(entity1, return_smile=return_smile)
                GCNData_mol = DATA.Data(x=torch.Tensor(np.array(features)), edge_weight = torch.Tensor(np.array(mol_edge_weight)),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels])
                                    )
            else:        
                GCNData_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                                        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                        y=torch.FloatTensor([labels])
                                        )
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
            
            if return_smile: 
                GCNData_mol.smile = smile
                GCNData_mol.atom_names = atom_names
                GCNData_mol.residue_names = target_key[i]
                
            data_list_mol.append(GCNData_mol)
            data_list_pro.append(torch.Tensor(target_rep[target_key[i]]).to("cpu"))
            if i%10000==0:
                print(i)
            
        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]


# initialize the dataset
class DTADataset_ct(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, y=None, transform= None,
                 pre_transform=None, smile_graph=None, target_key=None, target_rep=None, return_smile=False):
        super(DTADataset_ct, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, target_key, y, smile_graph, target_rep, return_smile)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, target_key, y, smile_graph, target_rep, return_smile):
        assert (len(xd) == len(target_key) and len(xd) == len(y)), 'The three lists must be the same length!'
        data_list_mol = []
        data_list_pro = []
        data_len = len(xd)
        for i in range(data_len):
            entity1 = xd[i]
            labels = y[i]
            
            if entity1 in smile_graph.keys():
                c_size, features, edge_index,edge_weight = smile_graph[entity1]
            else:
                c_size, features, edge_index,edge_weight = smile_to_graph(entity1)

            if return_smile: 
                smile, c_size, features, atom_names, edge_index, mol_edge_weight = smile_to_graph(entity1, return_smile=return_smile)
                GCNData_mol = DATA.Data(x=torch.Tensor(np.array(features)), edge_weight = torch.Tensor(np.array(mol_edge_weight)),
                                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                    y=torch.FloatTensor([labels])
                                    )
            else:        
                GCNData_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                                        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                        y=torch.FloatTensor([labels])
                                        )
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))
            
            
                
            
            data_pro_feat = DATA.Data( x = torch.Tensor( target_rep[target_key[i]] ).to("cpu").mean(0) )
            
            
            if return_smile: 
                GCNData_mol.smile = smile
                GCNData_mol.atom_names = atom_names
                data_pro_feat.residue_names = target_key[i]
                
            data_list_mol.append( GCNData_mol )
            data_list_pro.append( data_pro_feat )
            
            if i%10000==0:
                print(i)
            
        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
            
        self.data_mol = data_list_mol
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]
        
def collate(batch):
    graphs = Batch.from_data_list([item[0] for item in batch])
    #proteins = Batch.from_data_list([item[1] for item in batch])
    tensors = [item[1] for item in batch]
    tensors = torch.stack(tensors)
    return graphs, tensors
