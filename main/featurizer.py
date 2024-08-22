from typing import Union, List
from rdkit import Chem
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from torch_geometric.data import Data

def SMILES_to_mol(smiles):
    '''
    Convert SMILES string into an RDKit mol object
    Parameters
        -
    Returns
        -
    '''
    return Chem.MolFromSmiles(smiles)

    
def SMILES_to_pygraph(smiles_list: list):
    '''
    
    '''

    if len(smiles_list) == 1:
        
        #assert isinstance(smiles, Chem.mol)
        mol_obj = SMILES_to_mol(str(smiles_list[0]))
        
        x = node_features(mol_obj)
        edge_inx, edge_attribs = edge_features(molecules=mol_obj)
       
        return Data(x=x, edge_attr = edge_attribs, edge_index = edge_inx, smiles=smiles_list[0])
    
    elif len(smiles_list) > 1:
        data = []
        for i in range(0, len(smiles_list)):
            mol_obj = SMILES_to_mol(str(smiles_list[i]))
            x = node_features(mol_obj)
            edge_inx, edge_attribs = edge_features(molecules=mol_obj)
            data.append(Data(x=x, edge_attr = edge_attribs, edge_index = edge_inx, smiles=smiles_list[i]))

        return data
    else:
        raise TypeError()

    

def node_features(molecule, df = False):
    '''
    Extract node features from RDkit molecule objects
    Args
        - molecule: rdkit molecule object
        - df = False: return as pandas dataframe if true, otherwise return as tensors
    Returns
        -
    '''
    # Optionally, 1-hot encoding could be preferrable so that magnitudes are the same
    feature_list = ['atomic_number', 'chirality', 'hybridization',
                'num_h', 'aromatic', 'cyclic', 'degree']
    
    features = pd.DataFrame(columns=feature_list)
    # Iterate through each atom in the RDKit molecule
    for atom in molecule.GetAtoms():
        row = []
        row.append(int(atom.GetAtomicNum()))
        row.append(int(atom.GetChiralTag()))
        row.append(int(atom.GetHybridization()))
        row.append(int(atom.GetTotalNumHs()))
        row.append(int(atom.GetIsAromatic()))
        row.append(int(atom.IsInRing()))
        row.append(int(atom.GetTotalDegree()))
        
        # Concatenate the row for the given atom
        #features = pd.concat([pd.DataFrame([row], columns=features.columns), df], ignore_index=True)
        features = pd.concat([pd.DataFrame([row], columns=feature_list), features], ignore_index=True)

    if df:
        return features
    else:
        return torch.from_numpy(features.astype('float').to_numpy())

def edge_features(molecules: list, one_hot = True, df = False):
    '''
    Extract edge features from RDkit molecule objects
    Args
        - molecule: rdkit molecule object
        - df: return as pandas dataframe if true, otherwise return as tensors
    Returns
        -
    '''
    indices, attribs = [], []

    feature_list = ['bond_type', 'cyclic', 'conjugated', 'stereo']
    features = pd.DataFrame(columns=feature_list)

    for bond in molecules.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row = []
        row.append(str(bond.GetBondType()))
        row.append(int(bond.IsInRing()))
        row.append(int(bond.GetIsConjugated()))
        row.append(str(bond.GetStereo()))
        # indexes and attributes for pytorch tensors
        indices += [[i,j], [j, i]]
        attribs += [row, row]
        features = pd.concat([pd.DataFrame([row], columns=feature_list), features], ignore_index=True)

    if one_hot:
        features = encoder_util(features, 'stereo')
        features = encoder_util(features, 'bond_type')

    if df:
        return features
    else:
        edge_index = torch.tensor(indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(features.astype('float').to_numpy(), dtype=torch.long)

        return edge_index, edge_attr
    
def encoder_util(df, column):
    '''
    returns pandas dataframe with additional columns frome one-hot encoding
    '''
    
    bondtype_list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    stereo_list = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
    if column == 'stereo':
        enc = OneHotEncoder(sparse_output=False, categories=[stereo_list])
    elif column == 'bond_type':
        enc = OneHotEncoder(sparse_output=False, categories=[bondtype_list])
    
    data = df[column].to_numpy().reshape(-1,1)
    #enc = OneHotEncoder(sparse_output=False)
    encoded = pd.DataFrame(enc.fit_transform(data), columns=enc.get_feature_names_out())
    df = pd.concat([df, encoded],axis=1).drop(columns=column, axis=1)
    return df