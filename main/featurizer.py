from typing import Union, List
from rdkit import Chem
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def SMILES_to_mol(smiles):
    '''
    Convert SMILES string into an RDKit mol object
    Parameters
        -
    Returns
        -
    '''
    if len(smiles) == 1:
        return Chem.MolFromSmiles(smiles)
    elif len(smiles) > 1:
        return [Chem.MolFromSmiles(smile) for smile in smiles]
    else:
        raise TypeError()
    
def SMILES_to_pygraph(smiles):
    '''
    
    '''

    if len(smiles) == 1:
        
        assert isinstance(smiles, Chem.mol)
        mol_obj = SMILES_to_mol(smiles)
            

        return
    elif len(smiles) > 1:

        for i in range(0, len(smiles)):
            pass

        return
    else:
        raise TypeError()

    

def node_features(molecule, df = True):
    '''
    Extract node features from RDkit molecule objects
    Args
        -
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
        row.append(atom.GetAtomicNum())
        row.append(atom.GetChiralTag())
        row.append(atom.GetHybridization())
        row.append(atom.GetTotalNumHs())
        row.append(atom.GetIsAromatic())
        row.append(atom.IsInRing())
        row.append(atom.GetTotalDegree())
        
        # Concatenate the row for the given atom
        #features = pd.concat([pd.DataFrame([row], columns=features.columns), df], ignore_index=True)
        features = pd.concat([pd.DataFrame([row], columns=feature_list), features], ignore_index=True)

    if df:
        return features
    else:
        return

def edge_features(molecules: list, one_hot = True, df = True):
    '''
    Extract edge features from RDkit molecule objects
    '''
    indices, attribs = [], []

    feature_list = ['bond_type', 'cyclic', 'conjugated', 'stereo']
    bondtype_list = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
    stereo_list = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
    features = pd.DataFrame(columns=feature_list)

    bt_encoder = OneHotEncoder()
    stereo_encoder = OneHotEncoder()

    for bond in molecules.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row = []
        row.append(str(bond.GetBondType()))
        row.append(str(bond.IsInRing()))
        row.append(int(bond.GetIsConjugated()))
        row.append(str(bond.GetStereo()))
        # indexes and attributes for pytorch tensors
        indices += [[i,j], [j, i]]
        attribs += [row, row]
        features = pd.concat([pd.DataFrame([row], columns=feature_list), features], ignore_index=True)

    if one_hot:
        features = pd.concat([features, pd.get_dummies(features['bond_type'])], axis=1)
        features = pd.concat([features, pd.get_dummies(features['stereo'])], axis=1)
        features.drop(['bond_type', 'stereo'], axis=1, inplace=True)

    if df:
        return features
    else:
        return