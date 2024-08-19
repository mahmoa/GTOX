import torch
import pandas as pd
import torch_geometric.utils as pyg_utils
import numpy as np

def get_data(as_df=False):
    '''
    
    '''
    if as_df:
        return preprocess_dataset()
    else:
        return df_to_graph(preprocess_dataset())

def preprocess_dataset():
    '''
    
    '''
    data = pd.read_csv('/Users/ali/Documents/portfolio_projects/GTOX/data/tox21_processed_resampled1500.csv')
    # Remove any column with incomplete toxicity data
    data.dropna(inplace=True)
    try:
        data['toxicity'] = data[['NR-AR',
                            'NR-AR-LBD',
                            'NR-AhR', 'NR-Aromatase',
                            'NR-ER', 'NR-ER-LBD',
                            'NR-PPAR-gamma',
                            'SR-ARE', 'SR-ATAD5',
                            'SR-HSE','SR-MMP','SR-p53']].any(axis=1).astype(int)
    except KeyError:
        pass

    cleaned_data = data[['smiles', 'toxicity']]
    return cleaned_data.reset_index()

def remove_zero_rows(df, x):
    '''
    remove x number of rows where the value of 'toxicity' is zero
    '''
    # Find the indices of rows where the column value is 0
    zero_indices = df[df['toxicity'] == 0].index
    
    # Ensure we don't try to remove more rows than exist
    if len(zero_indices) < x:
        x = len(zero_indices)
    
    # Randomly select `x` indices to drop
    indices_to_drop = np.random.choice(zero_indices, x, replace=False)
    
    # Drop those rows from the DataFrame
    df_dropped = df.drop(indices_to_drop)
    
    return df_dropped

def df_to_graph(data, EF=True):
    '''
    
    '''
    # Construct graph representations using PyG built in utility function
    graphs = [pyg_utils.smiles.from_smiles(data['smiles'][i]) for i in range(len(data))]
    
    # Declare x and y in the PyG dataset of graphs
    for i in range(len(data)):
        graphs[i].x = graphs[i].x.float()
        graphs[i].y = torch.Tensor([data['toxicity'][i]])

    # Incorporate edge features by pooling 
    if EF:
        for mol in graphs:
            if (len(mol.edge_index) == 0) or len(mol.edge_attr) == 0:
                # no edge features, concat 0s
                mol.x = torch.cat((mol.x, torch.zeros((len(mol.x), 3))), dim=1)
                continue
            pooled_edge_features = []
            for atm_idx in range(len(mol.x)):
                sum_edge_feature = torch.zeros(len(mol.edge_attr[0]))
                for i, start_atm_idx in enumerate(mol.edge_index[0]):
                    if start_atm_idx == atm_idx:
                        sum_edge_feature = torch.add(sum_edge_feature, mol.edge_attr[i])
                pooled_edge_features.append(sum_edge_feature)
            mol.x = torch.cat((mol.x, torch.stack(pooled_edge_features)), dim=1)
        return graphs
    else:
        return graphs