import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch.nn import Linear

embedding_size = 64

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(50)

        # Construct model layers
        # 3 convolutional layers
        self.initial_conv = GCNConv(12, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        x = self.initial_conv(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
          
        # Global Pooling (stack different aggregations)
        # x = torch.cat([gmp(x, batch_index), 
        #                   gap(x, batch_index)], dim=1)
        x = gap(x, batch_index)

        # Fully connected layer
        x = self.fc(x)

        #out = self.out(x)
        return torch.sigmoid(x)
