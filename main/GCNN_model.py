import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch.nn import Linear

embedding_size = 64

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
        Class to build graph convolution neural network for molecular toxicity prediction
        args
            in_channels: number of features
            out_channels: 1 for classification
        '''
        super(GCN, self).__init__()
        torch.manual_seed(50)

        # Construct model
        # 3 convolutional layers
        self.initial_conv = GCNConv(in_channels, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.fc = nn.Linear(embedding_size, out_channels)

    def forward(self, x, edge_index, batch_index):
        # First convolution layer
        x = self.initial_conv(x, edge_index)
        # Activation between convolutions
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
          
        # Pooling
        x = gap(x, batch_index)

        # Fully connected layer for a final classification
        x = self.fc(x)

        return torch.sigmoid(x)
