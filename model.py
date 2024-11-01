"""
Module containing the Graph Neural Network model definition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(torch.nn.Module):
    """
    Graph Neural Network model for predicting material properties.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        """
        Initialize the GNN model.

        Args:
            input_dim (int): Input dimension of node features.
            hidden_dim (int): Hidden dimension of the GCN layers.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout rate.
        """
        super(GNN, self).__init__()
        self.convs = nn.ModuleList([GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the GNN model.

        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Graph connectivity in COO format.
            batch (Tensor): Batch vector, which assigns each node to a specific example.

        Returns:
            Tensor: Predicted property value.
        """
        mask = (x.sum(dim=-1) != 0)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x[mask], batch[mask])
        x = self.lin(x)
        return x.squeeze()