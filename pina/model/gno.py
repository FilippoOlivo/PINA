import torch
from torch_geometric.nn.conv import GraphConv

class GNO(torch.nn.Module):
    def __init__(self, lifting_operator, projection_operator):
        super().__init__()
        self.lifting_operator = lifting_operator
        self.projection_operator = projection_operator
        self.conv = GraphConv(in_channels=128, out_channels=128)
        self.func = torch.nn.Tanh()

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        x = self.lifting_operator(x)
        x = self.conv(x, edge_index)
        x = self.func(x)
        x = self.projection_operator(x)
        return x








