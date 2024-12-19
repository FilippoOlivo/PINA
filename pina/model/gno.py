import torch
from .base_no import KernelNeuralOperator
from torch.nn import ReLU
from torch_geometric.nn.conv import GCNConv

class GraphNeuralKernel(torch.nn.Module):
    def __init__(self, hidden_dimension, func=None, n_layers=None):
        super().__init__()
        if func is None:
            func = ReLU()
        if n_layers is None:
            n_layers = 1
        self.layers = torch.nn.ModuleList(
            [GCNConv(hidden_dimension, hidden_dimension)
             for _ in range(n_layers)]
        )
        self.activation = func

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            x = self.activation(x)
        return x


class GNO(KernelNeuralOperator):
    def __init__(self,
                 lifting_operator,
                 projection_operator,
                 hidden_dimension=None,
                 func=None,
                 n_layers=None):
        if hidden_dimension is None and isinstance(lifting_operator, torch.nn.Linear):
            hidden_dimension = lifting_operator.out_features
        integral_kernels = GraphNeuralKernel(hidden_dimension=hidden_dimension,
                                             func=func,
                                             n_layers=n_layers)
        super().__init__(lifting_operator=lifting_operator,
                       projection_operator=projection_operator,
                       integral_kernels=integral_kernels)

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        x = self.lifting_operator(x)
        x = self.integral_kernels(x, edge_index)
        x = self.projection_operator(x)
        return x







