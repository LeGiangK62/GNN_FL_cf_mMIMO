import torch
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias = True), ReLU())#, BN(channels[i]))
        for i in range(1, len(channels))
    ])
    
    
class APConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(APConv, self).__init__(aggr='max', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2

    def reset_parameters(self):
        reset(self.mlp1)
        reset(self.mlp2)

    def update(self, aggr_out, x):
        # print(f"aggr_out: {aggr_out}")
        tmp = torch.cat([x, aggr_out], dim=1)
        # print(f"tmp: {tmp}")
        comb = self.mlp2(tmp)
        # print(f"comb: {comb}")
        # print(f"torch.cat([x[:,:-1],comb],dim=1): {torch.cat([x[:,:-1],comb],dim=1)}")
        return torch.cat([x[:,:-1],comb],dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr) # propagate() internally call message(), aggregate() and update()
    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        # print(f"tmp: {tmp}")
        agg = self.mlp1(tmp)
        # print(f"agg: {agg}")
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1,self.mlp2)

class APNet(torch.nn.Module):
    def __init__(self, node_dim=11, edge_dim=2, hidden_dim=32):
        super(APNet, self).__init__()

        self.mlp1 = MLP([node_dim+edge_dim, hidden_dim, hidden_dim]) # 14 = 2+12
        self.mlp2 = MLP([node_dim+hidden_dim, hidden_dim]) # 43 = 32+12
        self.mlp2 = Seq(*[self.mlp2,Seq(Lin(hidden_dim, 1, bias = True), ReLU())]) #Sigmoid()
        self.conv = APConv(self.mlp1,self.mlp2)
        
        self.power = MLP([node_dim, hidden_dim, 1])

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x = self.conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
        x = self.conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
        x = self.conv(x = x, edge_index = edge_index, edge_attr = edge_attr)
        out = self.power(x)
        return out
    
    