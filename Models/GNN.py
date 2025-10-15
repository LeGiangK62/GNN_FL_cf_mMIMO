import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, Dropout, ReLU
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.utils import dropout_node


def MLP(channels, batch_norm=True, dropout_prob=0):
    layers = []
    for i in range(1, len(channels)):
        layers.append(Seq(Lin(channels[i - 1], channels[i])))
        if batch_norm:
            layers.append(BN(channels[i]))
        if dropout_prob:
            layers.append(Dropout(dropout_prob))  # Add dropout after batch norm or activation
        layers.append(ReLU())
        # layers.append(LeakyReLU(negative_slope=0.1))
    # layers.append(Dropout(0.3))

    return Seq(*layers)
    
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
    
# Heterogeneous GNN
class APConvLayer(MessagePassing):
    def __init__(
            self,
            src_dim_dict,
            edge_dim,
            out_channel,
            init_channel,
            metadata,
            **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.metadata = metadata
        self.src_init_dict = init_channel
        self.edge_init = edge_dim
        self.out_channel = out_channel
        self.src_dim_dict = src_dim_dict

        self.msg = nn.ParameterDict()
        self.upd = nn.ParameterDict()
        
        hidden = out_channel * 2
        for edge_type in metadata:
            src_type, _, dst_type = edge_type
            src_dim = src_dim_dict[src_type]
            dst_dim = src_dim_dict[dst_type]
            src_init = init_channel[src_type]
            dst_init = init_channel[dst_type]
            self.msg[src_type] = MLP([src_dim + edge_dim, hidden , out_channel], batch_norm=True, dropout_prob=0.1)
            self.upd[dst_type] = MLP([out_channel + dst_dim, hidden, out_channel - dst_init], batch_norm=True, dropout_prob=0.1)

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.msg)
        glorot(self.upd)

    def forward(
            self,
            x_dict,
            edge_index_dict,
            edge_attr_dict
    ):
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type not in self.metadata: continue;
            src_type, _, dst_type = edge_type

            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]

            # Node update
            out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr_dict[edge_type], edge_type=edge_type)
            tmp = torch.cat([x_dst, out], dim=1)
            tmp = self.upd[dst_type](tmp)
            src_init_dim = self.src_init_dict[dst_type]
            if self.src_dim_dict[dst_type] == self.out_channel:
                tmp = tmp + 0.1 * x_dst[:,src_init_dim:]
            x_dict[dst_type] = torch.cat([x_dst[:,:src_init_dim], tmp], dim=1)
        return x_dict, edge_attr_dict

    def message(self, x_j, x_i, edge_attr, edge_type):
        # x_j: source node
        # x_i: destination node
        src_type, _, dst_type = edge_type
        out = torch.cat([x_j, edge_attr], dim=1)
        out = self.msg[src_type](out)
        return out



class APHetNet(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, num_layers=0, hid_layers=4):
        super(APHetNet, self).__init__()
        src_dim_dict = dim_dict

        self.ue_dim = src_dim_dict['UE']
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']

        self.convs = torch.nn.ModuleList()
        # First Layer to update RRU
        self.convs.append(
            APConvLayer(
                {'UE': self.ue_dim, 'AP': self.ap_dim},
                self.edge_dim,
                out_channels, src_dim_dict,
                [('UE', 'up', 'AP')]
            )
        )
        
        self.convs.append(
            APConvLayer(
                {'UE': self.ue_dim, 'AP': out_channels},
                self.edge_dim,
                out_channels, src_dim_dict,
                [('AP', 'down', 'UE')]
            )
        )
        for _ in range(num_layers):
            conv = APConvLayer({'UE': out_channels, 'AP': out_channels}, self.edge_dim, out_channels, src_dim_dict, [('UE', 'up', 'AP'), ('AP', 'down', 'UE')])
            self.convs.append(conv)


        hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        self.power = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1)
        self.power = nn.Sequential(*[self.power, Seq(Lin(hid, 1)), Sigmoid()])
        
        self.AP_gen = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1)
        self.AP_gen = nn.Sequential(*[self.AP_gen, Seq(Lin(hid, 1)), Sigmoid()])


    def forward(self, batch):
        x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict
        for conv in self.convs:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        dl_power = self.power(x_dict['UE'])
        x_dict['UE'] = torch.cat([x_dict['UE'][:,:self.ue_dim], dl_power], dim=1)
        x_dict['AP'] = self.AP_gen(x_dict['AP'])

        return x_dict, edge_attr_dict, edge_index_dict