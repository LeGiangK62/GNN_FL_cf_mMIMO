import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, LayerNorm, Dropout, GELU, LeakyReLU
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.utils import dropout_node, dropout_edge
from torch_geometric.nn import GraphNorm

def MLP(channels, batch_norm=False, dropout_prob=0):
    layers = []
    for i in range(1, len(channels)):
        layers.append(Seq(Lin(channels[i - 1], channels[i])))
        if batch_norm:
            # layers.append(BN(channels[i]))
            layers.append(LayerNorm(channels[i]))
        if dropout_prob:
            layers.append(Dropout(dropout_prob))  # Add dropout after batch norm or activation
        # layers.append(ReLU())
        # layers.append(GELU())
        # layers.append(nn.SiLU()) # Shit
        layers.append(LeakyReLU(negative_slope=0.1))
    return Seq(*layers)
    
    
# Heterogeneous GNN
class APConvLayer(MessagePassing):
    def __init__(
            self,
            src_dim_dict,
            edge_dim,
            out_channel,
            init_channel,
            metadata,
            drop_p=0,
            **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.metadata = metadata
        self.src_init_dict = init_channel
        self.edge_init = init_channel['edge']
        self.out_channel = out_channel
        self.src_dim_dict = src_dim_dict
        self.drop_prob = drop_p

        self.msg = nn.ModuleDict() 
        self.upd = nn.ModuleDict() 
        self.edge_upd = nn.ModuleDict() 
        
        self.gamma = nn.ParameterDict()
        self.gamma_edge = nn.ParameterDict()
        
        hidden = out_channel//2
        for edge_type in metadata:
            src_type, short_edge_type, dst_type = edge_type
            src_dim = src_dim_dict[src_type]
            dst_dim = src_dim_dict[dst_type]
            src_init = init_channel[src_type]
            dst_init = init_channel[dst_type]
            self.msg[src_type] = MLP(
                [src_dim + edge_dim, hidden], 
                batch_norm=False, dropout_prob=0.1
            )
            self.upd[dst_type] = MLP(
                [hidden + dst_dim, out_channel - dst_init], 
                batch_norm=False, dropout_prob=0.1
            )
            
            # self.gamma[dst_type] = nn.Parameter(torch.full((out_channel - dst_init,), 1e-3))
            
            self.edge_upd[short_edge_type] = MLP(
                [sum(src_dim_dict.values()) + edge_dim, out_channel - self.edge_init], 
                batch_norm=False, dropout_prob=0.1
            )
        # self.gamma_edge = nn.Parameter(torch.full((out_channel - self.edge_init,), 1e-3))

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.msg)
        reset(self.upd)
        # reset(self.gamma)
        reset(self.edge_upd)
            # reset(self.gamma_edge)

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
            
            edge_attr = edge_attr_dict[edge_type]
            
            # if self.training and self.drop_prob > 0:
            #     # DropEdge
            #     edge_index, edge_mask  = dropout_edge(
            #         edge_index, p=self.drop_prob,
            #         training=True
            #     )
            #     edge_attr = edge_attr_dict[edge_type][edge_mask]
            # else:
            #     edge_attr = edge_attr_dict[edge_type]
            #     edge_mask = torch.ones(
            #         edge_attr.size(0),
            #         dtype=torch.bool,
            #         device=edge_attr.device,
            #     )

            # Node update
            
            # if edge_type == ('UE', 'interfere', 'UE'):
            #     print(src_type)
            #     print(dst_type)
            #     print(x_src.shape)
            #     print(x_dst.shape)
            #     print(edge_attr.shape)
            #     print(edge_index)
            #     print(edge_index.shape) # Must be the problem
                
            msg = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, edge_type=edge_type)
            tmp = torch.cat([x_dst, msg], dim=1)
            tmp = self.upd[dst_type](tmp)
            src_init_dim = self.src_init_dict[dst_type]
            if self.src_dim_dict[dst_type] == self.out_channel:
                tmp = tmp +  x_dst[:,src_init_dim:] # * self.gamma[dst_type]
            x_dict[dst_type] = torch.cat([x_dst[:,:src_init_dim], tmp], dim=1)
            
            
            
            # Edge update
            # try:
            edge_attr_dict[edge_type] = self.edge_updater(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, edge_type=edge_type)
            # except RuntimeError as e:
            #     print(e)
            #     print(edge_type)
            #     print(x_src.shape)
            #     print(x_dst.shape)
            # if self.training and self.drop_prob > 0:
            #     edge_attr_dict[edge_type][edge_mask,:] = self.edge_updater(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)
            # else:
            #     edge_attr_dict[edge_type] = self.edge_updater(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)
        return x_dict, edge_attr_dict

    def message(self, x_j, x_i, edge_attr, edge_type):
        # x_j: source node
        # x_i: destination node
        src_type, _, dst_type = edge_type
        out = torch.cat([x_j, edge_attr], dim=1)
        out = self.msg[src_type](out)
        return out

    def edge_update(self, x_j, x_i, edge_attr, edge_type):
        _, short_edge_type, _ = edge_type
        tmp = torch.cat([x_j, edge_attr, x_i], dim=1)
        try:
            out = self.edge_upd[short_edge_type](tmp)
        except:
            print(edge_type)
            print(x_j.shape)
            print(x_i.shape)
            print(edge_attr.shape)
            print(self.src_dim_dict)
            prin
        
        if self.out_channel == self.edge_init:
            out = out + edge_attr # * self.gamma_edge
        out = torch.cat([edge_attr[:,:self.edge_init], out], dim=1)
        return out


class APHetNet(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, num_layers=0, hid_layers=4, isDecentralized=False):
        super(APHetNet, self).__init__()
        src_dim_dict = dim_dict.copy()

        self.ue_dim = src_dim_dict['UE']
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']
        
        ## The dummy 
        # src_dim_dict['edge'] = dim_dict['edge'] - 1 # dummy power

        self.convs = torch.nn.ModuleList()        
        # First Layer to update RRU
        self.convs.append(
            APConvLayer(
                {'UE': self.ue_dim, 'AP': self.ap_dim},
                self.edge_dim,
                out_channels, src_dim_dict,
                [('UE', 'up', 'AP')],
                # drop_p=0
            )
        )
        
        self.convs.append(
            APConvLayer(
                {'UE': self.ue_dim, 'AP': out_channels},
                self.edge_dim,
                out_channels, src_dim_dict,
                [('AP', 'down', 'UE')],
                # drop_p=0
            )
        )
        
        # Layers for Global AP
        if isDecentralized:
            src_dim_dict_gap = dim_dict.copy()
            src_dim_dict_gap['edge'] = dim_dict['GAP_edge']
            
            src_dim_dict_ue = dim_dict.copy()
            src_dim_dict_ue['edge'] = dim_dict['UE_inteference_edge']
            self.gap_dim = out_channels # src_dim_dict['GAP']
            self.gap_edge_dim = src_dim_dict['GAP_edge']
            self.ue_inteference_edge_dim = src_dim_dict['UE_inteference_edge']
            self.convs.append(
                APConvLayer(
                    {'GAP': self.gap_dim, 'UE': out_channels},
                    self.gap_edge_dim,
                    out_channels, src_dim_dict_gap,
                    [('UE', 'g_up', 'GAP')],
                )
            )
            
            self.convs.append(
                APConvLayer(
                    {'GAP': out_channels, 'UE': out_channels},
                    self.gap_edge_dim,
                    out_channels, src_dim_dict_gap,
                    [('GAP', 'g_down', 'UE')],
                )
            )
            
            self.convs.append(
                APConvLayer(
                    {'UE': out_channels, 'UE_dummy': out_channels},
                    self.ue_inteference_edge_dim,
                    out_channels, src_dim_dict_ue,
                    [('UE', 'interfere', 'UE')],
                )
            )
    
        
        # Multiple conv layer for AP - UE
        for _ in range(num_layers):
            conv = APConvLayer(
                {'UE': out_channels, 'AP': out_channels}, 
                out_channels, out_channels, src_dim_dict, 
                [('UE', 'up', 'AP'), ('AP', 'down', 'UE')],
                # drop_p=0.2
            )
            self.convs.append(conv)
            if isDecentralized:
                conv = APConvLayer(
                    {'GAP': out_channels, 'UE': out_channels}, 
                    out_channels, out_channels, src_dim_dict, 
                    [('UE', 'g_up', 'GAP'), ('GAP', 'g_down', 'UE')],
                    # drop_p=0.2
                )
                self.convs.append(conv)


        hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
        self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        self.power_edge = nn.Sequential(
            *[
                self.power_edge, Seq(Lin(hid, 1)), 
            ]
        )
        

        # self.ap_gate = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        # self.ap_gate = nn.Sequential(
        #     *[
        #         self.ap_gate, Seq(Lin(hid, 1)), 
        #     ]
        # )
            
        
    def forward(self, batch):
        x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict
        for conv in self.convs:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        
        edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
        # edge_power = torch.exp(edge_power)
        edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
            [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
            dim=1
        )
        # x_dict['AP']  = self.ap_gate(x_dict['AP'])


        return x_dict, edge_attr_dict, edge_index_dict