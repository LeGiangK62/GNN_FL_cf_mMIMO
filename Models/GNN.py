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
            layers.append(LayerNorm(channels[i]))
        if dropout_prob:
            layers.append(Dropout(dropout_prob))  # Add dropout after batch norm or activation
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
            
            self.edge_upd[short_edge_type] = MLP(
                [sum(src_dim_dict.values()) + edge_dim, out_channel - self.edge_init], 
                batch_norm=False, dropout_prob=0.1
            )

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.msg)
        reset(self.upd)
        reset(self.edge_upd)

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
            

            # Node update                
            msg = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, edge_type=edge_type)
            tmp = torch.cat([x_dst, msg], dim=1)
            tmp = self.upd[dst_type](tmp)
            src_init_dim = self.src_init_dict[dst_type]
            if self.src_dim_dict[dst_type] == self.out_channel:
                tmp = tmp +  x_dst[:,src_init_dim:] # * self.gamma[dst_type]
            x_dict[dst_type] = torch.cat([x_dst[:,:src_init_dim], tmp], dim=1)
            # Edge update
            edge_attr_dict[edge_type] = self.edge_updater(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, edge_type=edge_type)

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
        out = self.edge_upd[short_edge_type](tmp)
        
        if self.out_channel == self.edge_init:
            out = out + edge_attr # * self.gamma_edge
        out = torch.cat([edge_attr[:,:self.edge_init], out], dim=1)
        return out



# Centralized GNN

class APHetNet(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, num_layers=0, hid_layers=4, isDecentralized=False):
        super(APHetNet, self).__init__()
        src_dim_dict = dim_dict.copy()

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
                [('UE', 'up', 'AP')],
            )
        )
        
        self.convs.append(
            APConvLayer(
                {'UE': self.ue_dim, 'AP': out_channels},
                self.edge_dim,
                out_channels, src_dim_dict,
                [('AP', 'down', 'UE')],
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


        hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
        self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        self.power_edge = nn.Sequential(
            *[
                self.power_edge, Seq(Lin(hid, 1)), 
            ]
        )
    
        
    def forward(self, batch):
        x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict
        for conv in self.convs:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        
        edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
        edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
            [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
            dim=1
        )

        return x_dict, edge_attr_dict, edge_index_dict

# FL
class APHetNetFL(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, aug_feat_dim=3, num_layers=0, hid_layers=4, isDecentralized=False):
        super(APHetNetFL, self).__init__()
        src_dim_dict = dim_dict.copy()
        

        self.ue_dim = src_dim_dict['UE']
        self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']


        self.convs_raw = self.create_conv_block(out_channels, src_dim_dict, num_layers)
        self.convs_aug = self.create_conv_block(out_channels, src_dim_dict, num_layers)  


        ##
        src_dim_dict_gap = dim_dict.copy()
        src_dim_dict_gap['GAP'] = 0
        src_dim_dict_gap['AP'] = 0
        src_dim_dict_gap['edge'] = 0
        self.convs_gap = torch.nn.ModuleList()
            
        self.convs_gap.append(APConvLayer(
            {'GAP': out_channels, 'AP': self.ap_dim }, 
            out_channels,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP')]
        ))
        
        for _ in range(num_layers):
            self.convs_gap.append(APConvLayer(
                {'GAP': out_channels, 'AP': out_channels }, 
                out_channels,
                out_channels, src_dim_dict_gap,
                [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
            ))

        hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
        self.ue_encoder_raw = MLP([self.ue_dim, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
        self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
        self.ap_encoder_raw = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1) 
        self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        self.power_edge = nn.Sequential(
            *[
                self.power_edge, Seq(Lin(hid, 1)), 
            ]
        )

    def create_conv_block(self, out_channels, src_dim_dict, num_layers):
            layers = torch.nn.ModuleList()
            
            # # First Layer to update RRU
            layers.append(APConvLayer(
                {'UE': out_channels, 'AP': out_channels}, 
                self.edge_dim,
                out_channels, src_dim_dict,
                [('UE', 'up', 'AP')]
            ))
            
            layers.append(APConvLayer(
                {'UE': out_channels, 'AP': out_channels},
                self.edge_dim,
                out_channels, src_dim_dict,
                [('AP', 'down', 'UE')]
            ))
            
            # Multiple conv layer for AP - UE
            for _ in range(num_layers):
                layers.append(APConvLayer(
                    {'UE': out_channels, 'AP': out_channels}, 
                    out_channels, out_channels, src_dim_dict, 
                    [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
                ))
            return layers
    
        
    def forward(self, batch, isRawData=False):
        x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict
        
        if isRawData:
            # The first round
            aug_ue = self.ue_encoder_raw(x_dict['UE'] )
            x_dict['UE'] = torch.cat(
                [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
                dim=1
            )
            aug_ap = self.ap_encoder_raw(x_dict['AP'] )
            x_dict['AP'] = aug_ap
            # for conv in self.convs_raw:
            #     x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        else:
            aug_ue = self.ue_encoder_aug(x_dict['UE'] )
            x_dict['UE'] = torch.cat(
                [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
                dim=1
            )
            for conv in self.convs_gap:
                x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        for conv in self.convs_aug:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        if not isRawData:
            edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
            edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
                [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
                dim=1
            )

        return x_dict, edge_attr_dict, edge_index_dict