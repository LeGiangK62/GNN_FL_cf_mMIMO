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
        # layers.append(GELU())
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

# class APHetNetFL_currentBest(nn.Module):
#     def __init__(self, metadata, dim_dict, out_channels, aug_feat_dim=3, num_layers=0, hid_layers=4, isDecentralized=False):
#         super(APHetNetFL, self).__init__()

#         GAP_init_dim = out_channels # + 3
#         GAP_edge_init_dim = out_channels
#         src_dim_dict = dim_dict.copy()
        

#         self.ue_dim = src_dim_dict['UE']
#         self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
#         self.ap_dim = src_dim_dict['AP']
#         self.edge_dim = src_dim_dict['edge']




#         ##
#         src_dim_dict_gap = dim_dict.copy()
#         src_dim_dict_gap['GAP'] = 0
#         src_dim_dict_gap['AP'] = src_dim_dict['AP']
#         src_dim_dict_gap['edge'] = 0
#         self.convs_gap = torch.nn.ModuleList()
            
#         self.convs_gap.append(APConvLayer(
#             {'GAP': GAP_init_dim, 'AP': self.ap_dim }, 
#             GAP_edge_init_dim,
#             out_channels, src_dim_dict_gap,
#             [('GAP', 'cross', 'AP')]
#         ))

#         self.convs_gap.append(APConvLayer(
#             {'GAP': GAP_init_dim, 'AP': out_channels }, 
#             GAP_edge_init_dim,
#             out_channels, src_dim_dict_gap,
#             [('AP', 'cross-back', 'GAP')]
#         ))
        
#         self.convs_gap.append(APConvLayer(
#             {'GAP': out_channels, 'AP': out_channels }, 
#             out_channels,
#             out_channels, src_dim_dict_gap,
#             [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
#         ))

#         # self.convs_gap.append(APConvLayer(
#         #     {'GAP': out_channels, 'UE': out_channels }, 
#         #     out_channels,
#         #     out_channels, src_dim_dict_gap,
#         #     [('GAP', 'g_down', 'UE'), ('UE', 'g_up', 'GAP')]
#         # ))

#         ##
#         # self.convs_gap_post = torch.nn.ModuleList()
            
#         # self.convs_gap_post.append(APConvLayer(
#         #     {'GAP': out_channels, 'AP': out_channels }, 
#         #     out_channels,
#         #     out_channels, src_dim_dict_gap,
#         #     [('GAP', 'cross', 'AP')]
#         # ))
        
#         # self.convs_gap_post.append(APConvLayer(
#         #     {'GAP': out_channels, 'AP': out_channels }, 
#         #     out_channels,
#         #     out_channels, src_dim_dict_gap,
#         #     [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
#         # ))


#         ##
#         # self.convs_pre = self.create_conv_block(out_channels, out_channels, self.edge_dim, out_channels, src_dim_dict, num_layers)  
#         self.convs_pre = self.create_conv_block(out_channels, out_channels, self.edge_dim, out_channels, src_dim_dict, num_layers-1)  
#         self.convs_aug = self.create_conv_block(out_channels, out_channels, out_channels, out_channels, src_dim_dict, num_layers-1)  
        
#         hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
#         self.ue_encoder_raw = MLP([self.ue_dim, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
#         self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
#         self.ap_encoder_raw = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1) 
#         self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
#         self.power_edge = nn.Sequential(
#             *[
#                 self.power_edge, Seq(Lin(hid, 1)), 
#             ]
#         )

#     def create_conv_block(self, ue_in, ap_in, edge, out_channels, src_dim_dict, num_layers):
#             layers = torch.nn.ModuleList()
            
#             # # First Layer to update RRU
#             layers.append(APConvLayer(
#                 {'UE': ue_in, 'AP': ap_in}, 
#                 edge,
#                 out_channels, src_dim_dict,
#                 [('UE', 'up', 'AP')]
#             ))
            
#             layers.append(APConvLayer(
#                 {'UE': ue_in, 'AP': out_channels},
#                 edge,
#                 out_channels, src_dim_dict,
#                 [('AP', 'down', 'UE')]
#             ))
            
#             # Multiple conv layer for AP - UE
#             for _ in range(num_layers):
#                 layers.append(APConvLayer(
#                     {'UE': out_channels, 'AP': out_channels}, 
#                     out_channels, out_channels, src_dim_dict, 
#                     [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
#                 ))
#             return layers
    
        
#     def forward(self, batch, isRawData=False):
#         x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict
        
#         if isRawData:
#             # The first round
#             aug_ue = self.ue_encoder_raw(x_dict['UE'] )
#             x_dict['UE'] = torch.cat(
#                 [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
#                 dim=1
#             )
#             aug_ap = self.ap_encoder_raw(x_dict['AP'] )
#             x_dict['AP'] = aug_ap
#         else:
#             aug_ue = self.ue_encoder_aug(x_dict['UE'] )
#             x_dict['UE'] = torch.cat(
#                 [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
#                 dim=1
#             )
#             for conv in self.convs_gap:
#                 x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

#         for conv in self.convs_pre:
#             x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

#         for conv in self.convs_aug:
#             x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

#         if not isRawData:
#             # for conv in self.convs_gap_post:
#             #     x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
#             # for conv in self.convs_aug:
#             #     x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
#             edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
#             edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
#                 [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
#                 dim=1
#             )

#         return x_dict, edge_attr_dict, edge_index_dict
    

# class APHetNetFL(nn.Module):
#     def __init__(self, metadata, dim_dict, out_channels, aug_feat_dim=3, num_layers=0, hid_layers=4, isDecentralized=False):
#         super(APHetNetFL, self).__init__()

#         GAP_init_dim = out_channels + 0# + 3
#         GAP_edge_init_dim = out_channels * 2 # * 3
#         GAP_UE_edge = out_channels
#         src_dim_dict = dim_dict.copy()
        

#         self.ue_dim = src_dim_dict['UE']
#         self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
#         self.ap_dim = src_dim_dict['AP']
#         self.edge_dim = src_dim_dict['edge']

#         self.out_channels = out_channels



#         ##
#         src_dim_dict_gap = dim_dict.copy()
#         src_dim_dict_gap['GAP'] = 0
#         src_dim_dict_gap['AP'] = src_dim_dict['AP']
#         src_dim_dict_gap['edge'] = 0
        

#         ##
#         # self.convs_pre = self.create_conv_block(out_channels, out_channels, self.edge_dim, out_channels, src_dim_dict, num_layers)  
#         # self.convs_pre = self.create_conv_block(out_channels, out_channels, self.edge_dim, out_channels, src_dim_dict, num_layers-1)  
#         # self.convs_aug = self.create_conv_block(out_channels, out_channels, out_channels, out_channels, src_dim_dict, num_layers-1)  

#         ### UE <-> AP
#         self.convs_pre = torch.nn.ModuleList()
            
#         self.convs_pre.append(APConvLayer(
#             {'UE': out_channels, 'AP': out_channels}, 
#             self.edge_dim,
#             out_channels, src_dim_dict,
#             [('UE', 'up', 'AP')]
#         ))

#         self.convs_pre.append(APConvLayer(
#             {'UE': out_channels, 'AP': out_channels}, 
#             self.edge_dim,
#             out_channels, src_dim_dict,
#             [('AP','down','UE')]
#         ))

#         #####

#         ### GAP <-> AP
#         self.convs_gap = torch.nn.ModuleList()

#         self.convs_gap.append(APConvLayer(
#             {'GAP': GAP_init_dim, 'AP': out_channels}, 
#             GAP_edge_init_dim,
#             out_channels, src_dim_dict_gap,
#             [('GAP', 'cross', 'AP')]
#         ))

#         self.convs_gap.append(APConvLayer(
#             {'GAP': GAP_init_dim, 'AP': out_channels }, 
#             GAP_edge_init_dim,
#             out_channels, src_dim_dict_gap,
#             [('AP', 'cross-back', 'GAP')]
#         ))
        
#         # for _ in range(num_layers): # too many layers break the model
#         self.convs_gap.append(APConvLayer(
#             {'GAP': out_channels, 'AP': out_channels }, 
#             out_channels,
#             out_channels, src_dim_dict_gap,
#             [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
#         ))

#         #####


#         # ### GAP <-> UE
#         # self.conv_gap_ue = APConvLayer(
#         #     {'GAP': out_channels, 'UE': out_channels},
#         #     GAP_UE_edge + 2,           # F+2
#         #     out_channels, src_dim_dict_gap,
#         #     [('GAP', 'serves', 'UE')]
#         # )
        
#         ###

#         ### AP -> UE, then AP <-> UE
#         self.convs_post = torch.nn.ModuleList()

#         for _ in range(num_layers):
#             self.convs_post.append(APConvLayer(
#                 {'UE': out_channels, 'AP': out_channels}, 
#                 out_channels, out_channels, src_dim_dict, 
#                 [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
#             ))

        
#         hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
#         # self.ue_encoder_raw = MLP([self.ue_dim, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
#         self.ue_encoder_raw = MLP([self.ue_dim, hid, self.ue_dim_aug], batch_norm=True, dropout_prob=0.1) 
#         self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
#         self.ap_encoder_raw = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1) 
#         self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
#         self.power_edge = nn.Sequential(
#             *[
#                 self.power_edge, Seq(Lin(hid, 1)), 
#             ]
#         )

#     def create_conv_block(self, ue_in, ap_in, edge, out_channels, src_dim_dict, num_layers):
#             layers = torch.nn.ModuleList()
            
#             # # First Layer to update RRU
#             layers.append(APConvLayer(
#                 {'UE': ue_in, 'AP': ap_in}, 
#                 edge,
#                 out_channels, src_dim_dict,
#                 [('UE', 'up', 'AP')]
#             ))
            
#             layers.append(APConvLayer(
#                 {'UE': ue_in, 'AP': out_channels},
#                 edge,
#                 out_channels, src_dim_dict,
#                 [('AP', 'down', 'UE')]
#             ))
            
#             # Multiple conv layer for AP - UE
#             for _ in range(num_layers):
#                 layers.append(APConvLayer(
#                     {'UE': out_channels, 'AP': out_channels}, 
#                     out_channels, out_channels, src_dim_dict, 
#                     [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
#                 ))
#             return layers
    
        
#     def forward(self, batch, isRawData=False):
#         x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict


#         aug_ap = self.ap_encoder_raw(x_dict['AP'] )
#         x_dict['AP'] = aug_ap
#         tmp = x_dict['UE'] 
#         if isRawData:
#             tmp = self.ue_encoder_raw(x_dict['UE'] )
#             # aug_ue = torch.zeros(x_dict['UE'].shape[0], self.out_channels - self.ue_dim, device=x_dict['UE'].device)
#         # else:
#         aug_ue = self.ue_encoder_aug(tmp)

#         x_dict['UE'] = torch.cat(
#             [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
#             dim=1
#         )

#         # UE -> AP
#         for conv in self.convs_pre:
#             x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

#         if not isRawData:
#             for conv in self.convs_gap:
#                 x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
#             # x_dict, edge_attr_dict = self.conv_gap_ue(x_dict, edge_index_dict, edge_attr_dict)
        

#         # AP <-> UE
#         for conv in self.convs_post:
#             x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

#         edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
#         # if not isRawData:
#         #     edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
#         #         [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
#         #         dim=1
#         #     )
#         # else:
#         edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
#             [edge_attr_dict[('AP', 'down', 'UE')][:,:-1], edge_power], 
#             dim=1
#         )

#         return x_dict, edge_attr_dict, edge_index_dict
    




# class APHetNetFL_sumrate(nn.Module):
#     def __init__(self, metadata, dim_dict, out_channels, aug_feat_dim=3, num_layers=0, hid_layers=4, isDecentralized=False):
#         super(APHetNetFL_sumrate, self).__init__()

#         GAP_init_dim = out_channels + 0# + 3
#         GAP_edge_init_dim = out_channels * 2 # * 3
#         GAP_UE_edge = out_channels
#         src_dim_dict = dim_dict.copy()
        

#         self.ue_dim = src_dim_dict['UE']
#         self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
#         self.ap_dim = src_dim_dict['AP']
#         self.edge_dim = src_dim_dict['edge']

#         self.out_channels = out_channels

#         ##
#         src_dim_dict_gap = dim_dict.copy()
#         src_dim_dict_gap['GAP'] = 0
#         src_dim_dict_gap['AP'] = src_dim_dict['AP']
#         src_dim_dict_gap['edge'] = 0

#         ### UE <-> AP
#         self.convs_pre = torch.nn.ModuleList()
            
#         self.convs_pre.append(APConvLayer(
#             {'UE': out_channels, 'AP': out_channels}, 
#             self.edge_dim,
#             out_channels, src_dim_dict,
#             [('UE', 'up', 'AP')]
#         ))

#         self.convs_pre.append(APConvLayer(
#             {'UE': out_channels, 'AP': out_channels}, 
#             self.edge_dim,
#             out_channels, src_dim_dict,
#             [('AP','down','UE')]
#         ))

#         #####

#         ### GAP <-> AP
#         self.convs_gap = torch.nn.ModuleList()

#         self.convs_gap.append(APConvLayer(
#             {'GAP': GAP_init_dim, 'AP': out_channels}, 
#             GAP_edge_init_dim,
#             out_channels, src_dim_dict_gap,
#             [('GAP', 'cross', 'AP')]
#         ))

#         self.convs_gap.append(APConvLayer(
#             {'GAP': GAP_init_dim, 'AP': out_channels }, 
#             GAP_edge_init_dim,
#             out_channels, src_dim_dict_gap,
#             [('AP', 'cross-back', 'GAP')]
#         ))
        
#         # for _ in range(num_layers): # too many layers break the model
#         self.convs_gap.append(APConvLayer(
#             {'GAP': out_channels, 'AP': out_channels }, 
#             out_channels,
#             out_channels, src_dim_dict_gap,
#             [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
#         ))

#         ### AP -> UE, then AP <-> UE
#         self.convs_post = torch.nn.ModuleList()

#         for _ in range(num_layers):
#             self.convs_post.append(APConvLayer(
#                 {'UE': out_channels, 'AP': out_channels}, 
#                 out_channels, out_channels, src_dim_dict, 
#                 [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
#             ))

        
#         hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
#         # self.ue_encoder_raw = MLP([self.ue_dim, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
#         self.ue_encoder_raw = MLP([self.ue_dim, hid, self.ue_dim_aug], batch_norm=True, dropout_prob=0.1) 
#         self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
#         self.ap_encoder_raw = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1) 
#         self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
#         self.power_edge = nn.Sequential(
#             *[
#                 self.power_edge, Seq(Lin(hid, 1)), 
#             ]
#         )

#     def create_conv_block(self, ue_in, ap_in, edge, out_channels, src_dim_dict, num_layers):
#             layers = torch.nn.ModuleList()
            
#             # # First Layer to update RRU
#             layers.append(APConvLayer(
#                 {'UE': ue_in, 'AP': ap_in}, 
#                 edge,
#                 out_channels, src_dim_dict,
#                 [('UE', 'up', 'AP')]
#             ))
            
#             layers.append(APConvLayer(
#                 {'UE': ue_in, 'AP': out_channels},
#                 edge,
#                 out_channels, src_dim_dict,
#                 [('AP', 'down', 'UE')]
#             ))
            
#             # Multiple conv layer for AP - UE
#             for _ in range(num_layers):
#                 layers.append(APConvLayer(
#                     {'UE': out_channels, 'AP': out_channels}, 
#                     out_channels, out_channels, src_dim_dict, 
#                     [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
#                 ))
#             return layers
    
        
#     def forward(self, batch, isRawData=False):
#         x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict


#         aug_ap = self.ap_encoder_raw(x_dict['AP'] )
#         x_dict['AP'] = aug_ap
#         tmp = x_dict['UE'] 
#         if isRawData:
#             tmp = self.ue_encoder_raw(x_dict['UE'] )
#         aug_ue = self.ue_encoder_aug(tmp)

#         x_dict['UE'] = torch.cat(
#             [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
#             dim=1
#         )

#         # UE -> AP
#         for conv in self.convs_pre:
#             x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

#         if not isRawData:
#             for conv in self.convs_gap:
#                 x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

#         # AP <-> UE
#         for conv in self.convs_post:
#             x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

#         edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
#         edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
#             [edge_attr_dict[('AP', 'down', 'UE')][:,:-1], edge_power], 
#             dim=1
#         )

#         return x_dict, edge_attr_dict, edge_index_dict
    


## ISAC models

class IsacConvLayer(MessagePassing):
    """
    Heterogeneous GNN conv layer for ISAC graphs.

    Key difference from APConvLayer: msg MLPs are keyed by short_edge_type
    (not src_type), so AP->UE (comm) and AP->SR (sensing) get separate MLPs
    even though both share 'AP' as the source node type.

    Args:
        src_dim_dict   : {node_type: int}       current node embedding widths
        edge_dim_dict  : {short_edge_type: int}  edge attr width per relation
        out_channel    : int
        init_channel   : {node_type: int}        raw node dims kept for residual
        edge_init_dict : {short_edge_type: int}  raw edge dims kept for residual
        metadata       : list of (src, rel, dst) tuples this layer handles
    """
    def __init__(
            self,
            src_dim_dict,
            edge_dim_dict,
            out_channel,
            init_channel,
            edge_init_dict,
            metadata,
            **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.metadata       = metadata
        self.src_init_dict  = init_channel
        self.edge_init_dict = edge_init_dict
        self.out_channel    = out_channel
        self.src_dim_dict   = src_dim_dict

        self.msg      = nn.ModuleDict()
        self.upd      = nn.ModuleDict()
        self.edge_upd = nn.ModuleDict()

        hidden = out_channel // 2
        for src_type, short_edge_type, dst_type in metadata:
            src_dim   = src_dim_dict[src_type]
            dst_dim   = src_dim_dict[dst_type]
            dst_init  = init_channel[dst_type]
            edge_dim  = edge_dim_dict[short_edge_type]
            edge_init = edge_init_dict[short_edge_type]

            self.msg[short_edge_type] = MLP(
                [src_dim + edge_dim, hidden],
                batch_norm=False, dropout_prob=0.1
            )
            self.upd[dst_type] = MLP(
                [hidden + dst_dim, out_channel - dst_init],
                batch_norm=False, dropout_prob=0.1
            )
            self.edge_upd[short_edge_type] = MLP(
                [src_dim + edge_dim + dst_dim, out_channel - edge_init],
                batch_norm=False, dropout_prob=0.1
            )

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.msg)
        reset(self.upd)
        reset(self.edge_upd)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type not in self.metadata:
                continue
            src_type, _, dst_type = edge_type
            x_src     = x_dict[src_type]
            x_dst     = x_dict[dst_type]
            edge_attr = edge_attr_dict[edge_type]

            msg = self.propagate(edge_index, x=(x_src, x_dst),
                                 edge_attr=edge_attr, edge_type=edge_type)
            tmp      = self.upd[dst_type](torch.cat([x_dst, msg], dim=1))
            dst_init = self.src_init_dict[dst_type]
            if self.src_dim_dict[dst_type] == self.out_channel:
                tmp = tmp + x_dst[:, dst_init:]
            x_dict[dst_type] = torch.cat([x_dst[:, :dst_init], tmp], dim=1)

            edge_attr_dict[edge_type] = self.edge_updater(
                edge_index, x=(x_src, x_dst),
                edge_attr=edge_attr, edge_type=edge_type
            )
        return x_dict, edge_attr_dict

    def message(self, x_j, x_i, edge_attr, edge_type):
        _, short_edge_type, _ = edge_type
        return self.msg[short_edge_type](torch.cat([x_j, edge_attr], dim=1))

    def edge_update(self, x_j, x_i, edge_attr, edge_type):
        _, short_edge_type, _ = edge_type
        tmp       = torch.cat([x_j, edge_attr, x_i], dim=1)
        out       = self.edge_upd[short_edge_type](tmp)
        edge_init = self.edge_init_dict[short_edge_type]
        if self.out_channel == edge_init:
            print(out.shape)
            print(edge_attr.shape)
            out = out + edge_attr
        return torch.cat([edge_attr[:, :edge_init], out], dim=1)

class IsacHetNet(nn.Module):
    """
    Centralized heterogeneous GNN for ISAC cell-free mMIMO (sum-rate objective).

    Graph topology (from isac_data.py):
        Nodes : AP, UE, SR
        Edges : (UE, comm_up,   AP)  [beta, gamma]
                (AP, comm_down, UE)  [beta, gamma]
                (AP, senses,    SR)  [rcs]
                (SR, sensed_by, AP)  [rcs]

    Forward pass:
        1. Encode  AP / UE / SR to out_channels
        2. Conv    UE->AP (comm)
        3. Conv    AP->UE (comm)
        4. Conv    SR->AP (sensing)
        5. Gate    fuse comm + sensing at AP
        6. Conv    AP->SR (sensing)
        7. Post    joint [UE<->AP, SR<->AP] x num_layers
        8. Heads   power_edge (AP->UE -> scalar)
                   sense_edge (AP->SR -> scalar, sensing quality proxy)

    Args:
        dim_dict : {
            'AP':         ap raw feature dim,
            'UE':         ue raw feature dim (= tau),
            'SR':         sr raw feature dim,
            'comm_edge':  comm edge attr dim (2: beta, gamma),
            'sense_edge': sense edge attr dim (1: rcs),
        }
    """
    def __init__(self, dim_dict, out_channels, num_layers=0, hid_layers=4):
        super().__init__()
        self.ue_dim         = dim_dict['UE']
        self.ap_dim         = dim_dict['AP']
        self.sr_dim         = dim_dict['SR']
        self.comm_edge_dim  = dim_dict['comm_edge']
        self.sense_edge_dim = dim_dict['sens_edge']
        self.out_channels   = out_channels

        hid = hid_layers

        # ── Encoders ──────────────────────────────────────────────────────────
        # AP, SR: fully encoded (ap_dim/sr_dim -> out_channels)
        # UE    : raw pilot sequences kept + learned part appended
        #         UE final = [phi (ue_dim) | learned (out_channels - ue_dim)]
        self.ap_encoder = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1)
        self.sr_encoder = MLP([self.sr_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1)
        self.ue_encoder = MLP([self.ue_dim, hid, out_channels - self.ue_dim],
                               batch_norm=True, dropout_prob=0.1)

        # ── Shared dim / residual bookkeeping ─────────────────────────────────
        # After encoders all nodes are out_channels wide.
        node_dim  = {'AP': out_channels, 'UE': out_channels, 'SR': out_channels}
        node_init = {'AP': self.ap_dim,  'UE': self.ue_dim,  'SR': self.sr_dim}

        # Initial edge dims (raw).  After a conv pass they grow to out_channels.
        _raw_edge = {
            'comm_up':   self.comm_edge_dim,
            'comm_down': self.comm_edge_dim,
            'sense_down':    self.sense_edge_dim,
            'sense_up': self.sense_edge_dim,
        }
        # Grown edge dims (after first conv touch).
        _grown_edge = {k: out_channels for k in _raw_edge}

        # ── Conv layers ───────────────────────────────────────────────────────

        self.convs_pre = nn.ModuleList()
        # Step 2: UE->AP  (edges start at comm_edge_dim, grow to out_channels)
        self.convs_pre.append(IsacConvLayer(
            node_dim, _raw_edge, out_channels, node_init, _raw_edge,
            [('UE', 'comm_up', 'AP')]
        ))

        # Step 3: AP->UE  (comm_down edges: raw -> grown)
        self.convs_pre.append(IsacConvLayer(
            node_dim, _raw_edge, out_channels, node_init, _raw_edge,
            [('AP', 'comm_down', 'UE')]
        ))

        # Step 4: SR->AP  (sensed_by edges: raw -> grown)
        self.convs_pre.append(IsacConvLayer(
            node_dim, _raw_edge, out_channels, node_init, _raw_edge,
            [('SR', 'sense_up', 'AP')]
        ))

        # Step 6: AP->SR  (senses edges: raw -> grown)
        self.convs_pre.append(IsacConvLayer(
            node_dim, _raw_edge, out_channels, node_init, _raw_edge,
            [('AP', 'sense_down', 'SR')]
        ))

        # Step 7: joint post layers — all edges are grown by now
        self.convs_post = nn.ModuleList()
        for _ in range(num_layers):
            self.convs_post.append(IsacConvLayer(
                node_dim, _grown_edge, out_channels, node_init, _raw_edge,
                [('UE', 'comm_up',   'AP'), ('AP', 'comm_down', 'UE'),
                 ('SR', 'sense_up', 'AP'), ('AP', 'sense_down',    'SR')]
            ))

        # ── Output heads ──────────────────────────────────────────────────────
        self.power_edge = nn.Sequential(
            MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1),
            Lin(hid, 1)
        )
        # self.sense_edge = nn.Sequential(
        #     MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1),
        #     Lin(hid, 1)
        # )

    def forward(self, batch):
        x_dict, edge_index_dict, edge_attr_dict = \
            batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict

        # 1. Encode
        x_dict['AP'] = self.ap_encoder(x_dict['AP'])
        x_dict['SR'] = self.sr_encoder(x_dict['SR'])
        aug_ue = self.ue_encoder(x_dict['UE'])
        x_dict['UE'] = torch.cat([x_dict['UE'][:, :self.ue_dim], aug_ue], dim=1)

        # 2. UE->AP (comm)
        # 3. AP->UE (comm)
        # 4. SR->AP (sensing)
        # 6. AP->SR (sensing)
        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)


        # 7. Joint post layers
        for conv in self.convs_post:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        # 8. Output heads
        ea_comm = edge_attr_dict[('AP', 'comm_down', 'UE')]
        power   = self.power_edge(ea_comm)
        edge_attr_dict[('AP', 'comm_down', 'UE')] = torch.cat(
            [ea_comm[:, :self.comm_edge_dim], power], dim=1
        )

        # ea_sense  = edge_attr_dict[('AP', 'sense_down', 'SR')]
        # sensing   = self.sense_edge(ea_sense)
        # edge_attr_dict[('AP', 'sense_down', 'SR')] = torch.cat(
        #     [ea_sense[:, :self.sense_edge_dim], sensing], dim=1
        # )

        return x_dict, edge_attr_dict, edge_index_dict