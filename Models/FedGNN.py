import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, BatchNorm1d as BN, LayerNorm, Dropout, GELU, LeakyReLU
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.utils import dropout_node, dropout_edge
from torch_geometric.nn import GraphNorm

from .GNN import APConvLayer, MLP



# FL

class APHetNetFL(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, aug_feat_dim=3, num_layers=0, hid_layers=4, isDecentralized=False):
        super(APHetNetFL, self).__init__()

        GAP_init_dim = out_channels + 0# + 3
        GAP_edge_init_dim = out_channels * 2 # * 3
        GAP_UE_edge = out_channels
        src_dim_dict = dim_dict.copy()
        

        self.ue_dim = src_dim_dict['UE']
        self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']

        self.out_channels = out_channels



        ##
        src_dim_dict_gap = dim_dict.copy()
        src_dim_dict_gap['GAP'] = 0
        src_dim_dict_gap['AP'] = src_dim_dict['AP']
        src_dim_dict_gap['edge'] = 0
        

        ##
        # self.convs_pre = self.create_conv_block(out_channels, out_channels, self.edge_dim, out_channels, src_dim_dict, num_layers)  
        # self.convs_pre = self.create_conv_block(out_channels, out_channels, self.edge_dim, out_channels, src_dim_dict, num_layers-1)  
        # self.convs_aug = self.create_conv_block(out_channels, out_channels, out_channels, out_channels, src_dim_dict, num_layers-1)  

        ### UE <-> AP
        self.convs_pre = torch.nn.ModuleList()
            
        self.convs_pre.append(APConvLayer(
            {'UE': out_channels, 'AP': out_channels}, 
            self.edge_dim,
            out_channels, src_dim_dict,
            [('UE', 'up', 'AP')]
        ))

        self.convs_pre.append(APConvLayer(
            {'UE': out_channels, 'AP': out_channels}, 
            self.edge_dim,
            out_channels, src_dim_dict,
            [('AP','down','UE')]
        ))

        #####

        ### GAP <-> AP
        self.convs_gap = torch.nn.ModuleList()

        self.convs_gap.append(APConvLayer(
            {'GAP': GAP_init_dim, 'AP': out_channels}, 
            GAP_edge_init_dim,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP')]
        ))

        self.convs_gap.append(APConvLayer(
            {'GAP': GAP_init_dim, 'AP': out_channels }, 
            GAP_edge_init_dim,
            out_channels, src_dim_dict_gap,
            [('AP', 'cross-back', 'GAP')]
        ))
        
        # for _ in range(num_layers): # too many layers break the model
        self.convs_gap.append(APConvLayer(
            {'GAP': out_channels, 'AP': out_channels }, 
            out_channels,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
        ))

        #####


        # ### GAP <-> UE
        # self.conv_gap_ue = APConvLayer(
        #     {'GAP': out_channels, 'UE': out_channels},
        #     GAP_UE_edge + 2,           # F+2
        #     out_channels, src_dim_dict_gap,
        #     [('GAP', 'serves', 'UE')]
        # )
        
        ###

        ### AP -> UE, then AP <-> UE
        self.convs_post = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs_post.append(APConvLayer(
                {'UE': out_channels, 'AP': out_channels}, 
                out_channels, out_channels, src_dim_dict, 
                [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
            ))

        
        hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
        # self.ue_encoder_raw = MLP([self.ue_dim, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
        self.ue_encoder_raw = MLP([self.ue_dim, hid, self.ue_dim_aug], batch_norm=True, dropout_prob=0.1) 
        self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
        self.ap_encoder_raw = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1) 
        self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        self.power_edge = nn.Sequential(
            *[
                self.power_edge, Seq(Lin(hid, 1)), 
            ]
        )

    def create_conv_block(self, ue_in, ap_in, edge, out_channels, src_dim_dict, num_layers):
            layers = torch.nn.ModuleList()
            
            # # First Layer to update RRU
            layers.append(APConvLayer(
                {'UE': ue_in, 'AP': ap_in}, 
                edge,
                out_channels, src_dim_dict,
                [('UE', 'up', 'AP')]
            ))
            
            layers.append(APConvLayer(
                {'UE': ue_in, 'AP': out_channels},
                edge,
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


        aug_ap = self.ap_encoder_raw(x_dict['AP'] )
        x_dict['AP'] = aug_ap
        tmp = x_dict['UE'] 
        if isRawData:
            tmp = self.ue_encoder_raw(x_dict['UE'] )
            # aug_ue = torch.zeros(x_dict['UE'].shape[0], self.out_channels - self.ue_dim, device=x_dict['UE'].device)
        # else:
        aug_ue = self.ue_encoder_aug(tmp)

        x_dict['UE'] = torch.cat(
            [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
            dim=1
        )

        # UE -> AP
        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        if not isRawData:
            for conv in self.convs_gap:
                x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            # x_dict, edge_attr_dict = self.conv_gap_ue(x_dict, edge_index_dict, edge_attr_dict)
        

        # AP <-> UE
        for conv in self.convs_post:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
        # if not isRawData:
        #     edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
        #         [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
        #         dim=1
        #     )
        # else:
        edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
            [edge_attr_dict[('AP', 'down', 'UE')][:,:-1], edge_power], 
            dim=1
        )

        return x_dict, edge_attr_dict, edge_index_dict
    




class APHetNetFL_sumrate(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, aug_feat_dim=3, num_layers=0, hid_layers=4, isDecentralized=False):
        super(APHetNetFL_sumrate, self).__init__()

        GAP_init_dim = out_channels + 0# + 3
        GAP_edge_init_dim = out_channels * 2 # * 3
        GAP_UE_edge = out_channels
        src_dim_dict = dim_dict.copy()
        

        self.ue_dim = src_dim_dict['UE']
        self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']

        self.out_channels = out_channels

        ##
        src_dim_dict_gap = dim_dict.copy()
        src_dim_dict_gap['GAP'] = 0
        src_dim_dict_gap['AP'] = src_dim_dict['AP']
        src_dim_dict_gap['edge'] = 0

        ### UE <-> AP
        self.convs_pre = torch.nn.ModuleList()
            
        self.convs_pre.append(APConvLayer(
            {'UE': out_channels, 'AP': out_channels}, 
            self.edge_dim,
            out_channels, src_dim_dict,
            [('UE', 'up', 'AP')]
        ))

        self.convs_pre.append(APConvLayer(
            {'UE': out_channels, 'AP': out_channels}, 
            self.edge_dim,
            out_channels, src_dim_dict,
            [('AP','down','UE')]
        ))

        #####

        ### GAP <-> AP
        self.convs_gap = torch.nn.ModuleList()

        self.convs_gap.append(APConvLayer(
            {'GAP': GAP_init_dim, 'AP': out_channels}, 
            GAP_edge_init_dim,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP')]
        ))

        self.convs_gap.append(APConvLayer(
            {'GAP': GAP_init_dim, 'AP': out_channels }, 
            GAP_edge_init_dim,
            out_channels, src_dim_dict_gap,
            [('AP', 'cross-back', 'GAP')]
        ))
        
        # for _ in range(num_layers): # too many layers break the model
        self.convs_gap.append(APConvLayer(
            {'GAP': out_channels, 'AP': out_channels }, 
            out_channels,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
        ))

        ### AP -> UE, then AP <-> UE
        self.convs_post = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs_post.append(APConvLayer(
                {'UE': out_channels, 'AP': out_channels}, 
                out_channels, out_channels, src_dim_dict, 
                [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
            ))

        
        hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
        # self.ue_encoder_raw = MLP([self.ue_dim, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
        self.ue_encoder_raw = MLP([self.ue_dim, hid, self.ue_dim_aug], batch_norm=True, dropout_prob=0.1) 
        self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
        self.ap_encoder_raw = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1) 
        self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        self.power_edge = nn.Sequential(
            *[
                self.power_edge, Seq(Lin(hid, 1)), 
            ]
        )

    def create_conv_block(self, ue_in, ap_in, edge, out_channels, src_dim_dict, num_layers):
            layers = torch.nn.ModuleList()
            
            # # First Layer to update RRU
            layers.append(APConvLayer(
                {'UE': ue_in, 'AP': ap_in}, 
                edge,
                out_channels, src_dim_dict,
                [('UE', 'up', 'AP')]
            ))
            
            layers.append(APConvLayer(
                {'UE': ue_in, 'AP': out_channels},
                edge,
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


        aug_ap = self.ap_encoder_raw(x_dict['AP'] )
        x_dict['AP'] = aug_ap
        tmp = x_dict['UE'] 
        if isRawData:
            tmp = self.ue_encoder_raw(x_dict['UE'] )
        aug_ue = self.ue_encoder_aug(tmp)

        x_dict['UE'] = torch.cat(
            [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
            dim=1
        )

        # UE -> AP
        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        if not isRawData:
            for conv in self.convs_gap:
                x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        # AP <-> UE
        for conv in self.convs_post:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
        edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
            [edge_attr_dict[('AP', 'down', 'UE')][:,:-1], edge_power], 
            dim=1
        )

        return x_dict, edge_attr_dict, edge_index_dict
    

## ISAC

class IsacHetNetFL(nn.Module):
    def __init__(
            self, dim_dict, out_channels, 
            aug_feat_dim=4, 
            aug_feat_dim_ap=1,
            num_layers=0, hid_layers=4, 
            isDecentralized=False
        ):
        super(IsacHetNetFL, self).__init__()

        GAP_init_dim     = out_channels
        GAP_edge_init_dim = out_channels * 2
        src_dim_dict = dim_dict.copy()

        self.ue_dim        = src_dim_dict['UE']
        self.ue_dim_aug    = src_dim_dict['UE'] + out_channels + aug_feat_dim
        self.ap_dim        = src_dim_dict['AP'] 
        self.ap_dim_aug        = src_dim_dict['AP'] + aug_feat_dim_ap
        self.sr_dim        = src_dim_dict['SR']
        self.comm_edge_dim = src_dim_dict['comm_edge']
        self.sense_edge_dim = src_dim_dict['sens_edge']
        self.out_channels  = out_channels

        # init_channel dicts: original raw dims for skip connections in APConvLayer
        # APConvLayer requires an 'edge' key in init_channel
        init_comm = {'UE': self.ue_dim,  'AP': self.ap_dim,  'edge': self.comm_edge_dim}
        init_sens = {'SR': self.sr_dim,  'AP': self.ap_dim,  'edge': self.sense_edge_dim}
        init_gap  = {'GAP': 0,           'AP': self.ap_dim,  'edge': 0}

        ### UE <-> AP (comm)
        self.convs_pre = torch.nn.ModuleList()

        self.convs_pre.append(APConvLayer(
            {'UE': out_channels, 'AP': out_channels},
            self.comm_edge_dim, out_channels, init_comm,
            [('UE', 'comm_up', 'AP')]
        ))
        self.convs_pre.append(APConvLayer(
            {'UE': out_channels, 'AP': out_channels},
            self.comm_edge_dim, out_channels, init_comm,
            [('AP', 'comm_down', 'UE')]
        ))

        ### SR <-> AP (sensing)
        self.convs_pre.append(APConvLayer(
            {'SR': out_channels, 'AP': out_channels},
            self.sense_edge_dim, out_channels, init_sens,
            [('SR', 'sens_up', 'AP')]
        ))
        self.convs_pre.append(APConvLayer(
            {'SR': out_channels, 'AP': out_channels},
            self.sense_edge_dim, out_channels, init_sens,
            [('AP', 'sens_down', 'SR')]
        ))

        ### GAP <-> AP
        self.convs_gap = torch.nn.ModuleList()

        self.convs_gap.append(APConvLayer(
            {'GAP': GAP_init_dim, 'AP': out_channels},
            GAP_edge_init_dim, out_channels, init_gap,
            [('GAP', 'cross', 'AP')]
        ))
        self.convs_gap.append(APConvLayer(
            {'GAP': GAP_init_dim, 'AP': out_channels},
            GAP_edge_init_dim, out_channels, init_gap,
            [('AP', 'cross-back', 'GAP')]
        ))
        self.convs_gap.append(APConvLayer(
            {'GAP': out_channels, 'AP': out_channels},
            out_channels, out_channels, init_gap,
            [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
        ))

        ### Post: comm and sensing run separately to keep node-dim sums consistent
        self.convs_post_comm = torch.nn.ModuleList()
        self.convs_post_sens = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs_post_comm.append(APConvLayer(
                {'UE': out_channels, 'AP': out_channels},
                out_channels, out_channels, init_comm,
                [('UE', 'comm_up', 'AP'), ('AP', 'comm_down', 'UE')]
            ))
            self.convs_post_sens.append(APConvLayer(
                {'SR': out_channels, 'AP': out_channels},
                out_channels, out_channels, init_sens,
                [('SR', 'sens_up', 'AP'), ('AP', 'sens_down', 'SR')]
            ))

        hid = hid_layers

        ## NOTE: Start AP augmentation
        # No ap aug
        self.ue_encoder_raw = MLP([self.ue_dim, hid, self.ue_dim_aug], batch_norm=True, dropout_prob=0.1)
        self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1)
        self.ap_encoder_raw = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1)

        # No ap aug
        # self.ue_encoder_raw = MLP([self.ue_dim, hid, self.ue_dim_aug], batch_norm=True, dropout_prob=0.1)
        # self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1)
        # self.ap_encoder_raw = MLP([self.ap_dim, hid, self.ap_dim_aug], batch_norm=True, dropout_prob=0.1)
        # self.ap_encoder_aug = MLP([self.ap_dim_aug, hid, out_channels - self.ap_dim], batch_norm=True, dropout_prob=0.1)

        #Endregion

        self.sr_encoder     = MLP([self.sr_dim,  hid, out_channels], batch_norm=True, dropout_prob=0.1)
        self.power_edge = nn.Sequential(
            MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1),
            Seq(Lin(hid, 1))
        )

    def forward(self, batch, isRawData=False):
        x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict

        x_dict['SR'] = self.sr_encoder(x_dict['SR'])

        ue_tmp = x_dict['UE']
        if isRawData:
            ue_tmp = self.ue_encoder_raw(x_dict['UE'])
        aug_ue = self.ue_encoder_aug(ue_tmp)
        x_dict['UE'] = torch.cat([x_dict['UE'][:, :self.ue_dim], aug_ue], dim=1)

        
        ## NOTE: Start AP augmentation
        x_dict['AP'] = self.ap_encoder_raw(x_dict['AP'])

        # ap_tmp = x_dict['AP']
        # if isRawData:
        #     ap_tmp = self.ap_encoder_raw(x_dict['AP'])
        # aug_ap = self.ap_encoder_aug(ap_tmp)
        # x_dict['AP'] = torch.cat([x_dict['AP'][:, :self.ap_dim], aug_ap], dim=1)



        # UE <-> AP (comm), SR <-> AP (sensing)
        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        if not isRawData:
            for conv in self.convs_gap:
                x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        for comm_conv, sens_conv in zip(self.convs_post_comm, self.convs_post_sens):
            x_dict, edge_attr_dict = comm_conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict, edge_attr_dict = sens_conv(x_dict, edge_index_dict, edge_attr_dict)

        edge_power = self.power_edge(edge_attr_dict[('AP', 'comm_down', 'UE')])
        edge_attr_dict[('AP', 'comm_down', 'UE')] = torch.cat(
            [edge_attr_dict[('AP', 'comm_down', 'UE')][:, :-1], edge_power],
            dim=1
        )

        return x_dict, edge_attr_dict, edge_index_dict
    