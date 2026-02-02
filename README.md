# GNN_FL_cf_mMIMO

A Graph Neural Network Federated Learning Apporach for Cell-Free Massive MIMO Communication 

---

## Table of Contents

- [Requirement](#requirements)
- [Installation](#installation)
- [Citation](#citation)
- [Contact](#contact)

---
## Requirements
- CUDA 11.8
- python=3.10
- pytorch=2.0.1
- torch-geometric=2.4.0

```bash
conda create -n env_name python=3.10 cudatoolkit=11.8 -y
```

---
## Installation
### Clone repo

```bash
git clone https://github.com/LeGiangK62/GNN_FL_cf_mMIMO.git
cd GNN_FL_cf_mMIMO
```
### Install dependencies
```bash
pip install -r requirements.txt
```
---

## System scheme

┌─────────────────────────────────────────────────────────────┐
│  1. ALL APs run forward pass (same time) → get_global_info  │
│     - Each AP gets: DS, PC, UI, UE embeddings               │
│                                                             │
│  2. Server aggregates → server_return                       │
│     - Augments UE features with global context              │
│     - Returns rate_pack (other APs' DS/PC/UI)               │
│                                                             │
│  3. ALL APs train on augmented data (same time)             │
│     - Each AP only modifies ITS OWN power                   │
│     - Uses rate_pack (FROZEN) for global rate calculation   │
│                                                             │
│  4. FedAvg aggregates weights                               │
└─────────────────────────────────────────────────────────────┘


## Running command
'''bash
python FlGrad.py --num_train 2000 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --cen_pretrain 01_14_19_18_18_cen --hidden_channels 64 --num_gnn_layers 3 --num_epochs 1 --num_rounds 550 --batch_size 32 --lr 1e-3  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg

python FlGrad.py --num_train 2000 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 64 --num_gnn_layers 3 --num_epochs 1 --num_rounds 550 --batch_size 32 --lr 1e-3  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg

'''
---
## Citation
Please cite my paper (To be update...)

---
## Contact

Mr. Le Tung GIANG - tung.giangle99@gmail.com or giang.lt2399144@pusan.ac.kr


## Note

lr 5e-3 is currently the best => try 1e-2

num_gnn_layer should be 2 or 3; 4 is bad; 3 is current the best
remove the sigmoid in the power MLP en 3 layers x 64

1e-2 > 5e-1 => plateue

bat_norm using is better

4 GNN layers are too much

num_gnn_layers 3 is current optimal 

###
FL GNN



system model: cf-mMIMO, K APs, serving M UEs at the same times

objective: maximize the min-rate over UEs

task: power allocation each AP to each UE, constraint of sum power budget in each AP



approach: using GNN in FL, where each AP is a client, with local data



1. raw forward, each client send a pack of data to server

2. server calculate the augmented local data for each AP, and return with DS, PC, UI (pack rate-which is depend on each client power allocation, but then can be used to calculate the real rate at each AP)

3. each client is trained on the corresponding local augmented data

def server_return_GAP(dataLoader, globalInformation, num_antenna=1):
    num_client = len(globalInformation)
    response_all = []

    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        aug_batch_list = []
        all_client_embeddings = [r['UE'] for r in all_response]
        all_AP_embeddings = [r['AP'] for r in all_response]
        all_edge_embeddings = [r['edge_down'] for r in all_response]

        all_DS = [r['DS'][:,0,:] for r in all_response] # [B, num_client, K]
        all_PC = [r['PC'][:,0,:,:] for r in all_response] # [B, num_client, K]
        all_UI = [r['UI'][:,0,:,:] for r in all_response] # [B, num_client, K]

        if not all(x.shape == all_client_embeddings[0].shape for x in all_client_embeddings):
            raise RuntimeError(f"Batch {batch_idx}: Mismatch in UE counts between clients. Cannot stack.")
        
        ## Calculate rate
        all_DS_stack = torch.stack(all_DS, dim=1)
        all_PC_stack = torch.stack(all_PC, dim=1)
        all_UI_stack = torch.stack(all_UI, dim=1)
        global_rate = rate_from_component(all_DS_stack, all_PC_stack, all_UI_stack, numAntenna=num_antenna)
        # min_rate_per_sample, bottleneck_ue_idx = torch.min(global_rate, dim=1)
        temperature = 0.001
        bottleneck_indicator = F.softmax(-global_rate / temperature, dim=1)
        bottleneck_indicator = bottleneck_indicator.reshape(-1,1)

        rank_indices = torch.argsort(torch.argsort(global_rate, dim=1), dim=1)
        normalized_rank = rank_indices.float() / (global_rate.shape[1] - 1)
        normalized_rank = normalized_rank.reshape(-1, 1)

        # Total DS across all APs for contribution ratio
        total_DS = all_DS_stack.sum(dim=1)  # [B, K]
        total_Interf = (all_PC_stack.sum(dim=2) + all_UI_stack.sum(dim=2)).sum(dim=1) # [B, K]

        global_ue_context = torch.sum(torch.stack(all_client_embeddings), dim=0)
        for client_id, (_, batch) in enumerate(zip(all_response, all_loader)):

            other_AP_indices = list(range(client_id)) + list(range(client_id + 1, num_client))

            ## DS, PC, UI
            other_DS = torch.stack(all_DS[:client_id] + all_DS[client_id+1:], dim=1)
            other_PC = torch.stack(all_PC[:client_id] + all_PC[client_id+1:], dim=1).sum(dim=2)
            other_UI = torch.stack(all_UI[:client_id] + all_UI[client_id+1:], dim=1).sum(dim=2)

            # Rate pack from other APs (DC, PC, UI)
            other_pack = []
            keys_needed = ['DS', 'PC', 'UI']
            for j in range(num_client):
                if j != client_id:
                    full_data = all_response[j]
                    filtered_data = {k: full_data[k] for k in keys_needed}
                    other_pack.append(filtered_data)

            aug_batch = batch.clone()
            device = aug_batch['UE'].x.device

            ########### Start
            # GAP 
            other_AP = torch.stack(all_AP_embeddings[:client_id] + all_AP_embeddings[client_id+1:], dim=1)
            num_batch, num_GAP, feat_dim = other_AP.shape

            # # Enhance the GAP node feature
            # gap_total_DS = other_DS.sum(dim=2) # [B, num_GAP]

            # # Per-GAP total interference: [B, num_GAP]  
            # gap_total_interf = (other_PC + other_UI).sum(dim=2)   # [B, num_GAP]

            # # Per-GAP load (signal-to-interference ratio)
            # gap_load = gap_total_DS / (gap_total_interf + 1e-6)  # [B, num_GAP]

            # # Concatenate to GAP features
            # gap_features = torch.cat([
            #     other_AP.reshape(num_batch, num_GAP, feat_dim),  # [B, num_GAP, feat_dim]
            #     # gap_total_DS.unsqueeze(-1),    # [B, num_GAP, 1]
            #     # gap_total_interf.unsqueeze(-1), # [B, num_GAP, 1]
            #     # gap_load.unsqueeze(-1)          # [B, num_GAP, 1]
            # ], dim=-1).reshape(-1, feat_dim + 1)

            gap_features = other_AP.reshape(-1, feat_dim)

            aug_batch['GAP'].x = gap_features.to(device)

            # GAP - AP
            ap_indices = torch.arange(num_batch, device=device).repeat_interleave(num_GAP)
            gap_indices = torch.arange(num_batch * num_GAP, device=device)
            edge_index_inteference = torch.stack([gap_indices, ap_indices], dim=0)
            edge_index_inteference_back = torch.stack([ap_indices, gap_indices], dim=0)

            other_edge = torch.stack(all_edge_embeddings[:client_id] + all_edge_embeddings[client_id+1:], dim=1)
            num_total_ue, _, edge_feat_dim = other_edge.shape
            num_ue_per_graph = num_total_ue // num_batch
            edge_reshaped = other_edge.reshape(num_batch, num_ue_per_graph, num_GAP, edge_feat_dim)
            # edge_summed = edge_reshaped.mean(dim=1) # sum or mean
            # edge_attr_inteference = edge_summed.reshape(-1, feat_dim)

            edge_mean = edge_reshaped.mean(dim=1)
            edge_max = edge_reshaped.max(dim=1)[0]
            edge_std = edge_reshaped.std(dim=1)

            edge_attr_inteference = torch.cat([edge_mean, edge_max], dim=-1).reshape(-1, edge_feat_dim*2)
            # edge_attr_inteference = edge_mean.reshape(-1, edge_feat_dim)
            # edge_attr_inteference_cross = edge_max.reshape(-1, edge_feat_dim)
            ## Enhace the edge

            # gap_DS = other_DS.mean(dim=2, keepdim=True)  # [B, num_GAP, 1]
            # gap_interference = (other_PC + other_UI).mean(dim=2, keepdim=True)  # [B, num_GAP, 1]

            # Concatenate to edge attr
            # edge_attr_inteference = torch.cat([
            #     edge_summed,  # [B, num_GAP, feat_dim]
            #     gap_DS,       # [B, num_GAP, 1]
            #     gap_interference  # [B, num_GAP, 1]
            # ], dim=-1).reshape(-1, feat_dim + 2)
            # edge_attr_inteference = edge_summed.reshape(-1, feat_dim)

            aug_batch['GAP', 'cross', 'AP'].edge_index = edge_index_inteference
            aug_batch['GAP', 'cross', 'AP'].edge_attr = edge_attr_inteference

            aug_batch['AP', 'cross-back', 'GAP'].edge_index = edge_index_inteference_back
            aug_batch['AP', 'cross-back', 'GAP'].edge_attr = edge_attr_inteference
            ########### End

            # Global UE context (from other APs)
            new_ue_features = ((global_ue_context - all_client_embeddings[client_id]) / (num_client - 1)).to(device)

            # Contribution ratio: how much this AP contributes to each UE's signal
            local_DS = all_DS[client_id]  # [B, K]
            contribution_ratio = (local_DS / (total_DS + 1e-6)).reshape(-1, 1)  # [B*K, 1]
            # log_rate_context = torch.log10(min_rate_per_sample + 1e-9).reshape(-1, 1).repeat(num_ue_per_graph, 1)

            # ap_conditioned_bottleneck = bottleneck_indicator * contribution_ratio

            # Context A: Global Embedding
            new_ue_features = ((global_ue_context - all_client_embeddings[client_id]) / (num_client - 1)).to(device)
            
            # Context B: Signal Contribution (My Signal / Total Signal)
            local_DS = all_DS[client_id]
            contribution_ratio = (local_DS / (total_DS + 1e-9)).reshape(-1, 1)
            
            # Context C: Interference Share (My Interference / Total Interference)
            # **CRITICAL NEW FEATURE**
            local_interf = (all_PC[client_id] + all_UI[client_id]).sum(dim=2)
            interference_share = (local_interf / (total_Interf + 1e-9)).reshape(-1, 1)

            # Context D: Global Quality (Log SINR)
            # **CRITICAL NEW FEATURE**
            # Tells the UE if it is in a "high power" or "low power" regime globally
            global_sinr = (2 ** global_rate - 1).reshape(-1, 1)


            aug_batch['UE'].x = torch.cat(
                [
                    aug_batch['UE'].x, # init dim
                    new_ue_features,   # out_channel
                    bottleneck_indicator,    # [Focus: Soft attention]
                    # normalized_rank,         # [Focus: Hard priority] <--- NEW
                    contribution_ratio,      # [Action: How much I help]
                    interference_share,      # [Action: How much I hurt] <--- NEW
                    global_sinr              # [State: How good is the user globally] <--- NEW
                ],
                dim=-1
            )

            client_data = {
                'loader': aug_batch,
                'rate_pack': other_pack
            }
            aug_batch_list.append(client_data)
        response_all.append(aug_batch_list)
    return response_all


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

class APHetNetFL_currentBest(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, aug_feat_dim=3, num_layers=0, hid_layers=4, isDecentralized=False):
        super(APHetNetFL, self).__init__()

        GAP_init_dim = out_channels # + 3
        GAP_edge_init_dim = out_channels
        src_dim_dict = dim_dict.copy()
        

        self.ue_dim = src_dim_dict['UE']
        self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']




        ##
        src_dim_dict_gap = dim_dict.copy()
        src_dim_dict_gap['GAP'] = 0
        src_dim_dict_gap['AP'] = src_dim_dict['AP']
        src_dim_dict_gap['edge'] = 0
        self.convs_gap = torch.nn.ModuleList()
            
        self.convs_gap.append(APConvLayer(
            {'GAP': GAP_init_dim, 'AP': self.ap_dim }, 
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
        
        self.convs_gap.append(APConvLayer(
            {'GAP': out_channels, 'AP': out_channels }, 
            out_channels,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
        ))

        # self.convs_gap.append(APConvLayer(
        #     {'GAP': out_channels, 'UE': out_channels }, 
        #     out_channels,
        #     out_channels, src_dim_dict_gap,
        #     [('GAP', 'g_down', 'UE'), ('UE', 'g_up', 'GAP')]
        # ))

        ##
        # self.convs_gap_post = torch.nn.ModuleList()
            
        # self.convs_gap_post.append(APConvLayer(
        #     {'GAP': out_channels, 'AP': out_channels }, 
        #     out_channels,
        #     out_channels, src_dim_dict_gap,
        #     [('GAP', 'cross', 'AP')]
        # ))
        
        # self.convs_gap_post.append(APConvLayer(
        #     {'GAP': out_channels, 'AP': out_channels }, 
        #     out_channels,
        #     out_channels, src_dim_dict_gap,
        #     [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
        # ))


        ##
        # self.convs_pre = self.create_conv_block(out_channels, out_channels, self.edge_dim, out_channels, src_dim_dict, num_layers)  
        self.convs_pre = self.create_conv_block(out_channels, out_channels, self.edge_dim, out_channels, src_dim_dict, num_layers-1)  
        self.convs_aug = self.create_conv_block(out_channels, out_channels, out_channels, out_channels, src_dim_dict, num_layers-1)  
        
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
        
        if isRawData:
            # The first round
            aug_ue = self.ue_encoder_raw(x_dict['UE'] )
            x_dict['UE'] = torch.cat(
                [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
                dim=1
            )
            aug_ap = self.ap_encoder_raw(x_dict['AP'] )
            x_dict['AP'] = aug_ap
        else:
            aug_ue = self.ue_encoder_aug(x_dict['UE'] )
            x_dict['UE'] = torch.cat(
                [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
                dim=1
            )
            for conv in self.convs_gap:
                x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        for conv in self.convs_aug:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        if not isRawData:
            # for conv in self.convs_gap_post:
            #     x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            # for conv in self.convs_aug:
            #     x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
            edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
                [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
                dim=1
            )

        return x_dict, edge_attr_dict, edge_index_dict
    

class APHetNetFL(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, aug_feat_dim=3, num_layers=0, hid_layers=4, isDecentralized=False):
        super(APHetNetFL, self).__init__()

        GAP_init_dim = out_channels # + 3
        GAP_edge_init_dim = out_channels * 2 # * 3
        src_dim_dict = dim_dict.copy()
        

        self.ue_dim = src_dim_dict['UE']
        self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']




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
        
        self.convs_gap.append(APConvLayer(
            {'GAP': out_channels, 'AP': out_channels }, 
            out_channels,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')]
        ))

        #####

        ### AP -> UE, then AP <-> UE
        self.convs_post = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs_post.append(APConvLayer(
                {'UE': out_channels, 'AP': out_channels}, 
                out_channels, out_channels, src_dim_dict, 
                [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
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
        
        if isRawData:
            aug_ue = self.ue_encoder_raw(x_dict['UE'] )
        else:
            aug_ue = self.ue_encoder_aug(x_dict['UE'] )

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

        if not isRawData:
            edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
            edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
                [edge_attr_dict[('AP', 'down', 'UE')][:,:self.edge_dim], edge_power], 
                dim=1
            )

        return x_dict, edge_attr_dict, edge_index_dict
    






