import re
import torch
import copy 
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
# from Utils.data_gen import full_het_graph
from Utils.comm import (
    variance_calculate, rate_calculation, 
    component_calculate, rate_from_component,
    power_from_raw
)


## ISAC

def sensing_component(power_matrix, q_a, q_b, q_c):
    p_sens = torch.sum(power_matrix, dim=2, keepdim=True)
    S_a = q_a * p_sens
    S_b = q_b * p_sens
    S_c = q_c * p_sens

    return S_a, S_b, S_c


def loss_function_isac_sumrate(
        graphData, nodeFeatDict, edgeDict, 
        clientResponse, bottleneckIndicator, 
        tau, rho_p, rho_d, num_antenna, 
        zeta, nu,
        round_ratio=0, alpha=None, responsibility=None, isTrain=True
    ):
    """
    Compute loss for FL training.

    Args:
        graphData: HeteroData batch
        nodeFeatDict: Node features from model
        edgeDict: Edge features from model
        clientResponse: List of dicts with DS/PC/UI from other clients
        tau, rho_p, rho_d, num_antenna: Communication parameters
        epochRatio: Training progress ratio (unused)

    Returns:
        loss: Scalar loss value
        sunm_rate_detach: [B] tensor of sum rates for monitoring
    """
    num_graphs = graphData.num_graphs
    num_UEs = graphData['UE'].x.shape[0]//num_graphs
    num_APs = graphData['AP'].x.shape[0]//num_graphs

    pilot_matrix = graphData['UE'].x[:,:tau].reshape(num_graphs, num_UEs, -1)

    large_scale = graphData['AP', 'comm_down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
    power_matrix_raw = edgeDict['AP','comm_down','UE'].reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]

    large_scale = torch.expm1(large_scale)
    channel_variance = graphData['AP', 'comm_down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]

    power_matrix_raw = power_matrix_raw[:,:1,:]
    channel_variance = channel_variance[:,:1,:]
    large_scale = large_scale[:,:1,:]

    power_matrix = power_from_raw(power_matrix_raw, channel_variance, num_antenna)
    DS_k, PC_k, UI_k = component_calculate(power_matrix, channel_variance, large_scale, pilot_matrix, rho_d=rho_d)
    
    q_a = graphData['AP'].x[:,2:3].reshape(num_graphs, num_APs, -1)
    q_b = graphData['AP'].x[:,3:4].reshape(num_graphs, num_APs, -1)
    q_c = graphData['AP'].x[:,4:5].reshape(num_graphs, num_APs, -1)

    Sa_k, Sb_k, Sc_k = sensing_component(power_matrix, q_a, q_b, q_c)



    if not isTrain:
        return DS_k, PC_k, UI_k
    
    all_DS = [DS_k] # + [r['DS'] for r in clientResponse]
    all_PC = [PC_k] # + [r['PC'] for r in clientResponse]
    all_UI = [UI_k] # + [r['UI'] for r in clientResponse]


    all_Sa = [Sa_k] # + [r['Sa'] for r in clientResponse]
    all_Sb = [Sb_k] # + [r['Sb'] for r in clientResponse]
    all_Sc = [Sc_k] # + [r['Sc'] for r in clientResponse]

    all_Sa = torch.cat(all_Sa, dim=1)
    all_Sb = torch.cat(all_Sb, dim=1)
    all_Sc = torch.cat(all_Sc, dim=1)

    if hasattr(graphData, 'w_a'):
        w_a = graphData.w_a.view(num_graphs, 1, 1)
        w_b = graphData.w_b.view(num_graphs, 1, 1)
        w_c = graphData.w_c.view(num_graphs, 1, 1)
        global_crlb = graphData.global_crlb.view(num_graphs, 1, 1)
        
        # Linear proxy loss mimicking the exact CRLB gradient (apply gradient only if global crlb > 0)
        relu_mask = (global_crlb > 0).float()
        crlb_proxy = w_a * Sa_k + w_b * Sb_k + w_c * Sc_k
        crlb_loss = (crlb_proxy * relu_mask).sum(dim=1)

    else:
        Sigma_a = torch.sum(all_Sa, dim=1)  
        Sigma_b = torch.sum(all_Sb, dim=1)
        Sigma_c = torch.sum(all_Sc, dim=1)

        crlb = (Sigma_a + Sigma_b) - nu * (Sigma_a * Sigma_b - Sigma_c ** 2)
        crlb_loss = torch.relu(crlb)


    ## ======== Comm

    all_DS = torch.cat(all_DS, dim=1)
    all_PC = torch.cat(all_PC, dim=1)
    all_UI = torch.cat(all_UI, dim=1)

    local_rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)  # [B, K]
    sum_rate_detach = torch.sum(local_rate.detach(), dim=1)

    global_sinr = graphData['UE'].x[:,-1:]
    bottle_neck = graphData['UE'].x[:,-4:-3]
    weight = bottle_neck/(1 + global_sinr)
    weight = weight.reshape(num_graphs, num_UEs).unsqueeze(1)

    local_interf_per_ue = PC_k.sum(dim=2) + UI_k.sum(dim=2) 
    if not alpha:
        alpha = 0.1 

    loss = -(weight * DS_k).sum(dim=1).mean() \
            + (alpha * weight * local_interf_per_ue).sum(dim=1).mean() \
            + crlb_loss.mean()


    return loss, sum_rate_detach



def fl_train_isac_sumrate(
        dataLoader, responseInfo, globalRates, interferences, model, optimizer,
        tau, rho_p, rho_d, num_antenna, 
        zeta, nu,
        round_ratio=0, alpha=None
):
    """
    Train a local FL model for one epoch.

    Args:
        dataLoader: List of graph batches for this client
        responseInfo: List of rate_pack (DS/PC/UI from other clients)
        model: Local model
        optimizer: Local optimizer
        tau, rho_p, rho_d, num_antenna: Communication parameters
    Returns:
        avg_loss, avg_min_rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    total_loss = 0.0
    total_min_rate = 0.0
    total_graphs = 0

    for batch, response, global_rate, interference in zip(dataLoader, responseInfo, globalRates, interferences):
        batch = batch.to(device)
        num_graph = batch.num_graphs
        optimizer.zero_grad()
        x_dict, attr_dict, _ = model(batch, isRawData=False)
        loss, min_rate = loss_function_isac_sumrate(
        # loss, min_rate = loss_function_guided(
            batch, x_dict, attr_dict, response, global_rate,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            # epochRatio=epochRatio,
            zeta=zeta, nu=nu,
            round_ratio=round_ratio, alpha=alpha,
            responsibility=interference,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()

        total_loss += loss.item() * num_graph
        total_min_rate += min_rate.mean().item() * num_graph
        total_graphs += num_graph


    return total_loss/total_graphs, total_min_rate/total_graphs


@torch.no_grad()
def fl_eval_isac_sumrate(
        dataLoader, local_models, comm_round,
        tau, rho_p, rho_d, num_antenna,
        zeta, nu
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_clients = len(local_models)

    send_to_server = get_global_info_isac(
        dataLoader, local_models,
        tau=tau, rho_p=rho_p, rho_d=rho_d,
        num_antenna=num_antenna
    )

    # response_from_server = server_return(dataLoader, send_to_server, num_antenna=num_antenna)
    response_from_server = server_return_isac(dataLoader, send_to_server, num_antenna=num_antenna)
    for _ in range(comm_round):
        send_to_server = get_global_info_isac_2nd(
            response_from_server, local_models,
            tau=tau, rho_p=rho_p, rho_d=rho_d,
            num_antenna=num_antenna
        )

        response_from_server = server_return_isac(dataLoader, send_to_server, num_antenna=num_antenna)

    

    all_DS = [[] for i in range(num_clients)]
    all_PC = [[] for i in range(num_clients)]
    all_UI = [[] for i in range(num_clients)]

    for client_idx, (model, client_data_tuple) in enumerate(zip(local_models, zip(*response_from_server))):
        batches = [item['loader'] for item in client_data_tuple]
        batch_rate = [item['rate_pack'] for item in client_data_tuple]
        globalRates = [item['global_rate'] for item in client_data_tuple]


        for batch, response, global_rate in zip(batches, batch_rate, globalRates):
            batch = batch.to(device)
            num_graph = batch.num_graphs
            x_dict, attr_dict, _ = model(batch, isRawData=False)
            DS_k, PC_k, UI_k = loss_function_isac_sumrate(
                batch, x_dict, attr_dict, response, global_rate,
                tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
                zeta=zeta, nu=nu,
                isTrain=False
            )

            all_DS[client_idx].append(DS_k.detach())
            all_PC[client_idx].append(PC_k.detach())
            all_UI[client_idx].append(UI_k.detach())

    all_DS = [torch.cat(client_batches, dim=0) for client_batches in all_DS]
    all_PC = [torch.cat(client_batches, dim=0) for client_batches in all_PC]
    all_UI = [torch.cat(client_batches, dim=0) for client_batches in all_UI]

    all_DS = torch.cat(all_DS, dim=1)
    all_PC = torch.cat(all_PC, dim=1)
    all_UI = torch.cat(all_UI, dim=1)
    
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)

    sum_rate = torch.sum(rate, dim=1)

    return sum_rate

def server_return_isac(dataLoader, globalInformation, num_antenna=1, nu=1):
    num_client = len(globalInformation)
    response_all = []

    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        aug_batch_list = []
        all_client_embeddings = [r['UE'] for r in all_response]
        all_AP_embeddings = [r['AP'] for r in all_response]
        all_SR_embeddings = [r['SR'] for r in all_response]
        all_edge_embeddings = [r['edge_down'] for r in all_response]

        all_DS = [r['DS'][:,0,:] for r in all_response] # [B, num_client, K]
        all_PC = [r['PC'][:,0,:,:] for r in all_response] # [B, num_client, K]
        all_UI = [r['UI'][:,0,:,:] for r in all_response] # [B, num_client, K]

        all_Sa = [r['Sa'][:,0] for r in all_response] # [B, num_client, K]
        all_Sb = [r['Sb'][:,0] for r in all_response] # [B, num_client, K]
        all_Sc = [r['Sc'][:,0] for r in all_response] # [B, num_client, K]

        all_Sa_stack = torch.stack(all_Sa, dim=1)
        all_Sb_stack = torch.stack(all_Sb, dim=1)
        all_Sc_stack = torch.stack(all_Sc, dim=1)

        Sigma_a = torch.sum(all_Sa_stack, dim=1)  
        Sigma_b = torch.sum(all_Sb_stack, dim=1)
        Sigma_c = torch.sum(all_Sc_stack, dim=1)

        global_crlb = (Sigma_a + Sigma_b) - nu * (Sigma_a * Sigma_b - Sigma_c ** 2) # [B, 1]

        global_crlb = torch.relu(global_crlb)


        if not all(x.shape == all_client_embeddings[0].shape for x in all_client_embeddings):
            raise RuntimeError(f"Batch {batch_idx}: Mismatch in UE counts between clients. Cannot stack.")
        
        ## Calculate rate
        all_DS_stack = torch.stack(all_DS, dim=1)
        all_PC_stack = torch.stack(all_PC, dim=1)
        all_UI_stack = torch.stack(all_UI, dim=1)
        global_rate = rate_from_component(all_DS_stack, all_PC_stack, all_UI_stack, numAntenna=num_antenna)
        # min_rate_per_sample, bottleneck_ue_idx = torch.min(global_rate, dim=1)

        ### New
        global_mean = global_rate.mean(dim=1, keepdim=True)  # [B, 1]
        global_min = global_rate.min(dim=1, keepdim=True).values  # [B, 1]
        ###
        temperature = 2
        bottleneck_indicator_mtx = F.softmax(-global_rate / temperature, dim=1)
        bottleneck_indicator = bottleneck_indicator_mtx.reshape(-1,1)

        rank_indices = torch.argsort(torch.argsort(global_rate, dim=1), dim=1)
        normalized_rank = rank_indices.float() / (global_rate.shape[1] - 1)
        normalized_rank = normalized_rank.reshape(-1, 1)

        # Total DS across all APs for contribution ratio
        total_DS = all_DS_stack.sum(dim=1)  # [B, K]
        total_Interf = (all_PC_stack.sum(dim=2) + all_UI_stack.sum(dim=2)).sum(dim=1) # [B, K]

        global_ue_context = torch.sum(torch.stack(all_client_embeddings), dim=0)

        all_interf_stack = torch.stack(
            [(all_PC[j] + all_UI[j]).sum(dim=1) for j in range(num_client)],
            dim=1
        )  # [B, num_client, K]
        total_interf_per_ue = all_interf_stack.sum(dim=1)

        for client_id, (_, batch) in enumerate(zip(all_response, all_loader)):
            responsibility_k = all_interf_stack[:, client_id, :] / (total_interf_per_ue + 1e-9)  # [B, K]
            other_AP_indices = list(range(client_id)) + list(range(client_id + 1, num_client))

            ## DS, PC, UI
            other_DS = torch.stack(all_DS[:client_id] + all_DS[client_id+1:], dim=1)
            other_PC = torch.stack(all_PC[:client_id] + all_PC[client_id+1:], dim=1).sum(dim=2)
            other_UI = torch.stack(all_UI[:client_id] + all_UI[client_id+1:], dim=1).sum(dim=2)

            # ### New
            gap_wo_features = []
            wo_mean = []
            wo_min = []
            for each_other in other_AP_indices:
                wo_other_DS = torch.stack(all_DS[:each_other] + all_DS[each_other+1:], dim=1)
                wo_other_PC = torch.stack(all_PC[:each_other] + all_PC[each_other+1:], dim=1)#.sum(dim=2)
                wo_other_UI = torch.stack(all_UI[:each_other] + all_UI[each_other+1:], dim=1)#.sum(dim=2)

                without_rate = rate_from_component(wo_other_DS, wo_other_PC, wo_other_UI, numAntenna=num_antenna)

                without_mean = without_rate.mean(dim=1)  # [B, 1]
                without_min = without_rate.min(dim=1).values  # [B, 1]
                wo_mean.append(without_mean)
                wo_min.append(without_min)
                gap_wo_features.append(torch.stack([without_mean, without_min], dim=-1))
            gap_wo_features = torch.stack(gap_wo_features, dim=1)
            wo_mean = torch.stack(wo_mean, dim=1)
            wo_min = torch.stack(wo_min, dim=1)
            ###
            # Rate pack from other APs (DC, PC, UI)
            other_pack = []
            keys_needed = ['DS', 'PC', 'UI'] # Sa, Sb, Sc are now embedded via w_a, w_b, w_c
            for j in range(num_client):
                if j != client_id:
                    full_data = all_response[j]
                    filtered_data = {k: full_data[k] for k in keys_needed}
                    other_pack.append(filtered_data)

            aug_batch = batch.clone()
            device = aug_batch['UE'].x.device
            
            # Calculate Sensing Proxy Weights (Global context)
            other_Sa = torch.stack(all_Sa[:client_id] + all_Sa[client_id+1:], dim=1).sum(dim=1) # [B, 1]
            other_Sb = torch.stack(all_Sb[:client_id] + all_Sb[client_id+1:], dim=1).sum(dim=1)
            other_Sc = torch.stack(all_Sc[:client_id] + all_Sc[client_id+1:], dim=1).sum(dim=1)
            
            # Option 1
            w_a = (1.0 - nu * other_Sb).to(device)
            w_b = (1.0 - nu * other_Sa).to(device)
            w_c = (2.0 * nu * other_Sc).to(device)

            # # Option 2
            # w_a = (1.0 - nu * Sigma_b).to(device)
            # w_b = (1.0 - nu * Sigma_a).to(device)
            # w_c = (2.0 * nu * Sigma_c).to(device)

            aug_batch.w_a = w_a
            aug_batch.w_b = w_b
            aug_batch.w_c = w_c


            aug_batch.global_crlb = global_crlb.to(device)

            ########### Start
            # GAP 
            other_AP = torch.stack(all_AP_embeddings[:client_id] + all_AP_embeddings[client_id+1:], dim=1)
            num_batch, num_other_AP, feat_dim = other_AP.shape

            # # Enhance the GAP node feature
            gap_total_DS = other_DS.sum(dim=2) # [B, num_GAP]

            # # Per-GAP total interference: [B, num_GAP]  
            gap_total_interf = (other_PC + other_UI).sum(dim=2)   # [B, num_GAP]

            # # Per-GAP load (signal-to-interference ratio)
            gap_load = gap_total_DS / (gap_total_interf + 1e-6)  # [B, num_GAP]

            ## Skip GAP 
            num_GAP = num_other_AP // 2  ########################################################################################
            w = bottleneck_indicator_mtx.to(device)              # [B, K]
            lambda_I = 1.0
            score_I = ((other_PC + other_UI) * w.unsqueeze(1)).sum(dim=2)  # [B, num_GAP]
            score_S = (other_DS * w.unsqueeze(1)).sum(dim=2)               # [B, num_GAP]
            score = score_S + lambda_I * score_I
            top_idx = torch.topk(score, k=num_GAP, dim=1).indices   

            other_AP = other_AP.gather(
                dim=1,
                index=top_idx.unsqueeze(-1).expand(-1, -1, feat_dim)
            )  # [B, L, feat_dim]
            ##


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
            edge_reshaped = other_edge.reshape(num_batch, num_ue_per_graph, num_other_AP, edge_feat_dim)

            edge_reshaped = edge_reshaped.gather(
                dim=2,
                index=top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_ue_per_graph, -1, edge_feat_dim)
            )  # [B, num_ue_per_graph, L, edge_feat_dim]
            
            ##### New #1
            X = 2               # top links per sign
            B_top = 2           # top bottleneck UEs to consider
            
            # 1) restrict to most bottleneck UEs (scalable)
            w = bottleneck_indicator_mtx.to(device)                        # [B, K]
            k_b = min(B_top, w.shape[1])
            ue_idx = torch.topk(w, k=k_b, dim=1).indices                  # [B, k_b]

            # gather DS/interference on bottleneck UE subset
            idx_bgk = ue_idx.unsqueeze(1).expand(-1, num_GAP, -1)         # [B,G,k_b]
            ds_b = torch.gather(other_DS, dim=2, index=idx_bgk)           # [B,G,k_b]
            itf_b = torch.gather(other_PC + other_UI, dim=2, index=idx_bgk)  # [B,G,k_b]

            # 2) top-X "good" and "bad" links for each GAP
            k_x = min(X, k_b)
            good_idx_local = torch.topk(ds_b, k=k_x, dim=2).indices       # [B,G,k_x]
            bad_idx_local  = torch.topk(itf_b, k=k_x, dim=2).indices      # [B,G,k_x]

            # map local idx (within bottleneck subset) -> true UE idx
            ue_idx_bgk = idx_bgk
            good_ue_idx = torch.gather(ue_idx_bgk, 2, good_idx_local)     # [B,G,k_x]
            bad_ue_idx  = torch.gather(ue_idx_bgk, 2, bad_idx_local)      # [B,G,k_x]

            # 3) gather edge features and pool
            edge_bguf = edge_reshaped.permute(0, 2, 1, 3)                 # [B,G,U,F]
            good_feat = torch.gather(
                edge_bguf, 2, good_ue_idx.unsqueeze(-1).expand(-1,-1,-1,edge_feat_dim)
            )                                                              # [B,G,k_x,F]
            bad_feat = torch.gather(
                edge_bguf, 2, bad_ue_idx.unsqueeze(-1).expand(-1,-1,-1,edge_feat_dim)
            )                                                              # [B,G,k_x,F]

            # weighted pool (better than plain mean)
            good_w = torch.softmax(torch.gather(ds_b, 2, good_idx_local), dim=2).unsqueeze(-1)
            bad_w  = torch.softmax(torch.gather(itf_b, 2, bad_idx_local), dim=2).unsqueeze(-1)
            pool_good = (good_feat * good_w).sum(dim=2)                   # [B,G,F]
            pool_bad  = (bad_feat * bad_w).sum(dim=2)                     # [B,G,F]

            # keep same interface size as current (2F)
            edge_attr_inteference = torch.cat([pool_good, pool_bad], dim=-1).reshape(-1, edge_feat_dim * 2)
            ##### Old
            # edge_mean = edge_reshaped.mean(dim=1)
            # edge_max = edge_reshaped.max(dim=1)[0]
            # # edge_std = edge_reshaped.std(dim=1)

            # edge_attr_inteference = torch.cat([edge_mean, edge_max], dim=-1).reshape(-1, edge_feat_dim*2)

            aug_batch['GAP', 'cross', 'AP'].edge_index = edge_index_inteference
            aug_batch['GAP', 'cross', 'AP'].edge_attr = edge_attr_inteference

            aug_batch['AP', 'cross-back', 'GAP'].edge_index = edge_index_inteference_back
            aug_batch['AP', 'cross-back', 'GAP'].edge_attr = edge_attr_inteference

            ## GAP - UE edge (selective: top bottleneck UEs only)
            B_top = 1 #num_ue_per_graph//2  # selective: only top-2 bottleneck UEs  # 1 UE is shiet, 2 is worse than all?
            K = num_ue_per_graph

            # 1) Select top-B_top bottleneck UEs
            w = bottleneck_indicator_mtx.to(device)          # [B, K]
            ue_top_idx = torch.topk(w, k=B_top, dim=1).indices  # [B, B_top]

            # 2) Filter DS/interference for selected GAPs (top_idx) and selected UEs (ue_top_idx)
            ds_filtered   = other_DS.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, K))     # [B, num_GAP, K]
            itf_filtered  = (other_PC + other_UI).gather(1, top_idx.unsqueeze(-1).expand(-1, -1, K))  # [B, num_GAP, K]

            # Gather at selected UEs: [B, num_GAP, B_top]
            ds_sel  = ds_filtered.gather(2, ue_top_idx.unsqueeze(1).expand(-1, num_GAP, -1))
            itf_sel = itf_filtered.gather(2, ue_top_idx.unsqueeze(1).expand(-1, num_GAP, -1))

            # 3) Edge features [B, num_GAP, B_top, F+2] → [B*G*B_top, F+2]
            edge_bgkf = edge_reshaped.permute(0, 2, 1, 3)    # [B, num_GAP, K, F]
            edge_bgkf_sel = edge_bgkf.gather(
                2, ue_top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, num_GAP, -1, edge_feat_dim)
            )  # [B, num_GAP, B_top, F]

            edge_attr_gap_ue = torch.cat([
                edge_bgkf_sel,
                ds_sel.unsqueeze(-1),
                itf_sel.unsqueeze(-1)
            ], dim=-1).reshape(-1, edge_feat_dim + 2)          # [B*G*B_top, F+2]

            # 4) Edge index: GAP_g → UE_k
            b_ids = torch.arange(num_batch, device=device)
            gap_global = (b_ids.unsqueeze(1) * num_GAP + torch.arange(num_GAP, device=device))  # [B, G]
            ue_global  = b_ids.unsqueeze(1) * K + ue_top_idx                        # [B, B_top]

            gap_src = gap_global.unsqueeze(2).expand(-1, -1, B_top).reshape(-1)     # [B*G*B_top]
            ue_dst  = ue_global.unsqueeze(1).expand(-1, num_GAP, -1).reshape(-1)          # [B*G*B_top]

            aug_batch['GAP', 'serves', 'UE'].edge_index = torch.stack([gap_src, ue_dst], dim=0)
            aug_batch['GAP', 'serves', 'UE'].edge_attr  = edge_attr_gap_ue


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

            global_sinr_val = (2 ** global_rate - 1)
            interference_sensitivity = (global_sinr_val / ((total_Interf + 1e-9) * (1 + global_sinr_val)))
            gradient_penalty = (interference_sensitivity * bottleneck_indicator_mtx).reshape(-1, 1)


            aug_batch['UE'].x = torch.cat(
                [
                    aug_batch['UE'].x, # init dim
                    new_ue_features,   # out_channel
                    bottleneck_indicator,    # [Focus: Soft attention]
                    # normalized_rank,         # [Focus: Hard priority] <--- NEW
                    contribution_ratio,      # [Action: How much I help]
                    interference_share,      # [Action: How much I hurt] <--- NEW
                    global_sinr,             # [State: How good is the user globally] <--- NEW
                    # gradient_penalty
                ],
                dim=-1
            )

            ## Global SR feature

            other_SR = torch.stack(
                all_SR_embeddings[:client_id] + all_SR_embeddings[client_id+1:], dim=0
            )  # [num_client-1, B*num_SR, feat_dim]
            global_sr_context = other_SR.mean(dim=0).to(device)  # [B*num_SR, feat_dim]

            sr_batch_idx = aug_batch['SR'].batch  # e.g., [0, 0, 1, 1, 2, 2...]
            expanded_crlb = global_crlb[sr_batch_idx].to(device)

            fim_det = Sigma_a * Sigma_b - Sigma_c**2   # [B, 1]  ← FIM determinant

            Sigma_a_expanded = Sigma_a[sr_batch_idx]   # [B * num_SR, 1]
            Sigma_b_expanded = Sigma_b[sr_batch_idx]
            Sigma_c_expanded = Sigma_c[sr_batch_idx]
            fim_det_expanded = fim_det[sr_batch_idx]

            # Test
            # 1
            crlb_margin = -expanded_crlb

            # 2
            local_Sa_k = all_Sa[client_id][:,0]  # [B, 1]
            local_Sb_k = all_Sb[client_id][:,0]
            local_Sc_k = all_Sc[client_id][:,0]

            sa_ratio = (local_Sa_k / (Sigma_a + 1e-9))[sr_batch_idx]  # [B*num_SR, 1]
            sb_ratio = (local_Sb_k / (Sigma_b + 1e-9))[sr_batch_idx]

            # 3
            # global_sr_max = other_SR.max(dim=0).values.to(device)      # [B*num_SR, feat_dim]
            # global_sr_mean = other_SR.mean(dim=0).to(device)            # [B*num_SR, feat_dim]
            # global_sr_context = torch.cat([global_sr_mean, global_sr_max], dim=-1)

            # print(global_sr_context.shape)

            aug_batch['SR'].x = torch.cat(
                [
                    aug_batch['SR'].x, 
                    Sigma_a_expanded,
                    Sigma_b_expanded,
                    Sigma_c_expanded,
                    fim_det_expanded,
                    w_a[sr_batch_idx],   # w_a: [B, 1] → [B*num_SR, 1]
                    w_b[sr_batch_idx],
                    w_c[sr_batch_idx],
                    global_sr_context, 
                    # expanded_crlb, crlb_margin,
                    # sa_ratio, sb_ratio
                ],
                dim=-1
            )
    
            # aug_batch['AP'].x = torch.cat(
            #     [
            #         aug_batch['AP'].x, # init dim
            #         global_crlb
            #     ],
            #     dim=-1
            # )

            client_data = {
                'loader': aug_batch,
                'rate_pack': other_pack,
                'global_rate': bottleneck_indicator_mtx,
                'responsibility': responsibility_k.to(device)
            }
            aug_batch_list.append(client_data)
        response_all.append(aug_batch_list)
    return response_all



def get_global_info_isac(
        loaderData, localModels, 
        tau, rho_p, rho_d, num_antenna, prev_rate=None
    ):
    num_client = len(localModels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    send_to_server = [[] for _ in range(num_client)] 
    for batches  in zip(*loaderData):                      
        for client_idx, (model, batch) in enumerate(zip(localModels, batches)):
            model.eval()
            batch = batch.to(device)
            with torch.no_grad():
                x_dict, edge_dict, edge_index = model(batch, isRawData=True)
                ##
                num_graphs = batch.num_graphs
                num_UEs = x_dict['UE'].shape[0] // num_graphs
                num_APs = x_dict['AP'].shape[0] // num_graphs
                edge_attr_up_full = edge_dict['AP','comm_down','UE']
                edge_attr_down_full = edge_dict['AP','comm_down','UE']

                largeScale = batch['AP', 'comm_down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
                power_raw = edge_attr_up_full.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
                    
                largeScale = torch.expm1(largeScale)
                phiMatrix = batch['UE'].x[:,:tau].reshape(num_graphs, num_UEs, -1)
                # channelVariance = variance_calculate(largeScale, phiMatrix, tau, rho_p)
                channelVariance = batch['AP', 'comm_down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]

                
                power = power_from_raw(power_raw, channelVariance, num_antenna)

                DS_single, PC_single, UI_single = component_calculate(power, channelVariance, largeScale, phiMatrix, rho_d=rho_d)

                q_a = batch['AP'].x[:,2:3].reshape(num_graphs, num_APs, -1)
                q_b = batch['AP'].x[:,3:4].reshape(num_graphs, num_APs, -1)
                q_c = batch['AP'].x[:,4:5].reshape(num_graphs, num_APs, -1)

                Sa_k, Sb_k, Sc_k = sensing_component(power, q_a, q_b, q_c)
                ##
            send_to_server[client_idx].append({
                ### rate pack
                'DS': DS_single.detach(),
                "PC": PC_single.detach(),
                "UI": UI_single.detach(),
                ### sensing
                'Sa': Sa_k.detach(),
                "Sb": Sb_k.detach(),
                "Sc": Sc_k.detach(),
                ### augmented data
                'UE': x_dict['UE'].detach(),
                'AP': x_dict['AP'].detach(),
                'SR': x_dict['SR'].detach(),
                'edge_down': edge_attr_down_full.detach(),
                # 'power_raw': power_raw.detach(),
                # 'largeScaleRaw': largeScale.detach(),
                # 'channelVarianceRaw': channelVariance.detach(),
                # 'edge_up': edge_attr_up_full.detach(),
                # 'phiMatrix': phiMatrix.detach(),
                # 'prev_round_rate': prev_rate
            })
            
    return send_to_server


def get_global_info_isac_2nd(
        responses, localModels,
        tau, rho_p, rho_d, num_antenna
    ):
    """
    Step 2.5: After receiving GAP-augmented data from server,
    each client computes REAL power allocation (with GAP context)
    and sends back the true DS/PC/UI to server.

    Args:
        responses: Output from server_return_GAP (list of batches, each containing list of client data)
        localModels: List of client models
        tau, rho_p, rho_d, num_antenna: Communication parameters

    Returns:
        send_to_server: List[client][batch] of dicts with REAL DS/PC/UI
    """
    num_client = len(localModels)
    num_batches = len(responses)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize: send_to_server[client_idx][batch_idx] = {...}
    send_to_server = [[] for _ in range(num_client)]

    for batch_idx in range(num_batches):
        for client_idx, model in enumerate(localModels):
            model.eval()
            client_data = responses[batch_idx][client_idx]
            batch = client_data['loader'].to(device)

            with torch.no_grad():
                # Forward with GAP context (isRawData=False)
                x_dict, edge_dict, edge_index = model(batch, isRawData=False)

                num_graphs = batch.num_graphs
                num_UEs = batch['UE'].x.shape[0] // num_graphs
                num_APs = batch['AP'].x.shape[0] // num_graphs

                # Get edge features (now contains real power from power_edge MLP)
                edge_attr_full = edge_dict['AP', 'comm_down', 'UE']

                # Extract channel info from original batch
                largeScale = batch['AP', 'comm_down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
                channelVariance = batch['AP', 'comm_down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]
                largeScale = torch.expm1(largeScale)

                # Get pilot matrix (first tau features of UE, before augmentation)
                phiMatrix = batch['UE'].x[:, :tau].reshape(num_graphs, num_UEs, -1)

                # Extract REAL power_raw (last dimension of edge features)
                power_raw = edge_attr_full.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]

                # Convert to actual power
                power = power_from_raw(power_raw, channelVariance, num_antenna)

                # Compute REAL DS/PC/UI from actual power allocation
                DS_single, PC_single, UI_single = component_calculate(
                    power, channelVariance, largeScale, phiMatrix, rho_d=rho_d
                )

                # Compute REAL FIM matrix from actual power allocation
                q_a = batch['AP'].x[:,2:3].reshape(num_graphs, num_APs, -1)
                q_b = batch['AP'].x[:,3:4].reshape(num_graphs, num_APs, -1)
                q_c = batch['AP'].x[:,4:5].reshape(num_graphs, num_APs, -1)

                Sa_k, Sb_k, Sc_k = sensing_component(power, q_a, q_b, q_c)

            send_to_server[client_idx].append({
                'DS': DS_single.detach(),
                'PC': PC_single.detach(),
                'UI': UI_single.detach(),
                'Sa': Sa_k.detach(),
                "Sb": Sb_k.detach(),
                "Sc": Sc_k.detach(),
                'edge_down': edge_attr_full.detach(),
                'power_raw': power_raw.detach(),
                'AP': x_dict['AP'].detach(),
                'UE': x_dict['UE'].detach(),
                'SR': x_dict['SR'].detach(),
            })

    return send_to_server