import torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, HeteroData
# 1. Location & Channel generation
def Generate_Input(num_H, tau, K, M, Pd, D=1, Hb=15, Hm=1.65, f=1900,
                    var_noise=1, Pmin=0, power_f=0.2, seed=2017, d0=0.01, d1=0.05):
    """
    Args:
        num_h: Number of channel realizations
        tau:
        K: Number of UEs
        M: Number of APs
        Pd: downlink power
        D, Hb, Hm, f,...: Channel parameters
        Pmin: ?
        seed: rand number of reproducibility
    Return: 
        BETAA_ALL: large-scale fading of shape (num_H, M, K)
        Phii_All:  pilot assignments (num_H, K, tau)
    """
    np.random.seed(seed)
    aL = (1.1 * np.log10(f) - 0.7) * Hm - (1.56 * np.log10(f) - 0.8)
    L = 46.3+33.9*np.log10(f)-13.82*np.log10(Hb)-aL
    
    random_matrix = np.random.randn(tau, tau)
    U, S, V = np.linalg.svd(random_matrix) # Pilot coodbook

    Beta_ALL = np.zeros((num_H, M, K), dtype=np.float32)
    Phii_All  = np.zeros((num_H, K, tau), dtype=np.float32)
    
        

    for each_data in range(num_H):
        # Pilot assignment
        Phii = np.zeros((K,tau))
        for k in range(K):
            Point = k % tau
            # Point = np.random.randint(1, tau+1)
            Phii[k,:] = U[Point - 1,:]
        Phii_All[each_data,:,:] = Phii
        
        
        # Random positions for APs and UEs
        AP = np.random.uniform(-D, D, size=(M, 2))
        Ter = np.random.uniform(-D, D, size=(K, 2))
        # Create an MxK large-scale coefficients beta_mk
        BETAA = np.zeros((M, K))
        # dist = np.zeros((M, K))

        for m in range(M):
            for k in range(K):
                dist = np.linalg.norm(AP[m, :] - Ter[k, :])

                if dist < d0:
                    betadB = -L - 35 * np.log10(d1) + 20 * np.log10(d1) - 20 * np.log10(d0)
                elif dist >= d0 and dist <= d1:
                    betadB = -L - 35 * np.log10(d1) + 20 * np.log10(d1) - 20 * np.log10(dist)
                else:
                    betadB = -L - 35 * np.log10(dist) + np.random.normal(0, 1) * 7

                BETAA[m, k] = 10 ** (betadB / 10) * Pd
        Beta_ALL[each_data,:,:] = BETAA
    return Beta_ALL, Phii_All



# Graph Data

# ========
# 3. Graph Utilities 

def get_cg(n):
    adj = []
    for i in range(0,n):
        for j in range(0,n):
            if(not(i==j)):
                adj.append([i,j])
    return adj

def create_graph(Beta_all, Gamma_all, Phi_all, type='het', isDecentralized=True):
    num_sample, num_AP, num_UE = Beta_all.shape
    data_list = []
    if isDecentralized:
        for each_AP in range(num_AP):
            data_single_AP = []
            for each_sample in range(num_sample):
                if type=='het':
                    data = full_het_graph(
                        Beta_all[each_sample, each_AP][np.newaxis, :], 
                        Gamma_all[each_sample, each_AP][np.newaxis, :], 
                        # Label_all[each_sample, each_AP][np.newaxis, :], 
                        None,
                        Phi_all[each_sample], 
                        each_AP, each_sample
                     )
                    # data = single_het_graph(Beta_all[each_sample, each_AP], Phi_all[each_sample])
                # elif type=='homo':
                #     data = single_graph(Beta_all[each_sample, each_AP], Phi_all[each_sample], Beta_mean, Beta_std)
                else:
                    raise ValueError(f'{type} graph is not defined!')
                data_single_AP.append(data)
            data_list.append(data_single_AP)
    else:
        for each_sample in range(num_sample):
            data = full_het_graph(
                Beta_all[each_sample], 
                Gamma_all[each_sample], 
                # Label_all[each_sample], 
                None,
                Phi_all[each_sample]
            )
            data_list.append(data)
    return data_list 


def full_het_graph(
        beta_single_sample, gamma_single_sample, 
        label_single_all, phi_single_sample, 
        ap_id=None, sample_id=None, 
        global_ap_information=None,
        tmp_ue_ue_information=None
        
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_AP, num_UE = beta_single_sample.shape

    # Creating node features (random values for AP and UE nodes)
    ap_features = np.ones((num_AP, 1), dtype=np.float32)   # np.random.rand(num_AP, 1)  # Random feature for AP node (dim 1)
    ue_features = phi_single_sample  # Random feature for UE nodes (dim 1)
    # ue_features_dummy = np.ones((num_UE, 3), dtype=np.float32)   # np.random.rand(num_AP, 1)  # Random feature for AP node (dim 1)
    # ue_features = np.concatenate([ue_features, ue_features_dummy], axis=1)
    
    # Concatenate features for both AP and UE nodes
    x_ap = torch.tensor(ap_features, dtype=torch.float32).to(device)
    x_ue = torch.tensor(ue_features, dtype=torch.float32).to(device)

    # Define edges (connect AP to all UEs in a bipartite manner)
    edge_index_ap_down_ue = []
    edge_index_ue_up_ap = []

    for ap_idx in range(num_AP):
        for ue_idx in range(num_UE):
            edge_index_ap_down_ue.append([ap_idx, ue_idx])  # AP (0) to UE (ue_idx)
            # edge_index_ue_up_ap.append([ue_idx, ap_idx])  # UE (ue_idx) to AP (0)
    
    for ue_idx in range(num_UE):
        for ap_idx in range(num_AP):
            edge_index_ue_up_ap.append([ue_idx, ap_idx])  # UE (ue_idx) to AP (0)

    edge_index_ap_down_ue = torch.tensor(edge_index_ap_down_ue, dtype=torch.long).t().contiguous().to(device)
    edge_index_ue_up_ap = torch.tensor(edge_index_ue_up_ap, dtype=torch.long).t().contiguous().to(device)

    # edge_attr_ap_to_ue = torch.tensor(beta_single_sample.reshape(-1, 1), dtype=torch.float32).to(device)
    # edge_attr_ue_up_ap = torch.tensor(beta_single_sample.T.reshape(-1, 1), dtype=torch.float32).to(device)
        
    beta_up = beta_single_sample.reshape(-1, 1)
    gamma_up = gamma_single_sample.reshape(-1, 1)
    edge_attr_ap_to_ue = np.concatenate((beta_up, gamma_up), axis=1)
    edge_attr_ap_to_ue = torch.tensor(edge_attr_ap_to_ue, dtype=torch.float32).to(device)
    
    
    beta_down = beta_single_sample.T.reshape(-1, 1)
    gamma_down = gamma_single_sample.T.reshape(-1, 1)
    edge_attr_ue_up_ap = np.concatenate((beta_down, gamma_down), axis=1)
    edge_attr_ue_up_ap = torch.tensor(edge_attr_ue_up_ap, dtype=torch.float32).to(device)   
    # Create the heterogeneous graph data
    data = HeteroData()
    data['AP'].x = x_ap
    data['UE'].x = x_ue
    data['AP', 'down', 'UE'].edge_index = edge_index_ap_down_ue
    data['AP', 'down', 'UE'].edge_attr = edge_attr_ap_to_ue
    data['UE', 'up', 'AP'].edge_index = edge_index_ue_up_ap
    data['UE', 'up', 'AP'].edge_attr = edge_attr_ue_up_ap
    
    # rate = None   
    # # Global AP
    # if global_ap_information is not None:
    #     global_ap, global_beta, global_gamma, global_power, global_ds, global_pc, global_ui, rate = global_ap_information
    #     global_ap = torch.tensor(global_ap, dtype=torch.float32).to(device)
    #     num_GAP, _ = global_beta.shape
        
    #     #Global AP
        
    #     edge_index_gap_down_ue = []
    #     edge_index_ue_up_gap = []

    #     for ap_idx in range(num_GAP):
    #         for ue_idx in range(num_UE):
    #             edge_index_gap_down_ue.append([ap_idx, ue_idx])  
        
    #     for ue_idx in range(num_UE):
    #         for ap_idx in range(num_GAP):
    #             edge_index_ue_up_gap.append([ue_idx, ap_idx])  
    #     edge_index_gap_down_ue = torch.tensor(edge_index_gap_down_ue, dtype=torch.long).t().contiguous().to(device)
    #     edge_index_ue_up_gap = torch.tensor(edge_index_ue_up_gap, dtype=torch.long).t().contiguous().to(device)

    #     beta_up = global_beta.reshape(-1, 1)
    #     gamma_up = global_gamma.reshape(-1, 1)
    #     power_up = global_power.reshape(-1, 1)
    #     global_ds_up = global_ds.reshape(-1, 1)
    #     global_pc_up = global_pc.reshape(-1, 1)
    #     global_ui_up = global_ui.reshape(-1, 1)
    #     edge_attr_gap_to_ue = np.concatenate(
    #         (
    #             beta_up, 
    #             gamma_up, 
    #             # power_up, 
    #             global_ds_up, 
    #             global_pc_up, 
    #             global_ui_up
    #         ), 
    #         axis=1
    #     )
    #     edge_attr_gap_to_ue = torch.tensor(edge_attr_gap_to_ue, dtype=torch.float32).to(device)
        
        
    #     beta_down = global_beta.T.reshape(-1, 1)
    #     gamma_down = global_gamma.T.reshape(-1, 1)
    #     power_down = global_power.T.reshape(-1, 1)
    #     global_ds_down = global_ds.T.reshape(-1, 1)
    #     global_pc_down = global_pc.T.reshape(-1, 1)
    #     global_ui_down = global_ui.T.reshape(-1, 1)
    #     edge_attr_ue_up_gap = np.concatenate(
    #         (
    #             beta_down, 
    #             gamma_down, 
    #             # power_down, 
    #             global_ds_down, 
    #             global_pc_down, 
    #             global_ui_down
    #         ), 
    #         axis=1
    #     )
    #     edge_attr_ue_up_gap = torch.tensor(edge_attr_ue_up_gap, dtype=torch.float32).to(device)   
    # # else:
    # #     global_ap = torch.zeros(0,1, dtype=torch.float32).to(device)
    # #     edge_index_gap_down_ue = torch.zeros(2,0, dtype=torch.long).contiguous().to(device)
    # #     edge_attr_gap_to_ue = torch.zeros(0,2, dtype=torch.float32).contiguous().to(device)
    # #     edge_index_ue_up_gap = torch.zeros(2,0, dtype=torch.long).contiguous().to(device)
    # #     edge_attr_ue_up_gap = torch.zeros(0,2, dtype=torch.float32).contiguous().to(device)
        
    #     data['GAP'].x = global_ap
        
    #     data['GAP', 'g_down', 'UE'].edge_index = edge_index_gap_down_ue
    #     data['GAP', 'g_down', 'UE'].edge_attr = edge_attr_gap_to_ue
    #     data['UE', 'g_up', 'GAP'].edge_index = edge_index_ue_up_gap
    #     data['UE', 'g_up', 'GAP'].edge_attr = edge_attr_ue_up_gap
        
        
    # ## UE-UE edge
    # if tmp_ue_ue_information is not None:
    #     global_pc_raw, global_ui_raw = tmp_ue_ue_information
        
    #     pc_matrix = global_pc_raw.sum(axis=0)  # [num_UE, num_UE]
    #     ui_matrix = global_ui_raw.sum(axis=0)  # [num_UE, num_UE]
    #     src, dst = np.meshgrid(np.arange(num_UE), np.arange(num_UE), indexing='ij')
    #     mask = src != dst  # Exclude self-loops
    #     ue_ue_edge_index = np.stack([src[mask], dst[mask]], axis=0)  # [2, num_edges]
    #     ue_ue_edge_attr = np.stack([
    #         pc_matrix[mask],  # PC values
    #         ui_matrix[mask]   # UI values
    #     ], axis=1)  # [num_edges, 2]
    
    #     # old
    #     # ue_ue_edge_index = []
    #     # ue_ue_edge_attr = []
        
    #     # for n in range(num_UE):
    #     #     for n_prime in range(num_UE):
    #     #         if n != n_prime: continue
    #     #         pc_from_prime = global_pc_raw[:, n_prime, n].sum().item()
    #     #         ui_from_prime = global_ui_raw[:, n_prime, n].sum().item() 

    #     #         ue_ue_edge_index.append([n_prime, n])
    #     #         ue_ue_edge_attr.append([pc_from_prime, ui_from_prime])
        
    #     ue_ue_edge_index = torch.tensor(ue_ue_edge_index, dtype=torch.long).contiguous()
    #     ue_ue_edge_attr = torch.tensor(ue_ue_edge_attr, dtype=torch.float32)
    #     data['UE', 'interfere', 'UE'].edge_index = ue_ue_edge_index.to(device)
    #     data['UE', 'interfere', 'UE'].edge_attr = ue_ue_edge_attr.to(device)
        
        
    # if label_single_all is not None:
    #     data.y = torch.tensor([label_single_all], dtype=torch.float32).to(device)
    
    # if rate is not None:
    #     #Augment the rate to UE
    #     assert rate.shape == (num_UE, 1)
    #     rate = torch.tensor(rate, dtype=torch.float32).to(device)
    #     data['UE'].x = torch.cat([x_ue, rate], dim=1)
    # else:
    #     dummy = torch.zeros(num_UE, 1, dtype=torch.float32).to(device)
    #     data['UE'].x = torch.cat([x_ue, dummy], dim=1)
    
    data.ap_id = ap_id
    data.sample_id = sample_id

    return data

    

def build_loader(per_ap_datasets, batch_size, seed, drop_last=True, num_workers=0):
    n = len(per_ap_datasets[0])
    assert all(len(ds) == n for ds in per_ap_datasets), "All AP datasets must have same length."
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()  # same random order for all APs

    loaders = []
    for ds in per_ap_datasets:
        subset = Subset(ds, order)  # fixes the order
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers))
    return loaders


def build_cen_loader(betaMatrix, gammaMatrix, phiMatrix, batchSize, isShuffle=False):
    log_large_scale = np.log1p(betaMatrix)
    deta_cen = create_graph(log_large_scale, gammaMatrix, phiMatrix, 'het', isDecentralized=False)
    loader_cen = DataLoader(deta_cen, batch_size=batchSize, shuffle=isShuffle)
    return deta_cen, loader_cen


def build_decen_loader(betaMatrix, gammaMatrix, phiMatrix, batchSize, seed=1712):
    log_large_scale = np.log1p(betaMatrix)
    data_decen = create_graph(log_large_scale, gammaMatrix, phiMatrix, 'het')
    loader_decen = build_loader(data_decen, batchSize, seed=seed, drop_last=False)
    return data_decen, loader_decen
    