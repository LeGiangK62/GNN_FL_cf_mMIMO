import torch

import numpy as np
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

    Beta_ALL = np.zeros((num_H, M, K), dtype=np.float64)
    Phii_All  = np.zeros((num_H, K, tau), dtype=np.float64)
    
        

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

# def Generate_Label(num_H, tau, K, M, BETAA_ALL,Phii_All, Pu, Pd):
#     # Eta_All = np.zeros((num_H,M,K))

#     # for ite in range(num_H):
#     #   Phii = Phii_All[ite,:,:]
#     #   BETAA = BETAA_ALL[ite,:,:]
#     #   Gammaa = np.zeros((M,K))
#     #   mau = np.zeros((M,K))

#     #   for m in range(M):
#     #     for k in range(K):
#     #       for kk in range(K):
#     #         mau[m, k] += BETAA[m, kk] * (np.linalg.norm(Phii[k,:].dot(Phii[kk,:].T))) ** 2

#     #   for m in range(M):
#     #     for k in range(K):
#     #       Gammaa[m, k] = tau * BETAA[m, k] **2 / (tau * mau[m, k] + 1)
#     #   # Compute etaa(m): each AP transmits equal power to K terminals

#     #   etaa = np.zeros((M,K))
#     #   for m in range(M):
#     #     for k in range(K):
#     #       etaa[m,k] = 0.2/sum(Gammaa[m,:])
#     #   Eta_All[ite,:,:] = etaa
#     Eta_All = np.ones((num_H,M,K))
#     return Eta_All


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

# def build_graph(K, H, Phii, adj, pos_AP, pos_sample, Power_scale):


#     x1 = np.expand_dims(H[pos_AP,:].T, axis = 1)
#     x2 = np.random.rand(x1.shape[0],1)
#     x = np.concatenate((x1,Phii,x2), axis = 1)
#     x = torch.tensor(x, dtype = torch.float)
#     edge_attr = []
#     for e in adj:
#       edge_attr.append([H[pos_AP,e[0]],H[pos_AP,e[1]]])

#     edge_index = torch.tensor(adj, dtype = torch.long)
#     edge_attr = torch.tensor(edge_attr, dtype = torch.float)

#     pos = torch.tensor(pos_sample, dtype = torch.float)
#     data = Data(x=x, edge_index=edge_index.t().contiguous(),edge_attr = edge_attr, pos = pos)
#     return data


# def proc_data(Beta_ALL,Phii_All,Power_scale): # Power_scale: Pd, Pu
#     num_H = Beta_ALL.shape[0]
#     K = Beta_ALL.shape[2]
#     M = Beta_ALL.shape[1]
#     data_list = []
#     cg = get_cg(K) # get adjacent matrix
#     for i in range(num_H):
#       for m in range(M):
#         data = build_graph(K, Beta_ALL[i], Phii_All[i], cg, m, i, Power_scale)
#         data_list.append(data)
#     return data_list


def create_graph(Beta_all, Phi_all, type='het', isDecentralized=True):
    num_sample, num_AP, num_UE = Beta_all.shape
    data_list = []
    if isDecentralized:
        for each_AP in range(num_AP):
            data_single_AP = []
            for each_sample in range(num_sample):
                if type=='het':
                    data = full_het_graph(Beta_all[each_sample, each_AP][np.newaxis, :], Phi_all[each_sample])
                    # data = single_het_graph(Beta_all[each_sample, each_AP], Phi_all[each_sample])
                elif type=='homo':
                    data = single_graph(Beta_all[each_sample, each_AP], Phi_all[each_sample])
                else:
                    raise ValueError(f'{type} graph is not defined!')
                data_single_AP.append(data)
            data_list.append(data_single_AP)
    else:
        for each_sample in range(num_sample):
            data = full_het_graph(Beta_all[each_sample], Phi_all[each_sample])
            data_list.append(data)
    return data_list 

def single_graph(beta_single_AP, phi_single_AP):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_UE = beta_single_AP.shape[0]
    
    x1 = np.expand_dims(beta_single_AP, axis=1)
    x2 = np.random.rand(x1.shape[0],1)
    x = np.concatenate((x1,phi_single_AP,x2), axis = 1)
    x = torch.tensor(x, dtype = torch.float).to(device)
    edge_index = get_cg(num_UE)
    
    edge_attr = []
    for e in edge_index:
        edge_attr.append([beta_single_AP[e[0]], beta_single_AP[e[1]]])
        
    edge_index = torch.tensor(edge_index, dtype = torch.long).to(device)
    edge_attr = torch.tensor(edge_attr, dtype = torch.float).to(device)
    
    
    data = Data(x=x, edge_index=edge_index.t().contiguous(),edge_attr = edge_attr)
    return data



def full_het_graph(beta_single_sample, phi_single_sample):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_AP, num_UE = beta_single_sample.shape

    # Creating node features (random values for AP and UE nodes)
    ap_features = np.random.rand(num_AP, 1)  # Random feature for AP node (dim 1)
    ue_features = phi_single_sample  # Random feature for UE nodes (dim 1)

    # Concatenate features for both AP and UE nodes
    x_ap = torch.tensor(ap_features, dtype=torch.float).to(device)
    x_ue = torch.tensor(ue_features, dtype=torch.float).to(device)

    # Combine AP and UE node features
    x = {'AP': x_ap, 'UE': x_ue}

    # Define edges (connect AP to all UEs in a bipartite manner)
    edge_index_ap_down_ue = []
    edge_index_ue_up_ap = []

    for ap_idx in range(num_AP):
        for ue_idx in range(num_UE):
            edge_index_ap_down_ue.append([ap_idx, ue_idx])  # AP (0) to UE (ue_idx)
            edge_index_ue_up_ap.append([ue_idx, ap_idx])  # UE (ue_idx) to AP (0)

    edge_index_ap_down_ue = torch.tensor(edge_index_ap_down_ue, dtype=torch.long).t().contiguous().to(device)
    edge_index_ue_up_ap = torch.tensor(edge_index_ue_up_ap, dtype=torch.long).t().contiguous().to(device)
    edge_attr_ap_to_ue = torch.tensor(beta_single_sample.reshape(-1, 1), dtype=torch.float).to(device)
    edge_attr_ue_up_ap = torch.tensor(beta_single_sample.T.reshape(-1, 1), dtype=torch.float).to(device)

    # Create the heterogeneous graph data
    data = HeteroData()
    data['AP'].x = x['AP']
    data['UE'].x = x['UE']
    data['AP', 'down', 'UE'].edge_index = edge_index_ap_down_ue
    data['AP', 'down', 'UE'].edge_attr = edge_attr_ap_to_ue
    data['UE', 'up', 'AP'].edge_index = edge_index_ue_up_ap
    data['UE', 'up', 'AP'].edge_attr = edge_attr_ue_up_ap

    return data

def single_het_graph(beta_single_AP, phi_single_AP):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    beta_single_AP = beta_single_AP[:,np.newaxis]
    num_UE = beta_single_AP.shape[0]
    num_AP = 1

    # Creating node features (random values for AP and UE nodes)
    ap_features = np.random.rand(num_AP, 1)  # Random feature for AP node (dim 1)
    ue_features = phi_single_AP  # Random feature for UE nodes (dim 1)

    # Concatenate features for both AP and UE nodes
    x_ap = torch.tensor(ap_features, dtype=torch.float).to(device)
    x_ue = torch.tensor(ue_features, dtype=torch.float).to(device)

    # Combine AP and UE node features
    x = {'AP': x_ap, 'UE': x_ue}

    # Define edges (connect AP to all UEs in a bipartite manner)
    edge_index_ap_down_ue = []
    edge_index_ue_up_ap = []

    for ue_idx in range(num_UE):
        edge_index_ap_down_ue.append([0, ue_idx])  # AP (0) to UE (ue_idx)
        edge_index_ue_up_ap.append([ue_idx, 0])  # UE (ue_idx) to AP (0)

    edge_index_ap_down_ue = torch.tensor(edge_index_ap_down_ue, dtype=torch.long).t().contiguous().to(device)
    edge_index_ue_up_ap = torch.tensor(edge_index_ue_up_ap, dtype=torch.long).t().contiguous().to(device)
    edge_attr_ap_to_ue = torch.tensor(beta_single_AP, dtype=torch.float).to(device)
    edge_attr_ue_up_ap = torch.tensor(beta_single_AP, dtype=torch.float).to(device)

    # Create the heterogeneous graph data
    data = HeteroData()
    data['AP'].x = x['AP']
    data['UE'].x = x['UE']
    data['AP', 'down', 'UE'].edge_index = edge_index_ap_down_ue
    data['AP', 'down', 'UE'].edge_attr = edge_attr_ap_to_ue
    data['UE', 'up', 'AP'].edge_index = edge_index_ue_up_ap
    data['UE', 'up', 'AP'].edge_attr = edge_attr_ue_up_ap

    return data


def single_syn_het_graph(ap_feat, large_scale_feat):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_UE = large_scale_feat.shape[0]
    num_AP = ap_feat.shape[0]

    # Creating node features (random values for AP and UE nodes)
    ap_features = ap_feat  # Random feature for AP node (dim 1)
    ue_features = torch.rand(num_UE, 1)  # Random feature for UE nodes (dim 1)

    # Concatenate features for both AP and UE nodes
    x_ap = ap_features.to(torch.float).to(device)
    x_ue = ue_features.to(torch.float).to(device)

    # Combine AP and UE node features
    x = {'AP': x_ap, 'UE': x_ue}

    # Define edges (connect AP to all UEs in a bipartite manner)
    edge_index_ap_down_ue = []
    edge_index_ue_up_ap = []

    for ue_idx in range(num_UE):
        edge_index_ap_down_ue.append([0, ue_idx])  # AP (0) to UE (ue_idx)
        edge_index_ue_up_ap.append([ue_idx, 0])  # UE (ue_idx) to AP (0)

    edge_index_ap_down_ue = torch.tensor(edge_index_ap_down_ue, dtype=torch.long).t().contiguous().to(device)
    edge_index_ue_up_ap = torch.tensor(edge_index_ue_up_ap, dtype=torch.long).t().contiguous().to(device)
    edge_attr_ap_to_ue = large_scale_feat.to(dtype=torch.float).to(device)
    edge_attr_ue_up_ap = large_scale_feat.to(dtype=torch.float).to(device)

    # Create the heterogeneous graph data
    data = HeteroData()
    data['AP'].x = x['AP']
    data['UE'].x = x['UE']
    data['AP', 'down', 'UE'].edge_index = edge_index_ap_down_ue
    data['AP', 'down', 'UE'].edge_attr = edge_attr_ap_to_ue
    data['UE', 'up', 'AP'].edge_index = edge_index_ue_up_ap
    data['UE', 'up', 'AP'].edge_attr = edge_attr_ue_up_ap

    return data

