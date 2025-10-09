import torch

import numpy as np
from torch_geometric.data import Data

# Location Generation
def Generate_Input(num_H, tau, K, M, Pd, D=1, Hb=15, Hm=1.65, f=1900,
                    var_noise=1, Pmin=0, power_f=0.2, seed=2017, d0=0.01, d1=0.05):
  # print('Generate Data ... (seed = %d)' % seed)
  # np.random.seed(seed)

  aL = (1.1 * np.log10(f) - 0.7) * Hm - (1.56 * np.log10(f) - 0.8)
  L = 46.3+33.9*np.log10(f)-13.82*np.log10(Hb)-aL
  BETAA_ALL = np.zeros((num_H, M, K))
  for ite in range(num_H):
    # generate location
    AP = np.random.uniform(-1, 1, size=(M, 2))
    Ter = np.random.uniform(-1, 1, size=(K, 2))
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
    BETAA_ALL[ite,:,:] = BETAA
    return BETAA_ALL

def Generate_Label(num_H, tau, K, M, BETAA_ALL,Phii_All, Pu, Pd):
    # Eta_All = np.zeros((num_H,M,K))

    # for ite in range(num_H):
    #   Phii = Phii_All[ite,:,:]
    #   BETAA = BETAA_ALL[ite,:,:]
    #   Gammaa = np.zeros((M,K))
    #   mau = np.zeros((M,K))

    #   for m in range(M):
    #     for k in range(K):
    #       for kk in range(K):
    #         mau[m, k] += BETAA[m, kk] * (np.linalg.norm(Phii[k,:].dot(Phii[kk,:].T))) ** 2

    #   for m in range(M):
    #     for k in range(K):
    #       Gammaa[m, k] = tau * BETAA[m, k] **2 / (tau * mau[m, k] + 1)
    #   # Compute etaa(m): each AP transmits equal power to K terminals

    #   etaa = np.zeros((M,K))
    #   for m in range(M):
    #     for k in range(K):
    #       etaa[m,k] = 0.2/sum(Gammaa[m,:])
    #   Eta_All[ite,:,:] = etaa
    Eta_All = np.ones((num_H,M,K))
    return Eta_All


# Graph Data

def get_cg(n):
    adj = []
    for i in range(0,n):
        for j in range(0,n):
            if(not(i==j)):
                adj.append([i,j])
    return adj

def build_graph(K, H, Phii, adj, Y, pos_AP, pos_sample, Power_scale):


    x1 = np.expand_dims(H[pos_AP,:].T, axis = 1)
    x2 = np.random.rand(x1.shape[0],1)
    x = np.concatenate((x1,Phii,x2), axis = 1)
    x = torch.tensor(x, dtype = torch.float)
    edge_attr = []
    for e in adj:
      edge_attr.append([H[pos_AP,e[0]],H[pos_AP,e[1]]])

    edge_index = torch.tensor(adj, dtype = torch.long)
    edge_attr = torch.tensor(edge_attr, dtype = torch.float)

    y = np.zeros((K,1))
    y = np.expand_dims(Y[pos_AP,:].T, axis = 1)

    y = torch.tensor(y, dtype = torch.float)

    pos = torch.tensor(pos_sample, dtype = torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(),edge_attr = edge_attr, y = y, pos = pos)
    return data


def proc_data(BETAA_ALL,Phii_All,Eta_All,Power_scale): # Power_scale: Pd, Pu
    n = BETAA_ALL.shape[0]
    K = BETAA_ALL.shape[2]
    M = BETAA_ALL.shape[1]
    data_list = []
    cg = get_cg(K) # get adjacent matrix
    for i in range(n):
      for m in range(M):
        data = build_graph(K, BETAA_ALL[i], Phii_All[i], cg, Eta_All[i], m, i, Power_scale)
        data_list.append(data)
    return data_list

