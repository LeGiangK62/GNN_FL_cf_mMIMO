import torch
import copy
import numpy as np


# Loss function
def sr_loss_matrix(data_trainn, out_trainn, K, M, sr_weight, nsu_weight, Rthr, tau):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Ther_noise = 20000000 * 10**(-17.4) * 10**-3
    Pp = 1/Ther_noise
    NoAntennas = 10
    N_graph = len(data_trainn[0].pos)
    K = int(len(data_trainn[0].x)/N_graph)
    DS_all = torch.zeros((M,N_graph,K,1))
    PC_all = torch.zeros((M,N_graph,K,K))
    UI_all = torch.zeros((M,N_graph,K,1))
    for m in range(M):
        # Calculate on the m-th AP for all sample of a batch
        TermBETAA = data_trainn[m].x[:,0].unsqueeze(1).view(N_graph,K,1)
        TermPhii = data_trainn[m].x[:,1:-1].view(N_graph,K,10)
        TermEtaa = out_trainn[m][:,-1].unsqueeze(1).view(N_graph,K,1)

        Phii_Multiplication = torch.stack([torch.square(torch.matmul(TermPhii[ite], TermPhii[ite].t())) for ite in range(N_graph)])

        BETAA_repeat_row = torch.stack([TermBETAA[ite].t().repeat(K,1) for ite in range(N_graph)]) # a row is a vector of channel

        # Gamma calculation
        Denom_matrix = tau * torch.sum(torch.mul(Phii_Multiplication, BETAA_repeat_row), dim=2, keepdim=True) + 1
        Gamma = tau * (torch.square(TermBETAA)/Denom_matrix)

        # Pilot Contamination
        BETAA_repeat_col = torch.transpose(BETAA_repeat_row, 1, 2) # a column is a vector of channel
        BETAA_divide = torch.div(BETAA_repeat_col,BETAA_repeat_row)
        Gamma_repeat_col = torch.stack([Gamma[ite].t().repeat(K,1) for ite in range(N_graph)])
        TermEtaa_repeat = torch.stack([TermEtaa[ite].t().repeat(K,1) for ite in range(N_graph)])
        TermEtaa_repeat_square_root = torch.sqrt(TermEtaa_repeat)
        mask = 1-torch.eye(K)
        PC = TermEtaa_repeat_square_root * Gamma_repeat_col * BETAA_divide * Phii_Multiplication**(1/2) * (mask.to(device))
        PC_all[m,:,:,:] = PC

        # User Interference
        SumPower = torch.sum(TermEtaa * Gamma, dim = 1, keepdim = True)
        Sum_expand = SumPower.expand(N_graph,K,1)
        UI = Sum_expand * TermBETAA
        UI_all[m,:,:,:] = UI

        # Desire signal calculation
        DS = torch.sqrt(TermEtaa) * Gamma
        DS_all[m,:,:,:] = DS

    # Rate Calculation
    Num = torch.sum(DS_all, dim = 0)**2
    UI1 = torch.sum(UI_all, dim = 0)
    PC1 = torch.sum(PC_all, dim = 0)**2
    PC2 = torch.sum(PC1, dim=-1, keepdim=True)
    R_batch = torch.log2(1 + (NoAntennas**2) * Num/((NoAntennas**2) * PC2 + NoAntennas * UI1 + 1/Pp))
    R_all = torch.transpose(torch.squeeze(R_batch), 0, 1)
    #   print(R_all[:,0].shape)
    # rate_min = torch.mean(torch.min(R_all,axis = 0)[0])
    # loss = torch.neg(rate_min)
    sum_rate = torch.mean(torch.sum(R_all,0))
    # Compute number of users exceeding the threshold for each sample
    satisfied_users = (R_all > Rthr).float()  # Boolean mask -> Float (1 if satisfied, 0 otherwise)
    #   print(satisfied_users)
    num_satisfied = torch.mean(torch.sum(satisfied_users, 0))  # Average over batch size
    #   print(num_satisfied)
    loss = torch.neg(sr_weight*sum_rate + nsu_weight*num_satisfied)
    #   print(sum_rate)
    #   print(num_satisfied)
    #   print(loss)
    #   aaa
    return loss, sum_rate, num_satisfied

# 1. Loss function
# Max-min Fairness Downlink

def max_min_loss_singleAP(data, power):
    return 1

def loss_function(data_trainn, out_trainn, K, M, sr_weight, nsu_weight, Rthr, tau):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Ther_noise = 20000000 * 10**(-17.4) * 10**-3
    Pp = 1/Ther_noise
    NoAntennas = 10
    N_graph = len(data_trainn[0].pos)
    K = int(len(data_trainn[0].x)/N_graph)
    DS_all = torch.zeros((M,N_graph,K,1))
    PC_all = torch.zeros((M,N_graph,K,K))
    UI_all = torch.zeros((M,N_graph,K,1))
    for m in range(M):
        # Calculate on the m-th AP for all sample of a batch
        TermBETAA = data_trainn[m].x[:,0].unsqueeze(1).view(N_graph,K,1)
        TermPhii = data_trainn[m].x[:,1:-1].view(N_graph,K,10)
        TermEtaa = out_trainn[m][:,-1].unsqueeze(1).view(N_graph,K,1)

        Phii_Multiplication = torch.stack([torch.square(torch.matmul(TermPhii[ite], TermPhii[ite].t())) for ite in range(N_graph)])

        BETAA_repeat_row = torch.stack([TermBETAA[ite].t().repeat(K,1) for ite in range(N_graph)]) # a row is a vector of channel

        # Gamma calculation
        Denom_matrix = tau * torch.sum(torch.mul(Phii_Multiplication, BETAA_repeat_row), dim=2, keepdim=True) + 1
        Gamma = tau * (torch.square(TermBETAA)/Denom_matrix)

        # Pilot Contamination
        BETAA_repeat_col = torch.transpose(BETAA_repeat_row, 1, 2) # a column is a vector of channel
        BETAA_divide = torch.div(BETAA_repeat_col,BETAA_repeat_row)
        Gamma_repeat_col = torch.stack([Gamma[ite].t().repeat(K,1) for ite in range(N_graph)])
        TermEtaa_repeat = torch.stack([TermEtaa[ite].t().repeat(K,1) for ite in range(N_graph)])
        TermEtaa_repeat_square_root = torch.sqrt(TermEtaa_repeat)
        mask = 1-torch.eye(K)
        PC = TermEtaa_repeat_square_root * Gamma_repeat_col * BETAA_divide * Phii_Multiplication**(1/2) * (mask.to(device))
        PC_all[m,:,:,:] = PC

        # User Interference
        SumPower = torch.sum(TermEtaa * Gamma, dim = 1, keepdim = True)
        Sum_expand = SumPower.expand(N_graph,K,1)
        UI = Sum_expand * TermBETAA
        UI_all[m,:,:,:] = UI

        # Desire signal calculation
        DS = torch.sqrt(TermEtaa) * Gamma
        DS_all[m,:,:,:] = DS

    # Rate Calculation
    Num = torch.sum(DS_all, dim = 0)**2
    UI1 = torch.sum(UI_all, dim = 0)
    PC1 = torch.sum(PC_all, dim = 0)**2
    PC2 = torch.sum(PC1, dim=-1, keepdim=True)
    R_batch = torch.log2(1 + (NoAntennas**2) * Num/((NoAntennas**2) * PC2 + NoAntennas * UI1 + 1/Pp))
    
    R_all = torch.transpose(torch.squeeze(R_batch), 0, 1)
    R_min, _ = torch.min(R_all, axis=0)
    loss = torch.neg(torch.mean(R_min))
    
    return loss


# 2. Training and Testing function

def train(dataLoader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        power = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = loss_function(power, batch.edge_attr)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs 


@torch.no_grad()
def eval(dataLoader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        power = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        loss = loss_function(power, batch.edge_attr_dict)
        
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return -total_loss/total_graphs 


# 3. Federated Learning

def average_weights(local_weights):
    """
    Perform FedAvg: average model weights from all clients (APs).
    local_weights: list of state_dicts from each local model.
    Returns averaged state_dict.
    """
    # deep copy weights of first client
    avg_weights = copy.deepcopy(local_weights[0])

    # iterate over all keys
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))

    return avg_weights


# ==================================================
## loop pilot contamination

# pilot_contamination_loop = torch.zeros(((num_graphs, num_UE, num_UE)), device=device)
# for s in range(num_graphs):
#     for k in range(num_UE):
#         for k_prime in range(num_UE):
#             # if k_prime == k: continue
#             pilot_contamination_loop[s, k, k_prime] = pilot_assignment[s,k,:] @ pilot_assignment[s,k_prime,:].T
# # pilot_contamination_loop = (pilot_contamination_loop.abs() > threshold).float()
# # pilot_contamination_loop - pilot_contamination
# pilot_contamination_loop.abs()
# ==================================================


def rate_calculation(powerMatrix, largeScale, channelVariance, pilotAssignment):
    #===========================================
    # 
    # Args:
    # powerMatrix:        power matrix of all AP and UE   [num_samples, num_AP, num_UE]
    # largeScale:         channel large scale fading      [num_samples, num_AP, num_UE]
    # channelVariance:    channel variance                [num_samples, num_AP, num_UE]
    # pilotAssignment:    Pilot Assignment                [num_samples, num_UE, pilot_length]
    # Output
    # rate:               Achievable rate of every UE     [num_samples, num_UE]
    #
    #===========================================
    SINR_num = torch.sum(powerMatrix*channelVariance, dim=1) ** 2

    powerExpanded = (powerMatrix*channelVariance).unsqueeze(-1)
    largeScaleExpanded = largeScale.unsqueeze(-2)
    userInterference = torch.sum(powerExpanded * largeScaleExpanded, dim=(1, 2))

    interm_var1 = (powerMatrix/ largeScale).unsqueeze(-1)
    interm_var2 = (channelVariance * largeScale).unsqueeze(-2)
    prod = torch.sum(interm_var1 * interm_var2, dim=1) ** 2
    diag_vec = prod.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * torch.eye(powerMatrix.shape[2], device=powerMatrix.device)
    
    pilotContamination = torch.bmm(
        pilotAssignment,
        pilotAssignment.transpose(1, 2),
    ).abs()

    pilotContamination = (prod - diag_vec) * pilotContamination
    pilotContamination = torch.sum(pilotContamination, dim=1)


    SINR_denom = 1 + userInterference + pilotContamination
    rate = torch.log2(1 + SINR_num/SINR_denom)
    return rate

