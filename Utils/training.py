import torch
import copy
import numpy as np
from Utils.synthetic_graph import return_graph, combine_graph


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

def loss_function_old(data_trainn, out_trainn, K, M, sr_weight, nsu_weight, Rthr, tau):
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

def train(dataLoader, complementLoader, model, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_loss = 0.0
    total_graphs = 0
    for batch, complement in zip(dataLoader, complementLoader):
        batch = batch.to(device)
        complement = complement.to(device)
        batch = combine_graph(batch, complement)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        loss = loss_function(batch, x_dict, edge_dict)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs 


@torch.no_grad()
def eval(dataLoader, complementLoader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_loss = 0.0
    total_graphs = 0
    for batch, complement in zip(dataLoader, complementLoader):
        batch = batch.to(device)
        complement = complement.to(device)
        batch = combine_graph(batch, complement)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        loss = loss_function(batch, x_dict, edge_dict)
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return -total_loss/total_graphs 


# 3. Federated Learning

def average_weights(local_weights):
    """Average model parameters from all clients (FedAvg)."""
    avg_weights = copy.deepcopy(local_weights[0])
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



def loss_function(graphData, nodeFeatDict, clientResponse, tau, rho_p, rho_d, num_antenna):
    num_graphs = graphData.num_graphs
    num_UEs = graphData['UE'].x.shape[0]//num_graphs
    num_APs = graphData['AP'].x.shape[0]//num_graphs
    
    pilot_matrix = graphData['UE'].x.reshape(num_graphs, num_UEs, -1)
    large_scale = graphData['AP','down','UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs)

    power = nodeFeatDict['UE'].reshape(num_graphs, num_UEs, -1)
    power_matrix = power[:,:,-1][:, None, :]

    channel_variance = variance_calculate(large_scale, pilot_matrix, tau, rho_p)

    DS_k, PC_k, UI_k = component_calculate(power_matrix, channel_variance, large_scale, pilot_matrix, rho=rho_d)
    
    all_DS = [DS_k] + [r['DS'] for r in clientResponse]
    all_PC = [PC_k] + [r['PC'] for r in clientResponse]
    all_UI = [UI_k] + [r['UI'] for r in clientResponse]

    all_DS = torch.cat(all_DS, dim=1)
    all_PC = torch.cat(all_PC, dim=1)   
    all_UI = torch.cat(all_UI, dim=1) 
    
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)
    min_rate,_ = torch.min(rate, dim=1)
    loss = torch.neg(min_rate) 

    return torch.mean(loss)

def rate_from_component(desiredSignal, pilotContamination, userInterference, numAntenna):
    num_graphs, num_APs, num_UEs = desiredSignal.shape
    devcie = desiredSignal.device
    dtype = desiredSignal.dtype
    
    sum_DS = desiredSignal.sum(dim=1)  
    num = (numAntenna**2) * (sum_DS ** 2) 

    sum_PC = pilotContamination.sum(dim=1)
    sum_UI = userInterference.sum(dim=1)  

    term1 = (numAntenna**2) * ((sum_PC * (1 - torch.eye(num_UEs, device=devcie))).pow(2).sum(dim=1)) 
    term2 = numAntenna * sum_UI.sum(dim=1)          
    denom = term1 + term2 + 1

    rate_all = torch.log2(1 + num/denom)  

    return rate_all
    

def component_calculate(power, channelVariance, largeScale, phiMatrix, rho=0.1):
    #################
    # power                 : torch.rand(num_graphs, num_AP, num_UE)
    # channelVariance       : torch.rand(num_graphs, num_AP, num_UE)
    # largeScale            : torch.rand(num_graphs, num_AP, num_UE)
    # phiMatrix             : torch.rand(num_graphs, num_UE, tau)
    #################
    device = power.device
    
    pilotContamination = torch.bmm(
        phiMatrix,
        phiMatrix.transpose(1, 2),
    ).abs()
    
    DS_all = torch.sqrt(rho * power) * channelVariance 

    tmp = rho * power * channelVariance
    tmp = tmp.unsqueeze(-1)
    largeScale_expand = largeScale.unsqueeze(-2)
    UI_all = tmp * largeScale_expand

    mask = torch.eye(UI_all.size(-1), device=device).bool()
    UI_all[:, :, mask] = 0

    tmp = torch.sqrt(rho * power) * channelVariance / largeScale
    tmp = tmp.unsqueeze(-1)

    tmp = tmp * largeScale_expand
    PC_all = tmp * pilotContamination.unsqueeze(-3)
    mask = torch.eye(PC_all.size(-1), device=device).bool()
    PC_all[:, :, mask] = 0


    return DS_all, PC_all, UI_all


def package_calculate(batch, x_dict, tau, rho_p, rho_d):
    num_graphs = batch.num_graphs
    num_UEs = x_dict['UE'].shape[0] // num_graphs
    num_APs = x_dict['AP'].shape[0] // num_graphs
    ue_feature = x_dict['UE'].reshape(num_graphs, num_UEs, -1)
    power = ue_feature[:,:, -1][:,None,:]
    phiMatrix = ue_feature[:,:, :-1]

    largeScale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs)

    channelVariance = variance_calculate(largeScale, phiMatrix, tau, rho_p)

    
    return component_calculate(power, channelVariance, largeScale, phiMatrix, rho=rho_d)


def variance_calculate(largeScale, phiMatrix, tau, rho_p):
    num = tau*rho_p * torch.square(largeScale) 

    tmp = torch.square(torch.bmm(
        phiMatrix,
        phiMatrix.transpose(1, 2),
    ).abs())

    largeScale_exp = largeScale.unsqueeze(-1)  # Shape: (num_graphs, num_AP, num_UE, 1)
    tmp_exp = tmp.unsqueeze(1)
    term1 = torch.sum(largeScale_exp * tmp_exp, dim=2)
    denom = tau * rho_p * term1 + 1

    return num/denom


def fl_train(
        dataLoader, responseInfo, model, optimizer,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_loss = 0.0
    total_graphs = 0
    for batch, response in zip(dataLoader , responseInfo):
        batch = batch.to(device)
        num_graph = batch.num_graphs
        x_dict, edge_dict, edge_index = model(batch)
        loss = loss_function(
            batch, x_dict, response, 
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
        )
        loss.backward()
        optimizer.step()
    
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs 


@torch.no_grad()
def fl_eval(
        dataLoader, responseInfo, model,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_loss = 0.0
    total_graphs = 0
    for batch, response in zip(dataLoader , responseInfo):
        batch = batch.to(device)
        num_graph = batch.num_graphs
        x_dict, edge_dict, edge_index = model(batch)
        loss = loss_function(
            batch, x_dict, response, 
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
        )
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return -total_loss/total_graphs 


def get_global_info(
        loaderData, localModels, optimizers,
        tau, rho_p, rho_d
    ):
    num_client = len(localModels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    send_to_server = [[] for _ in range(num_client)] 
    for batches  in zip(*loaderData):                       # sync step across APs
        
        for client_idx, (model, opt, batch) in enumerate(zip(localModels, optimizers, batches)):
            # Check batch here? something wrong?
            model.eval()
            batch = batch.to(device)
            
            x_dict, edge_dict, edge_index = model(batch)
            
            DS_single, PC_single, UI_single = package_calculate(batch, x_dict, tau, rho_p, rho_d)

            send_to_server[client_idx].append({
                'DS': DS_single.detach(),
                "PC": PC_single.detach(),
                "UI": UI_single.detach()
            })
            
    return send_to_server

def distribute_global_info(send_to_server):
    num_client = len(send_to_server)
    response_all = []
    for client_idx in range(num_client):
        # for AP i, responses = list of lists of other APs’ packages
        responses_this_ap = []
        for batch_idx in range(len(send_to_server[client_idx])):
            # all APs except itself for this batch index
            other_responses = [
                send_to_server[j][batch_idx]
                for j in range(num_client) if j != client_idx
            ]
            responses_this_ap.append(other_responses)
        response_all.append(responses_this_ap)
    return response_all