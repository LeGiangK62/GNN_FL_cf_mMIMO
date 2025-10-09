import torch
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