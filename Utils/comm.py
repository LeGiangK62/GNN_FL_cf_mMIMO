import torch 


# Common functions

def variance_calculate(largeScale, phiMatrix, tau, rho_p):
    num = tau * torch.square(largeScale) 

    tmp = torch.square(torch.bmm(
        phiMatrix,
        phiMatrix.transpose(1, 2),
    ).abs())

    largeScale_exp = largeScale.unsqueeze(-1)  # Shape: (num_graphs, num_AP, num_UE, 1)
    tmp_exp = tmp.unsqueeze(1)
    term1 = torch.sum(largeScale_exp * tmp_exp, dim=2)
    denom = tau * term1 + 1

    return num/denom


def rate_calculation(powerMatrix, largeScale, channelVariance, pilotAssignment, rho_d, num_antenna):
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
    powerMatrix = torch.sqrt(powerMatrix)
    SINR_num = torch.sum(powerMatrix*channelVariance, dim=1) ** 2 * (num_antenna ** 2)

    powerExpanded = ((powerMatrix**2) *channelVariance).unsqueeze(-1)
    largeScaleExpanded = largeScale.unsqueeze(-2)
    userInterference = torch.sum(powerExpanded * largeScaleExpanded, dim=(1, 2))

    interm_var1 = (powerMatrix * channelVariance/ largeScale).unsqueeze(-1)
    interm_var2 = largeScale.unsqueeze(-2)
    prod = torch.sum(interm_var1 * interm_var2, dim=1) ** 2
    diag_vec = prod.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * torch.eye(powerMatrix.shape[2], device=powerMatrix.device)
    
    pilotContamination = torch.bmm(
        pilotAssignment,
        pilotAssignment.transpose(1, 2),
    ).abs()

    pilotContamination = (prod - diag_vec) * pilotContamination
    pilotContamination = torch.sum(pilotContamination, dim=1)


    SINR_denom = 1 + userInterference * num_antenna + pilotContamination * num_antenna ** 2
    rate = torch.log2(1 + SINR_num/SINR_denom)
    return rate

# Only FL functions

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
    
    DS_all = torch.sqrt(power) * channelVariance 

    tmp = power * channelVariance
    tmp = tmp.unsqueeze(-1)
    largeScale_expand = largeScale.unsqueeze(-2)
    UI_all = tmp * largeScale_expand

    mask = torch.eye(UI_all.size(-1), device=device).bool()
    UI_all[:, :, mask] = 0

    tmp = torch.sqrt(power) * channelVariance / largeScale
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