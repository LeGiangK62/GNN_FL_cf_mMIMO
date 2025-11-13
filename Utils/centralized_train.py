import torch
from Utils.comm import power_from_raw, variance_calculate, rate_calculation, component_calculate, rate_from_component
import torch.nn.functional as F
# Centralized Training





def cen_loss_function(graphData, nodeFeatDict, edgeDict, tau, rho_p, rho_d, num_antenna, epochRatio=1, eval_mode=False):
    num_graph = graphData.num_graphs

    num_APs = graphData['AP'].x.shape[0]//num_graph
    num_UEs = graphData['UE'].x.shape[0]//num_graph
    
    large_scale = edgeDict['AP','down','UE'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,0]
    large_scale = torch.expm1(large_scale)
    power_matrix_raw = edgeDict['AP','down','UE'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,-1]
    ap_gate = nodeFeatDict['AP'].reshape(num_graph, num_APs, -1)
    phi_matrix = graphData['UE'].x.reshape(num_graph, num_UEs, -1)
    channel_var = variance_calculate(large_scale, phi_matrix, tau=tau, rho_p=rho_p)
    
    # p_max = (1.0 / num_antenna) ** 0.5
    # den = torch.logsumexp(power_matrix_raw + torch.log(channel_var), dim=2, keepdim=True)
    # term_1 = torch.exp(0.5 * (power_matrix_raw-den))
    # term_2 = torch.sigmoid(torch.sum(power_matrix_raw, dim=2, keepdim=True))
    # term_2 = term_2 ** 0.5
    # power_matrix = p_max  * term_1 # * term_2 # Sqrt of power 
    power_matrix = power_from_raw(power_matrix_raw, ap_gate, channel_var, num_antenna)
    
    # print(torch.sum(power_matrix_raw, dim=2, keepdim=True)[0])
    # rate = rate_calculation(power_matrix, large_scale, channel_var, phi_matrix, rho_d, num_antenna)
    
    all_DS, all_PC, all_UI = component_calculate(power_matrix, channel_var, large_scale, phi_matrix, rho_d=rho_d)
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna, rho_d=rho_d)
    
    if torch.isnan(rate).any():
        print(power_matrix_raw)
        raise ValueError('Nan in rate')
    
    
    if eval_mode:
        min_rate, _ = torch.min(rate, dim=1)
        full = torch.ones_like(power_matrix)
        rate_full_one = rate_calculation(full, large_scale, channel_var, phi_matrix, rho_d, num_antenna)
        min_rate_one, _ = torch.min(rate_full_one, dim=1)
        return min_rate, min_rate_one
    else:
        min_rate, _ = torch.min(rate, dim=1)
        loss = torch.mean(-min_rate)
        
        # T = 0.1
        # soft_min = -T * torch.logsumexp(-rate / T, dim=1)
        # if epochRatio <= 1:
        #     loss = epochRatio * torch.mean(-min_rate) + (1-epochRatio) * torch.mean(-soft_min)
        # else:
        #     loss = torch.mean(-min_rate)
        
        # mean_rate = torch.mean(rate, dim=1)
        # loss = (torch.mean(-min_rate) + torch.mean(-mean_rate))/2
        
        return loss, torch.mean(min_rate.detach())


def cen_train( epochRatio,
        dataLoader, model, optimizer,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        optimizer.zero_grad(set_to_none=True) 
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        loss, _ = cen_loss_function(
            batch, x_dict, edge_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            epochRatio=epochRatio
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs


@torch.no_grad()
def cen_eval(
        dataLoader, model,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_min_rate = 0.0
    total_graphs = 0
    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        _, min_rate = cen_loss_function(
            batch, x_dict, edge_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
        )
        
        total_min_rate += min_rate.item() * num_graph
        total_graphs += num_graph

    return total_min_rate/total_graphs 


