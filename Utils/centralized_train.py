import torch
from Utils.comm import variance_calculate, rate_calculation, component_calculate, rate_from_component

# Centralized Training


def cen_loss_function(graphData, nodeFeatDict, edgeDict, tau, rho_p, rho_d, num_antenna, eval_mode=False):
    num_graph = graphData.num_graphs

    num_APs = graphData['AP'].x.shape[0]//num_graph
    num_UEs = graphData['UE'].x.shape[0]//num_graph
    
    large_scale_mean, large_scale_std = graphData.mean, graphData.std


    large_scale = edgeDict['AP','down','UE'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,0]
    large_scale = large_scale * large_scale_std + large_scale_mean
    power_matrix = edgeDict['AP','down','UE'].reshape(num_graph, num_APs, num_UEs, -1)[:,:,:,1]
    phi_matrix = graphData['UE'].x.reshape(num_graph, num_UEs, -1)
    channel_var = variance_calculate(large_scale, phi_matrix, tau=tau, rho_p=rho_p)
    sum_weighted = torch.sum(power_matrix * channel_var, dim=1, keepdim=True)   # shape (M,1)
    power_matrix = power_matrix / torch.maximum(sum_weighted, torch.ones_like(sum_weighted)/num_antenna)
    # power_matrix = torch.softmax(power_matrix, dim=1)
    # power_matrix = power_matrix/channel_var
    
    # rate = rate_calculation(power_matrix, large_scale, channel_var, phi_matrix, rho_d, num_antenna)
    
    all_DS, all_PC, all_UI = component_calculate(power_matrix, channel_var, large_scale, phi_matrix, rho=rho_d)
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)
    
    
    min_rate, _ = torch.min(rate, dim=1)
    if eval_mode:
        full = torch.ones_like(power_matrix)
        rate_full_one = rate_calculation(full, large_scale, channel_var, phi_matrix, rho_d, num_antenna)
        min_rate_one, _ = torch.min(rate_full_one, dim=1)
        return min_rate, min_rate_one
    else:
        return torch.neg(torch.mean(min_rate))


def cen_train(
        dataLoader, model, optimizer,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    total_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        loss = cen_loss_function(
            batch, x_dict, edge_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
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
    
    total_loss = 0.0
    total_graphs = 0
    for batch in dataLoader:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        x_dict, edge_dict, edge_index = model(batch)
        loss = cen_loss_function(
            batch, x_dict, edge_dict,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
        )
        
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return -total_loss/total_graphs 


