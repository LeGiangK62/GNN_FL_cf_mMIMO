import torch
import copy
import numpy as np


# 2. Training and Testing function

# 3. Federated Learning




# @torch.no_grad()
# def fl_eval(
#         dataLoader, responseInfo, model,
#         tau, rho_p, rho_d, num_antenna
#     ):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.eval()
    
#     total_loss = 0.0
#     total_graphs = 0
#     for batch, response in zip(dataLoader , responseInfo):
#         batch = batch.to(device)
#         num_graph = batch.num_graphs
#         x_dict, edge_dict, edge_index = model(batch)
#         loss = loss_function(
#             batch, x_dict, response, 
#             tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
#         )
        
#         total_loss += loss.item() * num_graph
#         total_graphs += num_graph

#     return -total_loss/total_graphs 




