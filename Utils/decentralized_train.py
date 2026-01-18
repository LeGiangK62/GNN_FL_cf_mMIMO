import re
import torch
import copy 
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from Utils.data_gen import full_het_graph
from Utils.comm import (
    variance_calculate, rate_calculation, 
    component_calculate, rate_from_component,
    power_from_raw
)


def loss_function(graphData, nodeFeatDict, edgeDict, clientResponse, tau, rho_p, rho_d, num_antenna, epochRatio=1):
    """
    Compute loss for FL training.

    Args:
        graphData: HeteroData batch
        nodeFeatDict: Node features from model
        edgeDict: Edge features from model
        clientResponse: List of dicts with DS/PC/UI from other clients
        tau, rho_p, rho_d, num_antenna: Communication parameters
        epochRatio: Training progress ratio (unused)

    Returns:
        loss: Scalar loss value
        min_rate_detach: [B] tensor of min rates for monitoring
    """
    num_graphs = graphData.num_graphs
    num_UEs = graphData['UE'].x.shape[0]//num_graphs
    num_APs = graphData['AP'].x.shape[0]//num_graphs

    pilot_matrix = graphData['UE'].x[:,:tau].reshape(num_graphs, num_UEs, -1)

    large_scale = graphData['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
    power_matrix_raw = edgeDict['AP','down','UE'].reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]

    large_scale = torch.expm1(large_scale)
    channel_variance = graphData['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]

    power_matrix_raw = power_matrix_raw[:,:1,:]
    channel_variance = channel_variance[:,:1,:]
    large_scale = large_scale[:,:1,:]

    power_matrix = power_from_raw(power_matrix_raw, channel_variance, num_antenna)
    DS_k, PC_k, UI_k = component_calculate(power_matrix, channel_variance, large_scale, pilot_matrix, rho_d=rho_d)

    all_DS = [DS_k] + [r['DS'] for r in clientResponse]
    all_PC = [PC_k] + [r['PC'] for r in clientResponse]
    all_UI = [UI_k] + [r['UI'] for r in clientResponse]

    all_DS = torch.cat(all_DS, dim=1)
    all_PC = torch.cat(all_PC, dim=1)
    all_UI = torch.cat(all_UI, dim=1)

    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)  # [B, K]
    min_rate_detach, _ = torch.min(rate.detach(), dim=1)

    # === LOSS OPTIONS ===
    # Option 1: Hard min (baseline) - 0.757
    min_rate, _ = torch.min(rate, dim=1)
    loss = -min_rate.mean()

    # Option 2: Soft-min - smoother but similar
    # T = 0.7
    # soft_min = -T * torch.logsumexp(-rate / T, dim=1)
    # loss = -soft_min.mean()

    # Option 3: Rate-weighted loss - prioritize low-rate UEs smoothly
    # Each UE's rate contributes to loss, weighted inversely by its rate
    # with torch.no_grad():
    #     # Weight = softmax(-rate/T) → low rate UEs get higher weight
    #     T_weight = 0.5
    #     ue_weights = torch.softmax(-rate / T_weight, dim=1)  # [B, K]

    # # Weighted sum of negative rates
    # weighted_rate = (ue_weights * rate).sum(dim=1)  # [B]
    # loss = -weighted_rate.mean()

    return loss, min_rate_detach




def fl_train(
        dataLoader, responseInfo, model, optimizer,
        tau, rho_p, rho_d, num_antenna, epochRatio=1
    ):
    """
    Train a local FL model for one epoch.

    Args:
        dataLoader: List of graph batches for this client
        responseInfo: List of rate_pack (DS/PC/UI from other clients)
        model: Local model
        optimizer: Local optimizer
        tau, rho_p, rho_d, num_antenna: Communication parameters
        epochRatio: Training progress ratio (unused)

    Returns:
        avg_loss, avg_min_rate, local_gradients, all_min_rates
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    local_gradients = {}
    all_min_rates = []

    total_loss = 0.0
    total_min_rate = 0.0
    total_graphs = 0

    for batch, response in zip(dataLoader, responseInfo):
        batch = batch.to(device)
        num_graph = batch.num_graphs
        optimizer.zero_grad()
        x_dict, attr_dict, _ = model(batch, isRawData=False)
        loss, min_rate = loss_function(
            batch, x_dict, attr_dict, response,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            epochRatio=epochRatio
        )
        loss.backward()
        optimizer.step()

        # Collect min_rate per sample [B] for server aggregation
        all_min_rates.append(min_rate.detach().cpu())

        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in local_gradients:
                    local_gradients[name] = param.grad.clone()
                else:
                    local_gradients[name] += param.grad.clone()

        total_loss += loss.item() * num_graph
        total_min_rate += min_rate.mean().item() * num_graph
        total_graphs += num_graph

    # FedSoftMin - Normalize gradients by number of batches
    num_batches = len(dataLoader)
    for name in local_gradients:
        local_gradients[name] = local_gradients[name] / num_batches

    # Concatenate all min_rates: [total_samples]
    all_min_rates = torch.cat(all_min_rates, dim=0)

    return total_loss/total_graphs, total_min_rate/total_graphs, local_gradients, all_min_rates

@torch.no_grad()
def fl_eval(
        dataLoader, local_models,
        tau, rho_p, rho_d, num_antenna
    ):

    send_to_server = get_global_info(
        dataLoader, local_models,
        tau=tau, rho_p=rho_p, rho_d=rho_d,
        num_antenna=num_antenna
    )

    all_DS = torch.cat([
        torch.stack([client_data['DS'][:,0,:] for client_data in batch_clients], dim=1)
        for batch_clients in zip(*send_to_server)
    ], dim=0)

    all_PC = torch.cat([
        torch.stack([client_data['PC'][:,0,:,:] for client_data in batch_clients], dim=1)
        for batch_clients in zip(*send_to_server)
    ], dim=0)

    all_UI = torch.cat([
        torch.stack([client_data['UI'][:,0,:,:] for client_data in batch_clients], dim=1)
        for batch_clients in zip(*send_to_server)
    ], dim=0)
    
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)

    min_rate, _ = torch.min(rate, dim=1)
    
    # if eval_mode: 
    #     return min_rate
    # else:
    # min_rate = torch.mean(min_rate)

    return min_rate

# @torch.no_grad()
# def fl_eval_rate_old(
#         dataLoader, models,
#         tau, rho_p, rho_d, num_antenna,
#         eval_mode=False
#     ):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     all_power = []
#     all_large_scale = []
#     all_phi = []
#     all_channel_var = []
    
#     # send_to_server = get_global_info(
#     #     dataLoader, models,  
#     #     tau=tau, rho_p=rho_p, rho_d=rho_d, 
#     #     num_antenna=num_antenna
#     # )
#     # kg_dataLoader = kg_augmentation(dataLoader, send_to_server, tau)
    
#     for batch_idx, batches_at_k in enumerate(zip(*dataLoader)):
#         per_batch_channel_var = []
#         per_batch_power = []
#         per_batch_large_scale = []
#         per_batch_phi = []
#         for ap_idx, (model, batch) in enumerate(zip(models, batches_at_k)):
#             model.eval()
#             # iterate over all batch of each AP
#             batch = batch.to(device)
#             num_graphs = batch.num_graphs
#             num_UEs = batch['UE'].x.shape[0]//num_graphs
#             num_APs = batch['AP'].x.shape[0]//num_graphs
            
#             x_dict, edge_dict, edge_index = model(batch)

#             large_scale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
#             local_power_matrix_raw = edge_dict['AP','down','UE'].reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
                
#             large_scale = torch.expm1(large_scale)
    
#             pilot_matrix = batch['UE'].x[:,:tau].reshape(num_graphs, num_UEs, -1)
#             # channel_variance = variance_calculate(large_scale, pilot_matrix, tau, rho_p)
#             channel_variance = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]
            
#             local_power_matrix_raw = local_power_matrix_raw[:,:1,:]
#             channel_variance = channel_variance[:,:1,:]
#             large_scale = large_scale[:,:1,:]
            
            
#             local_power_matrix = power_from_raw(local_power_matrix_raw, channel_variance, num_antenna)

#             per_batch_power.append(local_power_matrix)
#             per_batch_large_scale.append(large_scale)
#             per_batch_channel_var.append(channel_variance)
#             per_batch_phi.append(pilot_matrix.unsqueeze(1))
            
#         per_batch_phi = torch.cat(per_batch_phi, dim=1) 
#         per_batch_power = torch.cat(per_batch_power, dim=1)
#         per_batch_large_scale = torch.cat(per_batch_large_scale, dim=1)
#         per_batch_channel_var = torch.cat(per_batch_channel_var, dim=1)
        
#         if per_batch_phi.shape[1] > 1:
#             ref = per_batch_phi[:, 0, :, :]
#             if not torch.allclose(per_batch_phi[:, 1:, :, :],
#                                 ref.unsqueeze(1).expand_as(per_batch_phi[:, 1:, :, :]),
#                                 atol=1e-6, rtol=0):
#                 raise ValueError("UE/pilot order differs across clients for this batch. "
#                                 "Use a shared sampler or disable per-client shuffle.")
         
#         all_power.append(per_batch_power)
#         all_large_scale.append(per_batch_large_scale)
#         all_channel_var.append(per_batch_channel_var)
#         all_phi.append(per_batch_phi[:,0,:,:])
        
#     all_power = torch.cat(all_power, dim=0)
#     all_large_scale = torch.cat(all_large_scale, dim=0)
#     all_channel_var = torch.cat(all_channel_var, dim=0)
#     all_phi = torch.cat(all_phi, dim=0)
    
#     all_DS, all_PC, all_UI = component_calculate(all_power, all_channel_var, all_large_scale, all_phi, rho_d=rho_d)
#     rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)
#     min_rate, _ = torch.min(rate, dim=1)
    
#     if eval_mode: 
#         return min_rate
#     else:
#         min_rate = torch.mean(min_rate)
#         return min_rate
    

@torch.no_grad()
def fl_eval_rate(
        dataLoader, models,
        tau, rho_p, rho_d, num_antenna,
        eval_mode=False
    ):
    """
    Federated Learning evaluation with proper global aggregation.
    Key insight: Must aggregate contributions from ALL APs across clients,
    just like training does via clientResponse.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Get global information from all clients (same as training)
    send_to_server = get_global_info(
        dataLoader, models,
        tau=tau, rho_p=rho_p, rho_d=rho_d,
        num_antenna=num_antenna
    )

    # Step 2: Distribute responses to each client
    response_all = distribute_global_info(send_to_server)

    # Step 3: Each client computes rate with local + global info
    all_rates = []
    for client_idx, (model, batches, responses_ap) in enumerate(zip(models, dataLoader, response_all)):
        model.eval()

        for batch, response in zip(batches, responses_ap):
            batch = batch.to(device)
            num_graphs = batch.num_graphs
            num_UEs = batch['UE'].x.shape[0]//num_graphs
            num_APs = batch['AP'].x.shape[0]//num_graphs

            x_dict, edge_dict, _ = model(batch)

            # LOCAL contribution (1 AP per client)
            large_scale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
            local_power_raw = edge_dict['AP','down','UE'].reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
            large_scale = torch.expm1(large_scale)
            pilot_matrix = batch['UE'].x[:,:tau].reshape(num_graphs, num_UEs, -1)
            channel_var = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]

            # Each client controls only its own AP
            local_power_raw = local_power_raw[:,:1,:]
            channel_var = channel_var[:,:1,:]
            large_scale = large_scale[:,:1,:]

            local_power = power_from_raw(local_power_raw, channel_var, num_antenna)
            DS_local, PC_local, UI_local = component_calculate(local_power, channel_var, large_scale, pilot_matrix, rho_d=rho_d)

            # GLOBAL contributions from other APs (via server)
            all_DS = [DS_local] + [r['DS'] for r in response]
            all_PC = [PC_local] + [r['PC'] for r in response]
            all_UI = [UI_local] + [r['UI'] for r in response]

            all_DS = torch.cat(all_DS, dim=1)
            all_PC = torch.cat(all_PC, dim=1)
            all_UI = torch.cat(all_UI, dim=1)

            # Compute rate with full information (local + global)
            rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)
            min_rate, _ = torch.min(rate, dim=1)
            all_rates.append(min_rate)

    all_rates = torch.cat(all_rates)

    if eval_mode:
        return all_rates
    else:
        return torch.mean(all_rates)
    
    
    
    
    # total_min_rate = 0.0
    # total_samples = 0.0
    # for each_power, each_large_scale, each_phi, each_channel_variance in zip(all_power, all_large_scale, all_phi, all_channel_var):
    #     num_graphs = len(each_power)
    #     each_phi = each_phi[:,0,:,:]
    #     # each_channel_variance = variance_calculate(each_large_scale, each_phi, tau=tau, rho_p=rho_p)
    #     all_DS, all_PC, all_UI = component_calculate(each_power, each_channel_variance, each_large_scale, each_phi, rho_d=rho_d)
    #     # rate = rate_calculation(each_power, each_large_scale, each_channel_variance, each_phi, rho_d=rho_d, num_antenna=num_antenna)
    #     rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)
    #     min_rate, _ = torch.min(rate, dim=1)
    #     if eval_mode: return min_rate
    #     min_rate = torch.mean(min_rate)
    #     total_min_rate += min_rate.item() * num_graphs
    #     total_samples += num_graphs

    # return total_min_rate/total_samples


## FL functions ############################################################
def average_weights(local_weights):
    """Average model parameters from all clients (FedAvg)."""
    avg_weights = copy.deepcopy(local_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
    return avg_weights


# class FedAdam:
#     def __init__(self, global_model, client_fraction=0.3, seed=None,
#                  lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8):
#         self.client_fraction = client_fraction
#         if seed is not None:
#             random.seed(seed)
#         self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
#         self.t = 0  # step counter

#         # moments only for updatable float params (skip BN buffers and personal layers)
#         def _updatable(k, v):
#             if not torch.is_floating_point(v): return False
#             if 'convs_per' in k: return False
#             if any(s in k for s in ('running_mean','running_var','num_batches_tracked')): return False
#             return True

#         templ = global_model.state_dict()
#         self.m = {k: torch.zeros_like(v) for k, v in templ.items() if _updatable(k, v)}
#         self.v = {k: torch.zeros_like(v) for k, v in templ.items() if _updatable(k, v)}

#     @torch.no_grad()
#     def sample_clients(self, num_clients):
#         n = max(1, int(self.client_fraction * num_clients))
#         return sorted(random.sample(range(num_clients), n))

#     @torch.no_grad()
#     def aggregate(self, global_model, local_weights):
#         """
#         local_weights: list of state_dicts from SELECTED clients (already filtered in main).
#         """
#         # FedAvg over selected clients (skip BN/personal & non-float)
#         avg = copy.deepcopy(local_weights[0])
#         def _skip(k, v):
#             return (
#                 'convs_per' in k or
#                 not torch.is_floating_point(v) or
#                 any(s in k for s in ('running_mean','running_var','num_batches_tracked'))
#             )

#         for k in avg.keys():
#             # if _skip(k, avg[k]): continue
#             for w in local_weights[1:]:
#                 avg[k] += w[k]
#             avg[k] /= float(len(local_weights))

#         # "gradient" = global - avg
#         g = {}
#         for k, v in global_model.state_dict().items():
#             # if _skip(k, v): continue
#             g[k] = v - avg[k]

#         # Adam moments + bias correction
#         self.t += 1
#         b1t = 1.0 - (self.beta1 ** self.t)
#         b2t = 1.0 - (self.beta2 ** self.t)

#         new_state = {}
#         for k, v in global_model.state_dict().items():
#             # if _skip(k, v):
#             #     new_state[k] = v
#             #     continue
#             self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
#             self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g[k] * g[k])
#             m_hat = self.m[k] / b1t
#             v_hat = self.v[k] / b2t
#             new_state[k] = v - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

#         global_model.load_state_dict(new_state)
#         return copy.deepcopy(global_model.state_dict())
    

class FedSoftMin:
    """
    Server aggregates gradients weighted by soft-min contribution.
    Clients with lower rates get higher gradient weights.
    """
    def __init__(self, client_fraction=1.0, server_lr=1.0, temperature=0.7, momentum=0.9):
        self.server_lr = server_lr
        self.T = temperature
        self.momentum = momentum
        self.velocity = None
        self.client_fraction = client_fraction 
    
    @torch.no_grad()
    def sample_clients(self, num_clients):
        n = max(1, int(self.client_fraction * num_clients))
        return sorted(random.sample(range(num_clients), n))
    
    ########################################################################
    # Option A: Gradient-based aggregation (NOT WORKING WELL)
    # - Server dùng gradient để update weights
    # - Vấn đề: gradient đã được dùng bởi client optimizer rồi → double update
    ########################################################################
    # def aggregate_option_a(self, local_gradients, local_rates, global_weights, selected_clients):
    #     valid_grads = [local_gradients[i] for i in selected_clients if local_gradients[i] is not None]
    #     valid_rates = [local_rates[i] for i in selected_clients]
    #
    #     if len(valid_grads) == 0:
    #         return global_weights
    #
    #     rate_scores = torch.stack([r.mean() for r in valid_rates])
    #     weights = torch.softmax(-rate_scores / self.T, dim=0)
    #
    #     avg_grad = {}
    #     for key in valid_grads[0].keys():
    #         weighted_sum = sum(w * g[key] for w, g in zip(weights, valid_grads))
    #         avg_grad[key] = weighted_sum
    #
    #     if self.velocity is None:
    #         self.velocity = {}
    #
    #     new_weights = {}
    #     for key in global_weights.keys():
    #         if key in avg_grad:
    #             if key not in self.velocity:
    #                 self.velocity[key] = torch.zeros_like(avg_grad[key])
    #             self.velocity[key] = self.momentum * self.velocity[key] + avg_grad[key]
    #             new_weights[key] = global_weights[key] - self.server_lr * self.velocity[key]
    #         else:
    #             new_weights[key] = global_weights[key]
    #     return new_weights

    ########################################################################
    # Option B: Weight averaging with rate-based client weighting
    # - Client vẫn optimizer.step() như bình thường
    # - Server average WEIGHTS, nhưng weight mỗi client theo rate
    # - Client có rate thấp → weight cao hơn (được ưu tiên)
    ########################################################################
    def aggregate(self, local_weights, local_rates, selected_clients):
        """
        Weighted average of client weights based on their rates.
        Clients with lower rate get higher weight (prioritize worst performers).

        Args:
            local_weights: list of weight dicts from each client
            local_rates: list of rate tensors [num_samples] from each client
            selected_clients: indices of participating clients

        Returns:
            new_weights: weighted average of client weights
        """
        valid_weights = [local_weights[i] for i in selected_clients]
        valid_rates = [local_rates[i] for i in selected_clients]

        if len(valid_weights) == 0:
            return local_weights[0]  # Fallback

        # 1. Compute weight for each client based on their rate
        # Client với rate thấp → cần được ưu tiên → weight cao
        rate_scores = torch.stack([r.mean() for r in valid_rates])  # [num_selected]

        # Soft-min weighting: lower rate → higher weight
        client_weights = torch.softmax(-rate_scores / self.T, dim=0)  # [num_selected]

        # 2. Weighted average of WEIGHTS (not gradients)
        new_weights = {}
        for key in valid_weights[0].keys():
            stacked = torch.stack([w[key].float() for w in valid_weights], dim=0)  # [num_clients, ...]
            # Reshape client_weights for broadcasting
            weight_shape = [-1] + [1] * (stacked.dim() - 1)
            weighted_avg = (client_weights.view(*weight_shape).to(stacked.device) * stacked).sum(dim=0)
            new_weights[key] = weighted_avg

        return new_weights
    

class FedAdam:
    def __init__(self, client_fraction=0.3, lr=1e-3, beta1=0.9, beta2=0.99, eps=1e-8):
        self.client_fraction = client_fraction
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m = {}   # first moment
        self.v = {}   # second moment
        self.t = 0    # timestep

    @torch.no_grad()
    def aggregate(self, global_model, local_weights):
        """
        global_model: nn.Module
        local_weights: list of state_dict from selected clients
        """

        # 1. FedAvg weights
        avg_state = copy.deepcopy(local_weights[0])
        for k in avg_state.keys():
            if not torch.is_floating_point(avg_state[k]): 
                continue
            for i in range(1, len(local_weights)):
                avg_state[k] += local_weights[i][k]
            avg_state[k] /= float(len(local_weights))

        # 2. Compute pseudo-gradient Δ = global - avg
        grad = {}
        g_state = global_model.state_dict()
        for k in avg_state.keys():
            if not torch.is_floating_point(avg_state[k]):
                continue
            grad[k] = g_state[k] - avg_state[k]

        # 3. Adam updates
        self.t += 1
        for k in grad.keys():
            if k not in self.m:
                self.m[k] = torch.zeros_like(grad[k])
                self.v[k] = torch.zeros_like(grad[k])

            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grad[k] * grad[k])

            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            g_state[k] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        # Load updated weights
        global_model.load_state_dict(g_state)
        return copy.deepcopy(g_state)

    @torch.no_grad()
    def sample_clients(self, num_clients):
        n = max(1, int(self.client_fraction * num_clients))
        return sorted(random.sample(range(num_clients), n))



class FedAvg:
    def __init__(self, client_fraction=0.3, seed=None):
        """
        Args:
            client_fraction: fraction of clients to sample each round (0 < C ≤ 1)
            seed: optional random seed for reproducibility
        """
        self.client_fraction = client_fraction
        if seed is not None:
            random.seed(seed)
    
    @torch.no_grad()
    def sample_clients(self, num_clients):
        """Randomly select participating clients for this round."""
        num_selected = max(1, int(self.client_fraction * num_clients))
        return sorted(random.sample(range(num_clients), num_selected))

    def aggregate(self, local_weights, selected_clients):
        """Average weights only from selected clients."""
        avg_state = copy.deepcopy(local_weights[selected_clients[0]])

        for k in avg_state.keys():
            # if 'convs_per' in k: continue
            # if 'running_mean' in k: continue
            # if 'running_var' in k: continue
            # if 'num_batches_tracked' in k: continue
            if not torch.is_floating_point(avg_state[k]):
                continue
            for i in selected_clients[1:]:
                avg_state[k] += local_weights[i][k]
            avg_state[k] = avg_state[k] / float(len(selected_clients))

        return avg_state

class FedAvgM:
    def __init__(self, client_fraction=1.0, momentum=0.9):
        self.client_fraction = client_fraction
        self.momentum = momentum
        self.velocity = None  # server momentum buffer
    
    def sample_clients(self, num_clients):
        num_selected = max(1, int(self.client_fraction * num_clients))
        return np.random.choice(num_clients, num_selected, replace=False)
    
    def aggregate(self, local_weights, selected_clients, prev_global_weights=None):
        # Average selected clients
        avg_state = {}
        selected_weights = [local_weights[i] for i in selected_clients]
        for key in selected_weights[0].keys():
            avg_state[key] = torch.stack([w[key].float() for w in selected_weights]).mean(dim=0)
        
        # Apply momentum if we have previous weights
        if prev_global_weights is not None and self.velocity is not None:
            for key in avg_state.keys():
                # velocity = momentum * velocity + (new_avg - old_global)
                delta = avg_state[key] - prev_global_weights[key]
                self.velocity[key] = self.momentum * self.velocity[key] + delta
                avg_state[key] = prev_global_weights[key] + self.velocity[key]
        else:
            # Initialize velocity
            self.velocity = {k: torch.zeros_like(v) for k, v in avg_state.items()}
        
        return avg_state

# Knowledge Graph handle

def get_global_info(
        loaderData, localModels, 
        tau, rho_p, rho_d, num_antenna
    ):
    num_client = len(localModels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    send_to_server = [[] for _ in range(num_client)] 
    for batches  in zip(*loaderData):                       # sync step across APs
        sum_large_scale_each = []
        sum_channel_var_each = []
        for client_idx, (model, batch) in enumerate(zip(localModels, batches)):
            # Check batch here? something wrong?
            model.eval()
            batch = batch.to(device)
            with torch.no_grad():
                x_dict, edge_dict, edge_index = model(batch, isRawData=True)
                # DS_single, PC_single, UI_single = package_calculate(batch, x_dict, tau, rho_p, rho_d)
                ##
                num_graphs = batch.num_graphs
                num_UEs = x_dict['UE'].shape[0] // num_graphs
                num_APs = x_dict['AP'].shape[0] // num_graphs
                
                largeScale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
                power_raw = edge_dict['AP','down','UE'].reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
                    
                largeScale = torch.expm1(largeScale)
                phiMatrix = batch['UE'].x[:,:tau].reshape(num_graphs, num_UEs, -1)
                # channelVariance = variance_calculate(largeScale, phiMatrix, tau, rho_p)
                channelVariance = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]

                
                power = power_from_raw(power_raw, channelVariance, num_antenna)

                DS_single, PC_single, UI_single = component_calculate(power, channelVariance, largeScale, phiMatrix, rho_d=rho_d)
                ##
            send_to_server[client_idx].append({
                'DS': DS_single.detach(),
                "PC": PC_single.detach(),
                "UI": UI_single.detach(),
                'largeScaleRaw': largeScale.detach(),
                'channelVarianceRaw': channelVariance.detach(),
                'phiMatrix': phiMatrix.detach(),
                'AP': x_dict['AP'].detach(),
                'power_raw': power_raw.detach(),
                'UE': x_dict['UE'].detach()
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


def server_return(dataLoader, globalInformation, num_antenna=1):
    num_client = len(globalInformation)
    response_all = []

    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        aug_batch_list = []
        all_client_embeddings = [r['UE'] for r in all_response]

        # Stack all DS/PC/UI for self-exclusion and rate calculation
        # Full shape for rate calculation
        stacked_DS_full = torch.stack([r['DS'] for r in all_response], dim=0)  # [num_client, B, 1, K]
        stacked_PC_full = torch.stack([r['PC'] for r in all_response], dim=0)  # [num_client, B, 1, K', K]
        stacked_UI_full = torch.stack([r['UI'] for r in all_response], dim=0)  # [num_client, B, 1, K', K]
        rate_total = rate_from_component(
            stacked_DS_full[:,:,0,:].permute(1, 0, 2), 
            stacked_PC_full[:,:,0,:,:].permute(1, 0, 2, 3), 
            stacked_UI_full[:,:,0,:,:].permute(1, 0, 2, 3), 
            num_antenna
        ) 

        # For UE augmentation features (summed over interferers)
        stacked_DS = torch.stack([r['DS'][:,0,:] for r in all_response], dim=0)  # [num_client, B*K]
        stacked_PC = torch.stack([r['PC'][:,0,:,:].sum(dim=2) for r in all_response], dim=0)  # [num_client, B*K]
        stacked_UI = torch.stack([r['UI'][:,0,:,:].sum(dim=2) for r in all_response], dim=0)  # [num_client, B*K]


        if not all(x.shape == all_client_embeddings[0].shape for x in all_client_embeddings):
            raise RuntimeError(f"Batch {batch_idx}: Mismatch in UE counts between clients. Cannot stack.")

        global_ue_context = torch.sum(torch.stack(all_client_embeddings), dim=0)

        # Get batch info from first loader
        num_graphs = all_loader[0].num_graphs
        num_UEs = all_loader[0]['UE'].x.shape[0] // num_graphs

        for client_id, (response, batch) in enumerate(zip(all_response, all_loader)):
            other_pack = []
            keys_needed = ['DS', 'PC', 'UI']
            for j in range(num_client):
                if j != client_id:
                    full_data = all_response[j]
                    filtered_data = {k: full_data[k] for k in keys_needed}
                    other_pack.append(filtered_data)

            aug_batch = batch.clone()
            device = aug_batch['UE'].x.device

            # Self-exclusion for UE context
            new_ue_features = ((global_ue_context - all_client_embeddings[client_id]) / (num_client - 1)).to(device)

            # Self-exclusion mask
            mask = torch.arange(num_client) != client_id

            # Self-exclusion for DS/PC/UI: mean of others only
            other_DS = stacked_DS[mask].mean(dim=0).reshape(-1, 1).to(device)
            other_PC = stacked_PC[mask].mean(dim=0).reshape(-1, 1).to(device)
            other_UI = stacked_UI[mask].mean(dim=0).reshape(-1, 1).to(device)

            # Compute rate WITHOUT this AP (from other APs only)
            other_DS_full = stacked_DS_full[mask]  # [num_client-1, B, 1, K]
            other_PC_full = stacked_PC_full[mask]  # [num_client-1, B, 1, K', K]
            other_UI_full = stacked_UI_full[mask]  # [num_client-1, B, 1, K', K]

            # Reshape: [B, num_client-1, ...] then concat along AP dim
            other_DS_cat = other_DS_full.permute(1, 0, 2, 3).reshape(num_graphs, num_client - 1, num_UEs)
            other_PC_cat = other_PC_full.permute(1, 0, 2, 3, 4).reshape(num_graphs, num_client - 1, num_UEs, num_UEs)
            other_UI_cat = other_UI_full.permute(1, 0, 2, 3, 4).reshape(num_graphs, num_client - 1, num_UEs, num_UEs)

            rate_without_me = rate_from_component(other_DS_cat, other_PC_cat, other_UI_cat, num_antenna)  # [B, K]
            rate_without_me_flat = (rate_without_me-rate_total/rate_total).reshape(-1, 1).to(device)  # [B*K, 1]

            aug_batch['UE'].x = torch.cat(
                [
                    aug_batch['UE'].x, new_ue_features,
                    other_DS, other_PC, other_UI,
                    rate_without_me_flat
                ],
                dim=-1
            )

            client_data = {
                'loader': aug_batch,
                'rate_pack': other_pack
            }
            aug_batch_list.append(client_data)
        response_all.append(aug_batch_list)
    return response_all


########################################################################
# Server Return with Bottleneck UE Identification
########################################################################
def server_return_with_bottleneck(dataLoader, globalInformation, num_antenna=1):
    """
    Augment client data with bottleneck UE information.
    Server identifies the UE with lowest rate globally and tells all clients to prioritize it.

    Args:
        dataLoader: list of dataloaders for each client
        globalInformation: output from get_global_info()
        num_antenna: number of antennas

    Returns:
        response_all: list of augmented batches with bottleneck_ue info
    """
    num_client = len(globalInformation)
    response_all = []

    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        aug_batch_list = []
        all_client_embeddings = [r['UE'] for r in all_response]

        # === Compute GLOBAL rate to find bottleneck UE ===
        # Gather DS/PC/UI from all clients
        full_DS = torch.cat([r['DS'] for r in all_response], dim=1)  # [B, num_AP, K]
        full_PC = torch.cat([r['PC'] for r in all_response], dim=1)  # [B, num_AP, K', K]
        full_UI = torch.cat([r['UI'] for r in all_response], dim=1)  # [B, num_AP, K', K]

        # Compute global rate for each UE
        global_rate = rate_from_component(full_DS, full_PC, full_UI, num_antenna)  # [B, K]

        # Find bottleneck UE (lowest rate) for each sample
        bottleneck_ue_idx = global_rate.argmin(dim=1)  # [B]
        bottleneck_ue_rate = global_rate.min(dim=1).values  # [B]

        # === Compute aggregated features for UE augmentation ===
        all_DS_agg = torch.stack([r['DS'][:,0,:] for r in all_response], dim=1).mean(dim=1).reshape(-1, 1)
        all_PC_agg = torch.stack([r['PC'][:,0,:,:].sum(dim=2) for r in all_response], dim=1).mean(dim=1).reshape(-1, 1)
        all_UI_agg = torch.stack([r['UI'][:,0,:,:].sum(dim=2) for r in all_response], dim=1).mean(dim=1).reshape(-1, 1)

        if not all(x.shape == all_client_embeddings[0].shape for x in all_client_embeddings):
            raise RuntimeError(f"Batch {batch_idx}: Mismatch in UE counts between clients. Cannot stack.")

        global_ue_context = torch.mean(torch.stack(all_client_embeddings), dim=0)

        for client_id, (response, batch) in enumerate(zip(all_response, all_loader)):
            other_pack = []
            keys_needed = ['DS', 'PC', 'UI']
            for j in range(num_client):
                if j != client_id:
                    full_data = all_response[j]
                    filtered_data = {k: full_data[k] for k in keys_needed}
                    other_pack.append(filtered_data)

            aug_batch = batch.clone()
            device = aug_batch['UE'].x.device
            new_ue_features = global_ue_context.to(device)
            aug_batch['UE'].x = torch.cat(
                [
                    aug_batch['UE'].x, new_ue_features,
                    all_DS_agg.to(device), all_PC_agg.to(device), all_UI_agg.to(device)
                ],
                dim=-1
            )

            client_data = {
                'loader': aug_batch,
                'rate_pack': other_pack,
                'bottleneck_ue': bottleneck_ue_idx.to(device),  # [B] - index of bottleneck UE
                'bottleneck_rate': bottleneck_ue_rate.to(device),  # [B] - rate of bottleneck UE
                'global_rate': global_rate.to(device),  # [B, K] - full rate matrix (optional)
            }
            aug_batch_list.append(client_data)
        response_all.append(aug_batch_list)
    return response_all

@torch.no_grad()
def server_return_with_gap(dataLoader, globalInformation, num_global_ap=None, num_antenna=1):
    """
    Augment each client's graph with Virtual Global AP nodes.
    num_global_ap = None means use ALL other APs (num_clients - 1)
    Each GAP is a direct copy of another AP (no aggregation).
    """
    num_client = len(globalInformation)
    response_all = []

    # Default: use all other APs as GAPs
    if num_global_ap is None:
        num_global_ap = num_client - 1

    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        aug_batch_list = []

        # Gather all APs' info
        all_ap_emb = torch.stack([r['AP'] for r in all_response], dim=1)  # [B, num_AP, dim]
        all_ue_emb = torch.stack([r['UE'] for r in all_response], dim=0)  # [num_AP, B*K, dim]
        all_DS = torch.stack([r['DS'][:,0,:] for r in all_response], dim=1)  # [B, num_AP, K]
        all_PC = torch.stack([r['PC'][:,0,:,:] for r in all_response], dim=1)  # [B, num_AP, K', K]
        all_UI = torch.stack([r['UI'][:,0,:,:] for r in all_response], dim=1)  # [B, num_AP, K', K]
        all_power = torch.stack([r['power_raw'][:,0,:] for r in all_response], dim=1)  # [B, num_AP, K]
        all_large_scale = torch.stack([r['largeScaleRaw'][:,0,:] for r in all_response], dim=1)  # [B, num_AP, K]
        all_channel_var = torch.stack([r['channelVarianceRaw'][:,0,:] for r in all_response], dim=1)  # [B, num_AP, K]

        # === Compute rate WITH all APs ===
        full_DS = torch.cat([r['DS'] for r in all_response], dim=1)  # [B, num_AP, K]
        full_PC = torch.cat([r['PC'] for r in all_response], dim=1)  # [B, num_AP, K', K]
        full_UI = torch.cat([r['UI'] for r in all_response], dim=1)  # [B, num_AP, K', K]
        rate_with_all = rate_from_component(full_DS, full_PC, full_UI, num_antenna)  # [B, K]

        for client_id, (response, batch) in enumerate(zip(all_response, all_loader)):
            aug_batch = batch.clone()
            device = aug_batch['UE'].x.device
            num_graphs = aug_batch.num_graphs
            num_ues = aug_batch['UE'].x.shape[0] // num_graphs
            num_aps = aug_batch['AP'].x.shape[0] // num_graphs  # Should be 1

            # === Get indices of other APs (exclude self) ===
            other_indices = [j for j in range(num_client) if j != client_id]
            n_gaps = len(other_indices)  # = num_client - 1 = 29

            # === 1. Create GAP node features (one per other AP) ===
            # Each GAP gets that AP's embedding directly (no aggregation)
            gap_feats = all_ap_emb[:, other_indices, :]  # [B, n_gaps, dim]
            gap_feats = gap_feats.reshape(-1, gap_feats.shape[-1])  # [B*n_gaps, dim]

            # === 2. Create GAP -> UE edge features (one set per GAP) ===
            # Each GAP has edges to all UEs with that AP's specific info
            gap_edge_feats_list = []
            gap_down_ue_edges = []
            ue_up_gap_edges = []

            for gap_local_idx, other_ap_idx in enumerate(other_indices):
                # Edge features for this GAP: [B, K] each
                this_DS = all_DS[:, other_ap_idx, :]  # [B, K]
                this_PC = all_PC[:, other_ap_idx, :, :].sum(dim=1)  # [B, K] - sum over interferers
                this_UI = all_UI[:, other_ap_idx, :, :].sum(dim=1)  # [B, K]
                this_power = all_power[:, other_ap_idx, :]  # [B, K]
                this_large_scale = all_large_scale[:, other_ap_idx, :]  # [B, K]
                this_channel_var = all_channel_var[:, other_ap_idx, :]  # [B, K]

                # Stack edge features: [B, K, 6] -> [B*K, 6]
                this_edge_feat = torch.stack([
                    this_large_scale,
                    this_channel_var,
                    this_DS,
                    this_PC,
                    this_UI,
                    this_power,
                ], dim=-1).reshape(-1, 6)
                gap_edge_feats_list.append(this_edge_feat)

                # Edge indices for this GAP
                for g in range(num_graphs):
                    gap_offset = g * n_gaps + gap_local_idx  # GAP index in flattened batch
                    ue_offset = g * num_ues
                    for ue in range(num_ues):
                        gap_down_ue_edges.append([gap_offset, ue_offset + ue])
                        ue_up_gap_edges.append([ue_offset + ue, gap_offset])

            # Stack all edge features: [n_gaps * B * K, 6]
            gap_edge_feat = torch.cat(gap_edge_feats_list, dim=0)

            gap_down_ue_idx = torch.tensor(gap_down_ue_edges, dtype=torch.long, device=device).t().contiguous()
            ue_up_gap_idx = torch.tensor(ue_up_gap_edges, dtype=torch.long, device=device).t().contiguous()

            # === 3. Add GAP nodes and edges to graph ===
            aug_batch['GAP'].x = gap_feats.to(device)

            aug_batch['GAP', 'down', 'UE'].edge_index = gap_down_ue_idx
            aug_batch['GAP', 'down', 'UE'].edge_attr = gap_edge_feat.to(device)

            aug_batch['UE', 'up', 'GAP'].edge_index = ue_up_gap_idx
            aug_batch['UE', 'up', 'GAP'].edge_attr = gap_edge_feat.to(device)

            # === 4. Compute rate difference (marginal contribution) ===
            mask = torch.arange(num_client, device=device) != client_id
            rate_without_me = rate_from_component(
                all_DS[:, mask, :],
                all_PC[:, mask, :, :],
                all_UI[:, mask, :, :],
                num_antenna
            )
            rate_diff = (rate_with_all - rate_without_me).reshape(-1, 1)  # [B*K, 1]

            # === 5. Augment UE features ===
            global_ue_context = all_ue_emb[mask].mean(dim=0)  # [B*K, dim]
            aug_batch['UE'].x = torch.cat([
                aug_batch['UE'].x,
                global_ue_context.to(device),
                rate_diff.to(device)
            ], dim=-1)
            
            # === Full Response ===
            other_pack = []
            keys_needed = ['DS', 'PC', 'UI']
            for j in range(num_client):
                if j != client_id:
                    filtered_data = {k: all_response[j][k] for k in keys_needed}
                    other_pack.append(filtered_data)
            
            client_data = {
                'loader': aug_batch,
                'rate_pack': other_pack
            }
            aug_batch_list.append(client_data)
        
        response_all.append(aug_batch_list)
    
    return response_all


    
# @torch.no_grad()
# def kg_augment_add_AP(
#         dataLoader, globalInformation, numGlobalAP, tau, num_antenna, rho_d,
#         scheme_global: str = 'mean',
#         ds_feature_mode: str = 'ratio'
#     ):
#     num_global_AP = numGlobalAP
#     num_clients = len(globalInformation)
#     augmented_batches = [[] for _ in range(num_clients)]

#     for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):    
#         # all_ds = torch.cat([all_response[c]['DS'] for c in range(num_clients)], dim=1)
#         # all_pc = torch.cat([all_response[c]['PC'] for c in range(num_clients)], dim=1)
#         # all_ui = torch.cat([all_response[c]['UI'] for c in range(num_clients)], dim=1)
#         all_large_scale = torch.cat([all_response[c]['largeScaleRaw'] for c in range(num_clients)], dim=1)
#         all_channel_var = torch.cat([all_response[c]['channelVarianceRaw'] for c in range(num_clients)], dim=1)
#         all_AP = torch.cat([all_response[c]['AP'][:,None,:] for c in range(num_clients)], dim=1)
#         all_power = torch.cat([all_response[c]['power_raw'] for c in range(num_clients)], dim=1)
        
#         equal_power = torch.ones_like(all_power, dtype=all_power.dtype, device=all_power.device)
#         all_phi_matrix = all_response[0]['phiMatrix']
#         all_norm_ds, all_norm_pc, all_norm_ui = component_calculate(
#             equal_power, all_channel_var, all_large_scale, all_phi_matrix, rho_d
#         )

        
        
#         for client_id, (response, batch) in enumerate(zip(all_response, all_loader)):
#             each_client_data = []
#             num_graphs = batch.num_graphs
#             num_UEs = batch['UE'].x.shape[0] // num_graphs
#             num_APs = batch['AP'].x.shape[0] // num_graphs
#             # apFeatOrg = batch['AP'].x[:,:,None]
            
#             # label = batch.y
#             largeScale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
#             # largeScale = torch.expm1(largeScale)
#             channelVariance = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]
#             phiMatrix = batch['UE'].x[:,:tau].reshape(num_graphs, num_UEs, -1)
#             powerClient = batch['AP','down','UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
            
#             ## client-off rate
#             power_mat = torch.cat(
#                 [
#                     all_power[:,:client_id,...],
#                     torch.zeros_like(powerClient),
#                     all_power[:,client_id+1:,...],
#                 ],
#                 dim=1
#             )
#             ds_off, pc_off, ui_off = component_calculate(power_mat, all_channel_var, all_large_scale, all_phi_matrix, rho_d)
#             rate_full = rate_from_component(ds_off, pc_off, ui_off, num_antenna).detach()
#             min_rate, _ = torch.min(rate_full, dim=1, keepdims=True)
#             min_rate = min_rate.detach()
#             rate_gap = (rate_full - min_rate) / (min_rate.abs() + 1e-8)
#             ##
            
            
            
#             global_large_scale = new_ap_information(
#                 all_large_scale, client_id, num_global_AP,
#                 scheme=scheme_global
#             )
#             global_channel_var = new_ap_information(
#                 all_channel_var, client_id, num_global_AP,
#                 scheme=scheme_global
#             )
#             global_power = new_ap_information(
#                 all_power, client_id, num_global_AP,
#                 scheme=scheme_global
#             )
#             global_ap = new_ap_information(
#                 all_AP, client_id, num_global_AP,
#                 scheme=scheme_global
#             )
            
#             ## Combined information
#             # global_ds = new_ap_information(
#             #     all_norm_ds, client_id, num_global_AP,
#             #     scheme=scheme_global
#             # )
#             # global_pc = new_ap_information(
#             #     all_norm_pc, client_id, num_global_AP,
#             #     scheme=scheme_global
#             # )
#             # global_ui = new_ap_information(
#             #     all_norm_ui, client_id, num_global_AP,
#             #     scheme=scheme_global
#             # )
#             # global_pc = global_pc.sum(dim=3)
#             # global_ui = global_ui.sum(dim=3)
            
#             ##
            
#             global_ds = new_ap_information(
#                 all_norm_ds, client_id, num_global_AP, scheme=scheme_global
#             )                             # [B, K, U]
#             global_pc_raw = new_ap_information(
#                 all_norm_pc, client_id, num_global_AP, scheme=scheme_global
#             )                             # [B, K, U, *]
#             global_ui_raw = new_ap_information(
#                 all_norm_ui, client_id, num_global_AP, scheme=scheme_global
#             )                             # [B, K, U, *]

#             # Sum over interfering-UE dimension for PC/UI (same as before)
#             global_pc = global_pc_raw.sum(dim=3)   # [B, K, U]
#             global_ui = global_ui_raw.sum(dim=3)   # [B, K, U]
#             ########################################
            
#             global_large_scale = torch.log1p(global_large_scale)
#             global_channel_var = torch.log1p(global_channel_var)
#             # global_ds = torch.log1p(global_ds)
#             # global_pc = torch.log1p(global_pc)
#             # global_ui = torch.log1p(global_ui)
            
#             global_ds = global_ds / global_ds.sum(dim=1, keepdim=True)
#             global_pc = global_pc / global_pc.sum(dim=1, keepdim=True)
#             global_ui = global_ui / global_ui.sum(dim=1, keepdim=True)
            
#             # ap_feat = torch.cat([apFeatOrg, new_ap], dim=1)
#             # power_matrix = torch.cat([powerClient, new_power], dim=1)
#             # large_scale_matrix = torch.cat([largeScale, new_large_scale], dim=1)
#             # large_scale_matrix = torch.log1p(large_scale_matrix)
#             # channel_variance_matrix = torch.cat([channelVariance, new_channel_var], dim=1)

#             for each_sample in range(num_graphs):
#                 global_information = (
#                     global_ap[each_sample].cpu().numpy(),
#                     global_large_scale[each_sample].cpu().numpy(),
#                     global_channel_var[each_sample].cpu().numpy(),
#                     global_power[each_sample].cpu().numpy(),
#                     global_ds[each_sample].cpu().numpy(),
#                     global_pc[each_sample].cpu().numpy(),
#                     global_ui[each_sample].cpu().numpy(),
#                     rate_gap[each_sample,:,None].cpu().numpy(),
#                 )
#                 tmp_ue_ue_information = (
#                     global_pc_raw[each_sample].cpu().numpy(),
#                     global_ui_raw[each_sample].cpu().numpy(),
#                 )
#                 data = full_het_graph(
#                     largeScale[each_sample].cpu().numpy(),
#                     channelVariance[each_sample].cpu().numpy(),
#                     None,
#                     phiMatrix[each_sample].cpu().numpy(),
#                     global_ap_information=global_information,
#                     tmp_ue_ue_information=tmp_ue_ue_information
#                 )
#                 each_client_data.append(data)
#             each_client_data = Batch.from_data_list(each_client_data) 
#             augmented_batches[client_id].append(each_client_data)
#     return augmented_batches

# def kg_augment_add_AP_old(dataLoader, globalInformation, numGlobalAP, tau):
#     num_global_AP = numGlobalAP
#     num_clients = len(globalInformation)
#     augmented_batches = [[] for _ in range(num_clients)]

#     for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):    
#         all_large_scale = torch.cat(
#             [all_response[c]['largeScaleRaw'] for c in range(num_clients)], 
#             dim=1
#         )
        
#         all_channel_var = torch.cat(
#             [all_response[c]['channelVarianceRaw'] for c in range(num_clients)], 
#             dim=1
#         )
#         all_AP = torch.cat(
#             [all_response[c]['AP'] for c in range(num_clients)], 
#             dim=1
#         )
        
#         all_power = torch.cat(
#             [all_response[c]['power_raw'] for c in range(num_clients)], 
#             dim=1
#         )
#         for client_id, (response, batch) in enumerate(zip(all_response, all_loader)):
#             each_client_data = []
#             num_graphs = batch.num_graphs
#             num_UEs = batch['UE'].x.shape[0] // num_graphs
#             num_APs = batch['AP'].x.shape[0] // num_graphs
#             apFeatOrg = batch['AP'].x[:,:,None]
            
#             # label = batch.y
            
#             largeScale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
#             # largeScale = torch.expm1(largeScale)
#             channelVariance = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,1]
#             phiMatrix = batch['UE'].x[:,:tau].reshape(num_graphs, num_UEs, -1)
#             # powerClient = batch['AP','down','UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
            
#             global_large_scale = new_ap_information(all_large_scale, client_id, num_global_AP)
#             global_channel_var = new_ap_information(all_channel_var, client_id, num_global_AP)
#             global_power = new_ap_information(all_power, client_id, num_global_AP)
#             global_ap = new_ap_information(all_AP[:,:,None], client_id, num_global_AP)
#             global_large_scale = torch.log1p(global_large_scale)
#             # ap_feat = torch.cat([apFeatOrg, new_ap], dim=1)
#             # power_matrix = torch.cat([powerClient, new_power], dim=1)
#             # large_scale_matrix = torch.cat([largeScale, new_large_scale], dim=1)
#             # large_scale_matrix = torch.log1p(large_scale_matrix)
#             # channel_variance_matrix = torch.cat([channelVariance, new_channel_var], dim=1)

            
#             for each_sample in range(num_graphs):
#                 global_information = (
#                     global_ap[each_sample].cpu().numpy(), 
#                     global_large_scale[each_sample].cpu().numpy(), 
#                     global_channel_var[each_sample].cpu().numpy(), 
#                     global_power[each_sample].cpu().numpy()
#                 )
#                 data = full_het_graph(
#                     largeScale[each_sample].cpu().numpy(), 
#                     channelVariance[each_sample].cpu().numpy(), 
#                     # label[each_sample].cpu().numpy(), 
#                     None,
#                     phiMatrix[each_sample].cpu().numpy(),
#                     # ap_feat=ap_feat[each_sample].cpu().numpy(),
#                     global_ap_information=global_information,
#                 )
#                 each_client_data.append(data)
#             each_client_data = Batch.from_data_list(each_client_data) 
#             augmented_batches[client_id].append(each_client_data)
#     return augmented_batches

def new_ap_information(aggregatedInfor, currentId, numGlobalAp, scheme='mean'):
    n_others = aggregatedInfor.shape[1]
    idx = torch.arange(aggregatedInfor.size(1)) != currentId
    aggregatedInfor = aggregatedInfor[:, idx]
    # aggregatedInfor = torch.cat([aggregatedInfor[:,:currentId], aggregatedInfor[:,currentId+1:]], dim=1)
    split_ratio = 0.6
    if scheme == 'sum':
        aggregatedInfor = aggregatedInfor.sum(dim=1, keepdim=True)
        aggregatedInfor = aggregatedInfor.repeat_interleave(numGlobalAp, dim=1)
    elif scheme == 'mean': # better than sum
        aggregatedInfor = aggregatedInfor.mean(dim=1, keepdim=True)
        aggregatedInfor = aggregatedInfor.repeat_interleave(numGlobalAp, dim=1)
    elif scheme == 'top': # not good
        # aggregatedInfor, _ = torch.topk(aggregatedInfor, k=numGlobalAp, dim=1)
        # aggregatedInfor = aggregatedInfor.repeat_interleave(numGlobalAp, dim=1)
        scores = aggregatedInfor.reshape(aggregatedInfor.size(0), aggregatedInfor.size(1), -1).norm(dim=-1)
        # scores: [B, M-1]

        top_scores, top_idx = torch.topk(scores, k=numGlobalAp, dim=1)  # top_idx: [B, K]

        # Gather along AP dimension
        # Expand indices to match extra feature dims
        expand_shape = (-1, -1) + (1,) * (aggregatedInfor.ndim - 2)
        gather_idx = top_idx.view(*top_idx.shape, *([1] * (aggregatedInfor.ndim - 2))).expand(
            *top_idx.shape, *aggregatedInfor.shape[2:]
        )
        aggregatedInfor = aggregatedInfor.gather(1, gather_idx)
    elif scheme == 'strong-weak':
        aggregatedInfor, idx = torch.sort(aggregatedInfor, dim=1, descending=True)
        n_strong = max(1, int(n_others * split_ratio))
        strong = aggregatedInfor[:, :n_strong, :]          
        weak   = aggregatedInfor[:, n_strong:, :] 
        strong_mean = strong.mean(dim=1, keepdim=True)   
        weak_mean = weak.mean(dim=1, keepdim=True)   
        if numGlobalAp == 1:
            aggregatedInfor = strong_mean
        elif numGlobalAp == 2:
            aggregatedInfor = torch.cat([strong_mean, weak_mean], dim=1)
        else:
            n_sel = numGlobalAp//2
            n_sel_weak = numGlobalAp - n_sel
            strong_sel = strong[:, :n_sel, :]          
            weak_sel   = weak[:, :n_sel_weak, :] 
            aggregatedInfor = torch.cat([strong_sel, weak_sel], dim=1)
            
    elif scheme == 'all':
        assert n_others == numGlobalAp + 1
        pass
    else:
        raise ValueError(f'{scheme} not supported!')
    return aggregatedInfor

# # Global AP nodes
# x_ap_global = torch.ones((num_global_AP * num_graphs,1), dtype=torch.float32).to(device)
# x_ap_org = batch['AP'].x
# num_old_AP = x_ap_org.shape[0]
# batch['AP'].x = torch.cat(
#     [x_ap_org, x_ap_global],
#     dim=0
# ).contiguous()

# s = torch.arange(num_graphs).view(num_graphs, 1, 1).to(device)
# a = torch.arange(num_global_AP).view(1, num_global_AP, 1).to(device)
# u = torch.arange(num_UEs).view(1, 1, num_UEs).to(device)

# src = num_old_AP + num_graphs * a + s
# dst = num_UEs * s + u

# src = src.expand(-1, -1, num_UEs) 
# dst = dst.expand(-1, num_global_AP, -1) 
# edge_index = torch.stack([src, dst], dim=3)


# ## Edge from Global APs to UE
# edge_index_down_org = batch['AP', 'down', 'UE'].edge_index # [2, 192]
# edge_attr_down_org = batch['AP', 'down', 'UE'].edge_attr # [192, 2]
# down_large_scale = new_large_scale.reshape(-1, 1)
# down_channel_var = new_channel_var.reshape(-1, 1)
# down_attr = torch.cat([down_large_scale, down_channel_var], dim=1)
# new_edge_index_down = edge_index.reshape(-1, 2).t()

# batch['AP', 'down', 'UE'].edge_index = torch.cat([edge_index_down_org, new_edge_index_down], dim=1).contiguous()
# batch['AP', 'down', 'UE'].edge_attr = torch.cat([edge_attr_down_org, down_attr], dim=0).contiguous()

# ## Edge from UE to global APs
# edge_index_up_org = batch['UE', 'up', 'AP'].edge_index # [2, 192]
# edge_attr_up_org = batch['UE', 'up', 'AP'].edge_attr # [192, 2]
# up_large_scale = new_large_scale.permute(0, 2, 1).reshape(-1, 1)
# up_channel_var = new_channel_var.permute(0, 2, 1).reshape(-1, 1)
# up_attr = torch.cat([up_large_scale, up_channel_var], dim=1)
# new_edge_index_up = edge_index.permute(0, 2, 1, 3).reshape(-1, 2).t()

# batch['UE', 'up', 'AP'].edge_index = torch.cat([edge_index_up_org, new_edge_index_up], dim=1).contiguous()
# batch['UE', 'up', 'AP'].edge_attr = torch.cat([edge_attr_up_org, up_attr], dim=0).contiguous()
# augmented_batches[client_id].append(batch)

@torch.no_grad()
def kg_augment_simple(dataLoader, globalInformation, tau):
    """
    Simplified knowledge graph augmentation.
    Augments UE node features with global information from other APs.

    Args:
        dataLoader: List of DataLoader (one per client/AP)
        globalInformation: List of list of dicts containing {DS, PC, UI, ...} from all APs
        tau: Pilot dimension

    Returns:
        augmented_batches: List of lists of augmented batches (one list per client)
    """
    num_clients = len(globalInformation)
    augmented_batches = [[] for _ in range(num_clients)]

    # Iterate over synchronized batches across all clients
    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        # Aggregate global information from ALL APs
        all_DS = torch.cat([all_response[c]['DS'] for c in range(num_clients)], dim=1)  # [B, M, K]
        all_PC = torch.cat([all_response[c]['PC'] for c in range(num_clients)], dim=1)  # [B, M, K] or [B, M, K, K]
        all_UI = torch.cat([all_response[c]['UI'] for c in range(num_clients)], dim=1)  # [B, M, K] or [B, M, K, K]

        # For each client, compute global info excluding itself
        for client_id, (response, batch) in enumerate(zip(all_response, all_loader)):
            num_graphs = batch.num_graphs
            num_UEs = batch['UE'].x.shape[0] // num_graphs

            # Extract original UE features (pilots)
            ue_pilots = batch['UE'].x[:, :tau]  # [B*K, tau]

            # Compute residual (global - local) for this client
            # DS: Direct Signal strength
            ds_global = all_DS.sum(dim=1) - response['DS'].squeeze(1)  # [B, K]

            # PC: Pilot Contamination (sum over interfering UEs if needed)
            if len(all_PC.shape) == 4:  # [B, M, K, K]
                pc_global = all_PC.sum(dim=1).sum(dim=2) - response['PC'].squeeze(1).sum(dim=2)  # [B, K]
            else:  # [B, M, K]
                pc_global = all_PC.sum(dim=1) - response['PC'].squeeze(1)  # [B, K]

            # UI: User Interference (sum over interfering UEs if needed)
            if len(all_UI.shape) == 4:  # [B, M, K, K]
                ui_global = all_UI.sum(dim=1).sum(dim=2) - response['UI'].squeeze(1).sum(dim=2)  # [B, K]
            else:  # [B, M, K]
                ui_global = all_UI.sum(dim=1) - response['UI'].squeeze(1)  # [B, K]

            # Normalize to prevent scale issues
            ds_global = torch.log1p(ds_global)  # [B, K]
            pc_global = torch.log1p(pc_global)  # [B, K]
            ui_global = torch.log1p(ui_global)  # [B, K]

            # Flatten to match UE node dimension [B*K, 1]
            ds_feat = ds_global.reshape(-1, 1)
            pc_feat = pc_global.reshape(-1, 1)
            ui_feat = ui_global.reshape(-1, 1)

            # Augment UE features: [pilots | DS_global | PC_global | UI_global]
            batch['UE'].x = torch.cat([
                ue_pilots,   # [B*K, tau]
                ds_feat,     # [B*K, 1]
                pc_feat,     # [B*K, 1]
                ui_feat      # [B*K, 1]
            ], dim=1).contiguous()  # [B*K, tau+3]

            augmented_batches[client_id].append(batch)

    return augmented_batches


def kg_augmentation(train_loader, send_to_server, tau):
    """Legacy function - redirects to simplified version"""
    return kg_augment_simple(train_loader, send_to_server, tau)


@torch.no_grad()
def load_state_dict_skipping(model: torch.nn.Module,
                             state_dict: dict,
                             exclude_contains=(),
                             exclude_regex: str = None):
    """
    Copy weights from state_dict into model, but skip params/buffers whose names
    contain any substring in exclude_contains, or match exclude_regex.
    """
    own = model.state_dict()
    for k, v in state_dict.items():
        if any(p in k for p in (exclude_contains or ())):
            continue
        if exclude_regex and re.search(exclude_regex, k):
            continue
        if k in own and own[k].shape == v.shape and own[k].dtype == v.dtype:
            own[k].copy_(v)
            
            

def personal_state(sd):
    keep = {}
    for k, v in sd.items():
        if ('convs_per' in k or
            'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k):
            keep[k] = v.cpu()
    return keep

def save_models(args, global_model, local_models, file_name):
    ckpt = {
        "args": vars(args),
        "global": {k: v.cpu() for k, v in global_model.state_dict().items()},
        "locals": {i: personal_state(m.state_dict()) for i, m in enumerate(local_models)},
    }
    torch.save(ckpt, file_name)
    
    
def load_models(file_name, global_model, local_models):
    state = torch.load(file_name, map_location="cpu")
    # 1) load global once
    global_model.load_state_dict(state["global"])
    # 2) broadcast global to clients, but keep personal + BN local (FedBN)
    for i, m in enumerate(local_models):
        load_state_dict_skipping(
            m, global_model.state_dict(),
            exclude_contains=("convs_per","running_mean","running_var","num_batches_tracked")
        )
        # 3) restore that client's personalization if you saved it
        if "locals" in state and i in state["locals"]:
            m.load_state_dict(state["locals"][i], strict=False)