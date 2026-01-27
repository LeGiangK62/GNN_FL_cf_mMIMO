import re
import torch
import copy 
import random
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
# from Utils.data_gen import full_het_graph
from Utils.comm import (
    variance_calculate, rate_calculation, 
    component_calculate, rate_from_component,
    power_from_raw
)


########################## Training and Testing function ##########################


###################################################################################
def loss_function(graphData, nodeFeatDict, edgeDict, clientResponse, tau, rho_p, rho_d, num_antenna, isTrain=True):
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
    
    if not isTrain:
        return DS_k, PC_k, UI_k
    
    all_DS = [DS_k] + [r['DS'] for r in clientResponse]
    all_PC = [PC_k] + [r['PC'] for r in clientResponse]
    all_UI = [UI_k] + [r['UI'] for r in clientResponse]

    all_DS = torch.cat(all_DS, dim=1)
    all_PC = torch.cat(all_PC, dim=1)
    all_UI = torch.cat(all_UI, dim=1)

    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)  # [B, K]
    min_rate_detach, _ = torch.min(rate.detach(), dim=1)
    
    # min_rate, _ = torch.min(rate, dim=1)
    temperature = 2.1
    min_rate = -torch.logsumexp(-rate / temperature, dim=1) * temperature
    loss = -min_rate.mean()

    return loss, min_rate_detach


def fl_train(
        dataLoader, responseInfo, model, optimizer,
        tau, rho_p, rho_d, num_antenna
):
    """
    Train a local FL model for one epoch.

    Args:
        dataLoader: List of graph batches for this client
        responseInfo: List of rate_pack (DS/PC/UI from other clients)
        model: Local model
        optimizer: Local optimizer
        tau, rho_p, rho_d, num_antenna: Communication parameters
    Returns:
        avg_loss, avg_min_rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    total_loss = 0.0
    total_min_rate = 0.0
    total_graphs = 0

    for batch, response in zip(dataLoader, responseInfo):
        batch = batch.to(device)
        num_graph = batch.num_graphs
        optimizer.zero_grad()
        x_dict, attr_dict, _ = model(batch, isRawData=False)
        loss, min_rate = loss_function(
        # loss, min_rate = loss_function_guided(
            batch, x_dict, attr_dict, response,
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
            # epochRatio=epochRatio,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * num_graph
        total_min_rate += min_rate.mean().item() * num_graph
        total_graphs += num_graph


    return total_loss/total_graphs, total_min_rate/total_graphs


@torch.no_grad()
def fl_eval(
        dataLoader, local_models,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_clients = len(local_models)

    send_to_server = get_global_info(
        dataLoader, local_models,
        tau=tau, rho_p=rho_p, rho_d=rho_d,
        num_antenna=num_antenna
    )

    # response_from_server = server_return(dataLoader, send_to_server, num_antenna=num_antenna)
    response_from_server = server_return_GAP(dataLoader, send_to_server, num_antenna=num_antenna)

    all_DS = [[] for i in range(num_clients)]
    all_PC = [[] for i in range(num_clients)]
    all_UI = [[] for i in range(num_clients)]

    for client_idx, (model, client_data_tuple) in enumerate(zip(local_models, zip(*response_from_server))):
        batches = [item['loader'] for item in client_data_tuple]
        batch_rate = [item['rate_pack'] for item in client_data_tuple]

        for batch, response in zip(batches, batch_rate):
            batch = batch.to(device)
            num_graph = batch.num_graphs
            x_dict, attr_dict, _ = model(batch, isRawData=False)
            DS_k, PC_k, UI_k = loss_function(
                batch, x_dict, attr_dict, response,
                tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
                isTrain=False
            )

            all_DS[client_idx].append(DS_k.detach())
            all_PC[client_idx].append(PC_k.detach())
            all_UI[client_idx].append(UI_k.detach())

    all_DS = [torch.cat(client_batches, dim=0) for client_batches in all_DS]
    all_PC = [torch.cat(client_batches, dim=0) for client_batches in all_PC]
    all_UI = [torch.cat(client_batches, dim=0) for client_batches in all_UI]

    all_DS = torch.cat(all_DS, dim=1)
    all_PC = torch.cat(all_PC, dim=1)
    all_UI = torch.cat(all_UI, dim=1)
    
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)

    min_rate, _ = torch.min(rate, dim=1)

    return min_rate


############################## Communication function #############################
# get_global_info: Clients send to server 
# server_return: Server returns to clients
###################################################################################

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
                edge_attr_up_full = edge_dict['AP','down','UE']
                edge_attr_down_full = edge_dict['AP','down','UE']

                largeScale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
                power_raw = edge_attr_up_full.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
                    
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
                'edge_up': edge_attr_up_full.detach(),
                'edge_down': edge_attr_down_full.detach(),
                'phiMatrix': phiMatrix.detach(),
                'AP': x_dict['AP'].detach(),
                'power_raw': power_raw.detach(),
                'UE': x_dict['UE'].detach()
            })
            
    return send_to_server

def server_return(dataLoader, globalInformation, num_antenna=1):
    num_client = len(globalInformation)
    response_all = []

    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        aug_batch_list = []
        all_client_embeddings = [r['UE'] for r in all_response]

        if not all(x.shape == all_client_embeddings[0].shape for x in all_client_embeddings):
            raise RuntimeError(f"Batch {batch_idx}: Mismatch in UE counts between clients. Cannot stack.")

        global_ue_context = torch.sum(torch.stack(all_client_embeddings), dim=0)
        for client_id, (_, batch) in enumerate(zip(all_response, all_loader)):

            # Rate pack from other APs (DC, PC, UI)
            other_pack = []
            keys_needed = ['DS', 'PC', 'UI']
            for j in range(num_client):
                if j != client_id:
                    full_data = all_response[j]
                    filtered_data = {k: full_data[k] for k in keys_needed}
                    other_pack.append(filtered_data)

            aug_batch = batch.clone()
            device = aug_batch['UE'].x.device

            # Global UE context (from other APs)
            new_ue_features = ((global_ue_context - all_client_embeddings[client_id]) / (num_client - 1)).to(device)

            aug_batch['UE'].x = torch.cat(
                [
                    aug_batch['UE'].x, # init dim
                    new_ue_features,   # out_channel
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


def server_return_GAP(dataLoader, globalInformation, num_antenna=1):
    num_client = len(globalInformation)
    response_all = []

    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        aug_batch_list = []
        all_client_embeddings = [r['UE'] for r in all_response]
        all_AP_embeddings = [r['AP'] for r in all_response]
        all_edge_embeddings = [r['edge_down'] for r in all_response]

        all_DS = [r['DS'][:,0,:] for r in all_response] # [B, num_client, K]
        all_PC = [r['PC'][:,0,:,:] for r in all_response] # [B, num_client, K]
        all_UI = [r['UI'][:,0,:,:] for r in all_response] # [B, num_client, K]

        if not all(x.shape == all_client_embeddings[0].shape for x in all_client_embeddings):
            raise RuntimeError(f"Batch {batch_idx}: Mismatch in UE counts between clients. Cannot stack.")
        
        ## Calculate rate
        all_DS_stack = torch.stack(all_DS, dim=1)
        all_PC_stack = torch.stack(all_PC, dim=1)
        all_UI_stack = torch.stack(all_UI, dim=1)
        global_rate = rate_from_component(all_DS_stack, all_PC_stack, all_UI_stack, numAntenna=num_antenna)
        # min_rate_per_sample, bottleneck_ue_idx = torch.min(global_rate, dim=1)
        temperature = 0.001
        bottleneck_indicator = F.softmax(-global_rate / temperature, dim=1)
        bottleneck_indicator = bottleneck_indicator.reshape(-1,1)

        rank_indices = torch.argsort(torch.argsort(global_rate, dim=1), dim=1)
        normalized_rank = rank_indices.float() / (global_rate.shape[1] - 1)
        normalized_rank = normalized_rank.reshape(-1, 1)

        # Total DS across all APs for contribution ratio
        total_DS = all_DS_stack.sum(dim=1)  # [B, K]
        total_Interf = (all_PC_stack.sum(dim=2) + all_UI_stack.sum(dim=2)).sum(dim=1) # [B, K]

        global_ue_context = torch.sum(torch.stack(all_client_embeddings), dim=0)
        for client_id, (_, batch) in enumerate(zip(all_response, all_loader)):

            other_AP_indices = list(range(client_id)) + list(range(client_id + 1, num_client))

            ## DS, PC, UI
            other_DS = torch.stack(all_DS[:client_id] + all_DS[client_id+1:], dim=1)
            other_PC = torch.stack(all_PC[:client_id] + all_PC[client_id+1:], dim=1).sum(dim=2)
            other_UI = torch.stack(all_UI[:client_id] + all_UI[client_id+1:], dim=1).sum(dim=2)

            # Rate pack from other APs (DC, PC, UI)
            other_pack = []
            keys_needed = ['DS', 'PC', 'UI']
            for j in range(num_client):
                if j != client_id:
                    full_data = all_response[j]
                    filtered_data = {k: full_data[k] for k in keys_needed}
                    other_pack.append(filtered_data)

            aug_batch = batch.clone()
            device = aug_batch['UE'].x.device

            # GAP 
            other_AP = torch.stack(all_AP_embeddings[:client_id] + all_AP_embeddings[client_id+1:], dim=1)
            num_batch, num_GAP, feat_dim = other_AP.shape

            # # Enhance the GAP node feature
            # gap_total_DS = other_DS.sum(dim=2) # [B, num_GAP]

            # # Per-GAP total interference: [B, num_GAP]  
            # gap_total_interf = (other_PC + other_UI).sum(dim=2)   # [B, num_GAP]

            # # Per-GAP load (signal-to-interference ratio)
            # gap_load = gap_total_DS / (gap_total_interf + 1e-6)  # [B, num_GAP]

            # # Concatenate to GAP features
            # gap_features = torch.cat([
            #     other_AP.reshape(num_batch, num_GAP, feat_dim),  # [B, num_GAP, feat_dim]
            #     # gap_total_DS.unsqueeze(-1),    # [B, num_GAP, 1]
            #     # gap_total_interf.unsqueeze(-1), # [B, num_GAP, 1]
            #     # gap_load.unsqueeze(-1)          # [B, num_GAP, 1]
            # ], dim=-1).reshape(-1, feat_dim + 1)

            gap_features = other_AP.reshape(-1, feat_dim)

            aug_batch['GAP'].x = gap_features.to(device)
            
            # GAP - AP
            ap_indices = torch.arange(num_batch, device=device).repeat_interleave(num_GAP)
            gap_indices = torch.arange(num_batch * num_GAP, device=device)
            edge_index_inteference = torch.stack([gap_indices, ap_indices], dim=0)
            edge_index_inteference_back = torch.stack([ap_indices, gap_indices], dim=0)

            other_edge = torch.stack(all_edge_embeddings[:client_id] + all_edge_embeddings[client_id+1:], dim=1)
            num_total_ue, _, edge_feat_dim = other_edge.shape
            num_ue_per_graph = num_total_ue // num_batch
            edge_reshaped = other_edge.reshape(num_batch, num_ue_per_graph, num_GAP, edge_feat_dim)
            edge_summed = edge_reshaped.mean(dim=1) # sum or mean
            edge_attr_inteference = edge_summed.reshape(-1, feat_dim)

            ## Enhace the edge

            # gap_DS = other_DS.mean(dim=2, keepdim=True)  # [B, num_GAP, 1]
            # gap_interference = (other_PC + other_UI).mean(dim=2, keepdim=True)  # [B, num_GAP, 1]

            # Concatenate to edge attr
            # edge_attr_inteference = torch.cat([
            #     edge_summed,  # [B, num_GAP, feat_dim]
            #     gap_DS,       # [B, num_GAP, 1]
            #     gap_interference  # [B, num_GAP, 1]
            # ], dim=-1).reshape(-1, feat_dim + 2)
            edge_attr_inteference = edge_summed.reshape(-1, feat_dim)

            aug_batch['GAP', 'cross', 'AP'].edge_index = edge_index_inteference
            aug_batch['GAP', 'cross', 'AP'].edge_attr = edge_attr_inteference

            aug_batch['AP', 'cross-back', 'GAP'].edge_index = edge_index_inteference_back
            aug_batch['AP', 'cross-back', 'GAP'].edge_attr = edge_attr_inteference

            # Global UE context (from other APs)
            new_ue_features = ((global_ue_context - all_client_embeddings[client_id]) / (num_client - 1)).to(device)

            # Contribution ratio: how much this AP contributes to each UE's signal
            local_DS = all_DS[client_id]  # [B, K]
            contribution_ratio = (local_DS / (total_DS + 1e-6)).reshape(-1, 1)  # [B*K, 1]
            # log_rate_context = torch.log10(min_rate_per_sample + 1e-9).reshape(-1, 1).repeat(num_ue_per_graph, 1)

            # ap_conditioned_bottleneck = bottleneck_indicator * contribution_ratio

            # Context A: Global Embedding
            new_ue_features = ((global_ue_context - all_client_embeddings[client_id]) / (num_client - 1)).to(device)
            
            # Context B: Signal Contribution (My Signal / Total Signal)
            local_DS = all_DS[client_id]
            contribution_ratio = (local_DS / (total_DS + 1e-9)).reshape(-1, 1)
            
            # Context C: Interference Share (My Interference / Total Interference)
            # **CRITICAL NEW FEATURE**
            local_interf = (all_PC[client_id] + all_UI[client_id]).sum(dim=2)
            interference_share = (local_interf / (total_Interf + 1e-9)).reshape(-1, 1)

            # Context D: Global Quality (Log SINR)
            # **CRITICAL NEW FEATURE**
            # Tells the UE if it is in a "high power" or "low power" regime globally
            global_sinr = (2 ** global_rate - 1).reshape(-1, 1)


            aug_batch['UE'].x = torch.cat(
                [
                    aug_batch['UE'].x, # init dim
                    new_ue_features,   # out_channel
                    bottleneck_indicator,    # [Focus: Soft attention]
                    # normalized_rank,         # [Focus: Hard priority] <--- NEW
                    contribution_ratio,      # [Action: How much I help]
                    interference_share,      # [Action: How much I hurt] <--- NEW
                    global_sinr              # [State: How good is the user globally] <--- NEW
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


############################## Server FL function #############################
# sample_clients: randomly select clients for each round, return a list []
# FedAvg
# FedAdam
###################################################################################


@torch.no_grad()
def sample_clients(client_fraction, num_clients):
    num_selected = max(1, int(client_fraction * num_clients))
    return sorted(random.sample(range(num_clients), num_selected))


class FedAvg:
    def __init__(self, client_fraction=0.3, seed=1712):
        self.client_fraction = client_fraction
        if seed is not None:
            random.seed(seed)

    def aggregate(self, global_model, local_models, selected_clients):

        local_weights = [copy.deepcopy(model.state_dict()) for model in local_models]
        global_weights = copy.deepcopy(global_model.state_dict()) 
        if not selected_clients:
            return global_weights
        
        avg_state = copy.deepcopy(local_weights[selected_clients[0]])
        n_clients = len(selected_clients)
        
        for k in avg_state.keys():
            if isinstance(avg_state[k], torch.Tensor):
                is_int = not torch.is_floating_point(avg_state[k])
                if is_int:
                    avg_state[k] = avg_state[k].float()
                for i in selected_clients[1:]:
                    w_client = local_weights[i][k]
                    if is_int:
                        w_client = w_client.float()
                    avg_state[k] += w_client
                avg_state[k] = avg_state[k] / n_clients
                if is_int:
                    avg_state[k] = avg_state[k].long()
                    
        return avg_state
    

class FedAdam:
    def __init__(self, client_fraction=0.3, server_lr=1e-2, beta1=0.9, beta2=0.99, epsilon=1e-3, seed=1712):
        """
        Args:
            server_lr: Learning rate for the server-side update.
            beta1, beta2: Adam momentum parameters.
            epsilon: Adam numerical stability term.
        """
        self.client_fraction = client_fraction
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {}
        self.v = {}
        
        if seed is not None:
            random.seed(seed)

    def aggregate(self, global_model, local_models, selected_clients):

        local_weights = [copy.deepcopy(model.state_dict()) for model in local_models]
        global_weights = copy.deepcopy(global_model.state_dict()) 
        
        if not selected_clients:
            return global_weights

        # 1. Compute Simple Average (FedAvg step)
        # We start by calculating the average of the client parameters
        avg_weights = copy.deepcopy(local_weights[selected_clients[0]])
        n_clients = len(selected_clients)
        
        for k in avg_weights.keys():
            # Skip complex logic for simplicity, assuming float weights
            # (In production, apply the same int/float check as FedAvg above)
            for i in selected_clients[1:]:
                avg_weights[k] += local_weights[i][k]
            avg_weights[k] = avg_weights[k] / n_clients

        # 2. Compute Pseudo-Gradient
        # Gradient = Global - Average
        # (This represents the direction the clients want to move AWAY from the current global)
        pseudo_gradient = {}
        for k in global_weights.keys():
            if k in avg_weights and isinstance(global_weights[k], torch.Tensor) and torch.is_floating_point(global_weights[k]):
                pseudo_gradient[k] = global_weights[k] - avg_weights[k]
            else:
                # Non-trainable params (like integer buffers) are just copied from average
                pseudo_gradient[k] = None

        # 3. Apply Adam Update to Global Weights
        updated_global_weights = copy.deepcopy(global_weights)

        for k in pseudo_gradient.keys():
            grad = pseudo_gradient[k]
            
            # If no gradient (e.g. integer buffer), just update with the average value
            if grad is None:
                updated_global_weights[k] = avg_weights[k]
                continue

            # Initialize Adam buffers if they don't exist
            if k not in self.m:
                self.m[k] = torch.zeros_like(grad)
                self.v[k] = torch.zeros_like(grad) # + self.epsilon**2 (optional variant)

            # --- Adam Algorithm ---
            
            # Update biased first moment estimate
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected moments (Optional in some FL papers, but standard in Adam)
            # Usually in FL papers (Reddi et al.), they skip bias correction for simplicity 
            # or treat it as absorbed in LR. We will use the standard version without bias correction
            # as typically implemented in FedOpt papers for stability.
            
            m_hat = self.m[k]
            v_hat = self.v[k]

            # Update parameters
            # w_new = w_old - lr * m / (sqrt(v) + eps)
            # Note: Since our "gradient" was (Global - Avg), moving *against* the gradient
            # means moving *towards* the Avg. 
            # However, standard notation is w_{t+1} = w_t - \eta \Delta_t.
            # Here \Delta_t = w_t - w_{avg}.
            # So w_{t+1} = w_t - \eta * Adam(w_t - w_{avg}).
            
            update_term = m_hat / (torch.sqrt(v_hat) + self.epsilon)
            updated_global_weights[k] = global_weights[k] - self.server_lr * update_term

        return updated_global_weights
    
class FedGM:
    """
    Federated Gradient Matching (FedGM).
    Solves the 'Conflicting Gradient' problem in Joint Optimization / Max-Min Fairness.
    
    If Client A's gradient points in a direction that hurts Client B (negative dot product),
    this algorithm projects Client A's gradient onto the normal plane of Client B's gradient.
    """
    def __init__(self, client_fraction=0.3, server_lr=1e-2, seed=1712):
        self.client_fraction = client_fraction
        self.server_lr = server_lr
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def flatten_grads(self, grads_dict):
        """Helper: Flatten a gradient dictionary into a single 1D tensor."""
        flat_list = []
        # Sort keys to ensure consistent ordering across all clients
        for key in sorted(grads_dict.keys()):
            if grads_dict[key] is not None:
                flat_list.append(grads_dict[key].view(-1))
        return torch.cat(flat_list)

    def unflatten_grads(self, flat_grad, template_dict):
        """Helper: Reshape a 1D tensor back into the model's state_dict structure."""
        unflattened = {}
        ptr = 0
        for key in sorted(template_dict.keys()):
            if template_dict[key] is not None:
                tensor_shape = template_dict[key].shape
                num_elements = template_dict[key].numel()
                
                # Extract slice and reshape
                unflattened[key] = flat_grad[ptr : ptr + num_elements].view(tensor_shape)
                ptr += num_elements
            else:
                unflattened[key] = None
        return unflattened
    
    

    def aggregate(self, global_model, local_models, selected_clients):
        """
        Aggregates gradients by projecting conflicting ones.
        
        Args:
            global_model: The global model (needed for current weights and shapes).
            local_gradients: List of gradient dictionaries from clients.
            selected_clients: Indices of the clients selected for this round.
            
        Returns:
            new_weights: The updated state_dict for the global model.
        """
        if not selected_clients:
            return global_model.state_dict()
        
        local_gradients = get_model_gradients(global_model, local_models)

        # 1. Flatten all client gradients into a list of vectors
        # Result: List of tensors, each with shape [Total_Params]
        grads_list = []
        valid_indices = []
        
        for i in selected_clients:
            if local_gradients[i] is None:
                continue
            grads_list.append(self.flatten_grads(local_gradients[i]))
            valid_indices.append(i)

        if not grads_list:
            return global_model.state_dict()

        # 2. Gradient Matching / Projection Process (PCGrad)
        # We work on a copy so we don't modify the originals during comparison
        projected_grads = copy.deepcopy(grads_list)
        num_participants = len(projected_grads)
        
        # Shuffle the order of comparison (randomized fairness)
        # This prevents the first client in the list from always dominating projections
        order = list(range(num_participants))
        random.shuffle(order)

        for i in range(num_participants):
            idx_i = order[i]
            grad_i = projected_grads[idx_i]
            
            # Compare grad_i against all other gradients j
            for j in range(num_participants):
                idx_j = order[j]
                if idx_i == idx_j: 
                    continue
                
                grad_j = projected_grads[idx_j]
                
                # Calculate Dot Product (Similarity)
                inner_prod = torch.dot(grad_i, grad_j)
                
                # If Negative (Conflict), project grad_i to be orthogonal to grad_j
                # This removes the part of grad_i that "hurts" client j
                if inner_prod < 0:
                    denom = torch.dot(grad_j, grad_j)
                    if denom > 1e-8: # Numerical stability check
                        # Projection: g_i = g_i - ( (g_i . g_j) / ||g_j||^2 ) * g_j
                        coef = inner_prod / denom
                        grad_i -= coef * grad_j
                        
            # Update the gradient in our list after all projections
            projected_grads[idx_i] = grad_i

        # 3. Average the now-aligned gradients
        avg_flat_grad = torch.stack(projected_grads).mean(dim=0)

        # 4. Apply Update to Global Model
        # W_new = W_old - lr * GM_Gradient
        new_weights = copy.deepcopy(global_model.state_dict())
        
        # Convert the flat gradient vector back to a dictionary
        update_dict = self.unflatten_grads(avg_flat_grad, local_gradients[valid_indices[0]])
        
        for key in new_weights.keys():
            # Only update float parameters that have gradients
            if key in update_dict and update_dict[key] is not None:
                # Ensure device compatibility
                update_val = update_dict[key].to(new_weights[key].device)
                new_weights[key] -= self.server_lr * update_val

        return new_weights
    
def get_model_gradients(global_model, local_models):
    """
    Calculates the 'pseudo-gradient' for each client by comparing 
    their trained weights to the frozen global weights.
    
    Gradient = (Global_Weights - Local_Weights)
    
    Args:
        global_model: The server model (frozen state before aggregation).
        local_models: List of trained client models.
        
    Returns:
        local_gradients: A list of dictionaries (state_dicts) containing gradients.
    """
    local_gradients = []
    
    # 1. Get global state once to save time
    # Ensure we are on the same device (CPU is safest for storage)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    
    for client_model in local_models:
        client_grad = {}
        client_state = client_model.state_dict()
        
        for key, global_tensor in global_state.items():
            # We only calculate gradients for trainable floating-point parameters
            # Skip integer buffers (like num_batches_tracked)
            if not torch.is_floating_point(global_tensor):
                continue
            
            # Skip keys that might be missing (safety check)
            if key not in client_state:
                continue
                
            local_tensor = client_state[key].cpu()
            
            # Calculate the difference
            # Direction: The client moved from Global to Local.
            # So the 'force' (gradient) that pushed it there is roughly (Global - Local).
            client_grad[key] = global_tensor - local_tensor
            
        local_gradients.append(client_grad)
        
    return local_gradients