import re
import torch
import copy 
import random
import torch.nn.functional as F
from Utils.comm import (
    variance_calculate, rate_calculation, 
    component_calculate, rate_from_component,
    power_from_raw
)


def loss_function(graphData, nodeFeatDict, edgeDict, clientResponse, tau, rho_p, rho_d, num_antenna, isEdgeUpd):
    num_graphs = graphData.num_graphs
    num_UEs = graphData['UE'].x.shape[0]//num_graphs
    num_APs = graphData['AP'].x.shape[0]//num_graphs
    
    
    pilot_matrix = graphData['UE'].x.reshape(num_graphs, num_UEs, -1)
    ap_gate = nodeFeatDict['AP'].reshape(num_graphs, num_APs, -1)
    
    if isEdgeUpd:
        large_scale = graphData['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
        power_matrix_raw = edgeDict['AP','down','UE'].reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
    else:
        large_scale = graphData['AP','down','UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs)
        power_matrix_raw = nodeFeatDict['UE'].reshape(num_graphs, num_UEs, -1)
        power_matrix_raw = power_matrix_raw[:, :, -1][:, None, :]

    large_scale = torch.expm1(large_scale)
    channel_variance = variance_calculate(large_scale, pilot_matrix, tau, rho_p)
    
    power_matrix = power_from_raw(power_matrix_raw, ap_gate, channel_variance, num_antenna)
    # power_matrix = torch.softmax(power_matrix, dim=1)
    # power_matrix = power_matrix/channel_variance
    
    DS_k, PC_k, UI_k = component_calculate(power_matrix, channel_variance, large_scale, pilot_matrix, rho_d=rho_d)
    
    all_DS = [DS_k] + [r['DS'] for r in clientResponse]
    all_PC = [PC_k] + [r['PC'] for r in clientResponse]
    all_UI = [UI_k] + [r['UI'] for r in clientResponse]

    all_DS = torch.cat(all_DS, dim=1)
    all_PC = torch.cat(all_PC, dim=1)   
    all_UI = torch.cat(all_UI, dim=1) 
    
    rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)
    min_rate, _ = torch.min(rate, dim=1)
    loss = torch.mean(min_rate) 

    return -loss




def fl_train(
        dataLoader, responseInfo, model, optimizer,
        tau, rho_p, rho_d, num_antenna, isEdgeUpd=False
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    local_gradients = {}
    total_loss = 0.0
    total_graphs = 0
    for batch, response in zip(dataLoader , responseInfo):
        batch = batch.to(device)
        num_graph = batch.num_graphs
        optimizer.zero_grad() 
        x_dict, attr_dict, _ = model(batch)
        loss = loss_function(
            batch, x_dict, attr_dict, response, 
            tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna, isEdgeUpd=isEdgeUpd
        )
        loss.backward()
        optimizer.step()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if name not in local_gradients:
                    local_gradients[name] = param.grad.clone()
                else:
                    local_gradients[name] += param.grad.clone()
    
        total_loss += loss.item() * num_graph
        total_graphs += num_graph

    return total_loss/total_graphs, local_gradients


@torch.no_grad()
def fl_eval_rate(
        dataLoader, models,
        tau, rho_p, rho_d, num_antenna, isEdgeUpd=False,
        eval_mode=False
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_power = []
    all_large_scale = []
    all_phi = []
    for batch_idx, batches_at_k in enumerate(zip(*dataLoader)):
        per_batch_power = []
        per_batch_large_scale = []
        per_batch_phi = []
        for ap_idx, (model, batch) in enumerate(zip(models, batches_at_k)):
            model.eval()
            # iterate over all batch of each AP
            batch = batch.to(device)
            num_graphs = batch.num_graphs
            num_UEs = batch['UE'].x.shape[0]//num_graphs
            num_APs = batch['AP'].x.shape[0]//num_graphs
            # large_scale_mean, large_scale_std = batch.mean, batch.std
            
            x_dict, edge_dict, edge_index = model(batch)
            ap_gate = x_dict['AP'].reshape(num_graphs, num_APs, -1)
            if isEdgeUpd:
                large_scale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
                power_matrix_raw = edge_dict['AP','down','UE'].reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
            else:
                large_scale = batch['AP','down','UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs)
                power_matrix_raw = x_dict['UE'].reshape(num_graphs, num_UEs, -1)
                power_matrix_raw = power_matrix_raw[:, :, -1][:, None, :]
                
            large_scale = torch.expm1(large_scale)
    
            pilot_matrix = batch['UE'].x.reshape(num_graphs, num_UEs, -1)
            channel_variance = variance_calculate(large_scale, pilot_matrix, tau, rho_p)
            # # sum_weighted = torch.sum(power_matrix * channel_variance, dim=2, keepdim=True)   # shape (M,1)
            # # power_matrix = power_matrix / torch.maximum(sum_weighted, torch.ones_like(sum_weighted)) 
            # # power_matrix /= num_antenna
            # power_matrix = F.softplus(power_matrix) 
            # sum_ue = torch.sum(power_matrix * channel_variance, dim=2, keepdim=True)  # [B, M, 1]
            # alpha = torch.clamp((1.0 / num_antenna) / (sum_ue), max=1.0)   # [B, M, 1]
            # power_matrix = power_matrix * alpha 
            # # power_matrix = torch.softmax(power_matrix, dim=1)
            # # power_matrix = power_matrix/channel_variance
            power_matrix = power_from_raw(power_matrix_raw, ap_gate, channel_variance, num_antenna)
            
            per_batch_power.append(power_matrix)
            per_batch_large_scale.append(large_scale)
            per_batch_phi.append(pilot_matrix.unsqueeze(1))
            
        per_batch_phi = torch.cat(per_batch_phi, dim=1) 
        per_batch_power = torch.cat(per_batch_power, dim=1)
        per_batch_large_scale = torch.cat(per_batch_large_scale, dim=1)
        
        if per_batch_phi.shape[1] > 1:
            ref = per_batch_phi[:, 0, :, :]
            if not torch.allclose(per_batch_phi[:, 1:, :, :],
                                ref.unsqueeze(1).expand_as(per_batch_phi[:, 1:, :, :]),
                                atol=1e-6, rtol=0):
                raise ValueError("UE/pilot order differs across clients for this batch. "
                                "Use a shared sampler or disable per-client shuffle.")
         
        all_power.append(per_batch_power)
        all_large_scale.append(per_batch_large_scale)
        all_phi.append(per_batch_phi)
    
    total_min_rate = 0.0
    total_samples = 0.0
    for each_power, each_large_scale, each_phi in zip(all_power, all_large_scale, all_phi):
        num_graphs = len(each_power)
        each_phi = each_phi[:,0,:,:]
        each_channel_variance = variance_calculate(each_large_scale, each_phi, tau=tau, rho_p=rho_p)
        all_DS, all_PC, all_UI = component_calculate(each_power, each_channel_variance, each_large_scale, each_phi, rho_d=rho_d)
        # rate = rate_calculation(each_power, each_large_scale, each_channel_variance, each_phi, rho_d=rho_d, num_antenna=num_antenna)
        rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)
        min_rate, _ = torch.min(rate, dim=1)
        if eval_mode: return min_rate
        min_rate = torch.mean(min_rate)
        total_min_rate += min_rate.item() * num_graphs
        total_samples += num_graphs

    return total_min_rate/total_samples


# FL functions
def average_weights(local_weights):
    """Average model parameters from all clients (FedAvg)."""
    avg_weights = copy.deepcopy(local_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
    return avg_weights


class FedAdam:
    def __init__(self, global_model, client_fraction=0.3, seed=None,
                 lr=1e-2, beta1=0.9, beta2=0.99, eps=1e-8):
        self.client_fraction = client_fraction
        if seed is not None:
            random.seed(seed)
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0  # step counter

        # moments only for updatable float params (skip BN buffers and personal layers)
        def _updatable(k, v):
            if not torch.is_floating_point(v): return False
            if 'convs_per' in k: return False
            if any(s in k for s in ('running_mean','running_var','num_batches_tracked')): return False
            return True

        templ = global_model.state_dict()
        self.m = {k: torch.zeros_like(v) for k, v in templ.items() if _updatable(k, v)}
        self.v = {k: torch.zeros_like(v) for k, v in templ.items() if _updatable(k, v)}

    @torch.no_grad()
    def sample_clients(self, num_clients):
        n = max(1, int(self.client_fraction * num_clients))
        return sorted(random.sample(range(num_clients), n))

    @torch.no_grad()
    def aggregate(self, global_model, local_weights):
        """
        local_weights: list of state_dicts from SELECTED clients (already filtered in main).
        """
        # FedAvg over selected clients (skip BN/personal & non-float)
        avg = copy.deepcopy(local_weights[0])
        def _skip(k, v):
            return (
                'convs_per' in k or
                not torch.is_floating_point(v) or
                any(s in k for s in ('running_mean','running_var','num_batches_tracked'))
            )

        for k in avg.keys():
            # if _skip(k, avg[k]): continue
            for w in local_weights[1:]:
                avg[k] += w[k]
            avg[k] /= float(len(local_weights))

        # "gradient" = global - avg
        g = {}
        for k, v in global_model.state_dict().items():
            # if _skip(k, v): continue
            g[k] = v - avg[k]

        # Adam moments + bias correction
        self.t += 1
        b1t = 1.0 - (self.beta1 ** self.t)
        b2t = 1.0 - (self.beta2 ** self.t)

        new_state = {}
        for k, v in global_model.state_dict().items():
            # if _skip(k, v):
            #     new_state[k] = v
            #     continue
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g[k] * g[k])
            m_hat = self.m[k] / b1t
            v_hat = self.v[k] / b2t
            new_state[k] = v - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

        global_model.load_state_dict(new_state)
        return copy.deepcopy(global_model.state_dict())


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
    
    

class FedAvgGradMatch:
    def __init__(self, client_fraction=0.3, seed=None, mu=0.1):
        """
        Args:
            client_fraction: fraction of clients to sample each round (0 < C ≤ 1)
            seed: optional random seed for reproducibility
            mu: prox regularization to match local gradients to global grad
        """
        self.client_fraction = client_fraction
        self.mu = mu
        if seed is not None:
            random.seed(seed)
    
    @torch.no_grad()
    def sample_clients(self, num_clients):
        """Randomly select participating clients for this round."""
        num_selected = max(1, int(self.client_fraction * num_clients))
        return sorted(random.sample(range(num_clients), num_selected))
    
    def compute_global_gradient(self, local_gradients, selected_clients):
        """Compute the global gradient by averaging gradients from selected clients."""
        global_grad = {}
        for key in local_gradients[selected_clients[0]].keys():
            if not torch.is_floating_point(local_gradients[selected_clients[0]][key]):
                continue
            global_grad[key] = torch.zeros_like(local_gradients[selected_clients[0]][key])
            for client_idx in selected_clients:
                global_grad[key] += local_gradients[client_idx][key]
            global_grad[key] /= len(selected_clients)
        return global_grad
    
    def apply_gradient_matching(self, avg_weights, local_gradients, selected_clients, key, global_grad):
        """Apply gradient matching (proximal regularization) to align local gradients with the global gradient."""
        if key not in local_gradients[selected_clients[0]]:
            return avg_weights
        
        local_grad = None
        for client_idx in selected_clients:
            if local_gradients is not None:
                if local_grad is None:
                    local_grad = local_gradients[client_idx][key]
                else:
                    local_grad += local_gradients[client_idx][key]

        # Gradient matching: Penalize the difference between local gradients and global gradients
        if local_grad is not None:
            gradient_penalty = torch.sum((local_grad - global_grad[key]) ** 2)
            avg_weights -= self.mu * gradient_penalty  # Apply the proximal regularization

        return avg_weights

    def aggregate(self, local_weights, selected_clients, local_gradients=None):
        avg_state = copy.deepcopy(local_weights[selected_clients[0]])

        # Compute the global gradient (average of selected clients' gradients)
        if local_gradients is not None:
            global_grad = self.compute_global_gradient(local_gradients, selected_clients)
        
        # Update model weights by averaging local weights (FedAvg)
        for k in avg_state.keys():
            if "running_mean" in k or "running_var" in k:
                continue 
            if 'convs_per' in k: continue
            if not torch.is_floating_point(avg_state[k]):
                continue
            for i in selected_clients[1:]:
                avg_state[k] += local_weights[i][k]
            avg_state[k] = avg_state[k] / float(len(selected_clients))

            # Apply gradient matching (proximal regularization)
            if local_gradients is not None:
                avg_state[k] = self.apply_gradient_matching(
                    avg_state[k], local_gradients, selected_clients, k, global_grad
                )

        return avg_state
    
    
class FedProx:
    def __init__(self, client_fraction=0.3, mu=0.1, seed=None):
        """
        Args:
            client_fraction: fraction of clients to sample each round (0 < C ≤ 1)
            mu: Proximal regularization parameter to penalize the deviation of local models from the global model
            seed: optional random seed for reproducibility
        """
        self.client_fraction = client_fraction
        self.mu = mu
        if seed is not None:
            random.seed(seed)
    
    @torch.no_grad()
    def sample_clients(self, num_clients):
        """Randomly select participating clients for this round."""
        num_selected = max(1, int(self.client_fraction * num_clients))
        return sorted(random.sample(range(num_clients), num_selected))
    
    def aggregate(self, local_weights, selected_clients, global_model):
        """FedProx aggregation with proximal regularization."""
        avg_state = copy.deepcopy(local_weights[selected_clients[0]])

        # Apply proximal regularization to the aggregation
        for k in avg_state.keys():
            if not torch.is_floating_point(avg_state[k]):
                continue
            if 'convs_per' in k: continue
            for i in selected_clients[1:]:
                avg_state[k] += local_weights[i][k]
            avg_state[k] = avg_state[k] / float(len(selected_clients))
            
            # Proximal regularization to align local updates with the global model
            avg_state[k] -= self.mu * (avg_state[k] - global_model.state_dict()[k])

        return avg_state
    

# Knowledge Graph handle

def get_global_info(
        loaderData, localModels, optimizers,
        tau, rho_p, rho_d, num_antenna, isEdgeUpd
    ):
    num_client = len(localModels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    send_to_server = [[] for _ in range(num_client)] 
    for batches  in zip(*loaderData):                       # sync step across APs
        
        for client_idx, (model, opt, batch) in enumerate(zip(localModels, optimizers, batches)):
            # Check batch here? something wrong?
            model.eval()
            batch = batch.to(device)
            with torch.no_grad():
                x_dict, edge_dict, edge_index = model(batch)
                # DS_single, PC_single, UI_single = package_calculate(batch, x_dict, tau, rho_p, rho_d)
                ##
                num_graphs = batch.num_graphs
                num_UEs = x_dict['UE'].shape[0] // num_graphs
                num_APs = x_dict['AP'].shape[0] // num_graphs
                ap_gate = x_dict['AP'].reshape(num_graphs, num_APs, -1)
                
                if isEdgeUpd:
                    largeScale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,0]
                    power = edge_dict['AP','down','UE'].reshape(num_graphs, num_APs, num_UEs, -1)[:,:,:,-1]
                else:
                    largeScale = batch['AP', 'down', 'UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs)
                    power = x_dict['UE'].reshape(num_graphs, num_UEs, -1)[:, :, -1][:, None, :]
                    
                largeScale = torch.expm1(largeScale)
                phiMatrix = batch['UE'].x.reshape(num_graphs, num_UEs, -1)
                channelVariance = variance_calculate(largeScale, phiMatrix, tau, rho_p)
                
                # sum_weighted = torch.sum(power * channelVariance, dim=2, keepdim=True)   # shape (M,1)
                # power = power / torch.maximum(sum_weighted, torch.ones_like(sum_weighted))
                # power /= num_antenna
                # power = F.softplus(power) 
                # sum_ue = torch.sum(power * channelVariance, dim=2, keepdim=True)  # [B, M, 1]
                # alpha = torch.clamp((1.0 / num_antenna) / (sum_ue), max=1.0)   # [B, M, 1]
                # power = power * alpha 
                
                power = power_from_raw(power, ap_gate, channelVariance, num_antenna)
                # power = torch.softmax(power, dim=1)
                # power = power/channelVariance
    

                DS_single, PC_single, UI_single = component_calculate(power, channelVariance, largeScale, phiMatrix, rho_d=rho_d)
                ##
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