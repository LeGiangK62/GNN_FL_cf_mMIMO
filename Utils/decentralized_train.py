import torch
import copy 
import random
from Utils.comm import (
    variance_calculate, rate_calculation, 
    component_calculate, rate_from_component, package_calculate
)


def loss_function(graphData, nodeFeatDict, clientResponse, tau, rho_p, rho_d, num_antenna):
    num_graphs = graphData.num_graphs
    num_UEs = graphData['UE'].x.shape[0]//num_graphs
    num_APs = graphData['AP'].x.shape[0]//num_graphs
    
    pilot_matrix = graphData['UE'].x.reshape(num_graphs, num_UEs, -1)
    large_scale = graphData['AP','down','UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs)
    power = nodeFeatDict['UE'].reshape(num_graphs, num_UEs, -1)
    power_matrix = power[:,:,-1][:, None, :]

    channel_variance = variance_calculate(large_scale, pilot_matrix, tau, rho_p)
    # scaling for power constraints
    sum_weighted = torch.sum(power_matrix * channel_variance, dim=1, keepdim=True)   # shape (M,1)
    power_matrix = power_matrix / torch.maximum(sum_weighted, torch.ones_like(sum_weighted)/num_antenna)
    
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
        optimizer.zero_grad() 
        x_dict, _, _ = model(batch)
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
def fl_eval_rate(
        dataLoader, models,
        tau, rho_p, rho_d, num_antenna
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_power = []
    all_large_scale = []
    all_phi = []
    for batch_idx, batches_at_k in enumerate(zip(*dataLoader)):
        per_batch_power = []
        per_batch_large_scale = []
        for ap_idx, (model, batch) in enumerate(zip(models, batches_at_k)):
            model.eval()
            # iterate over all batch of each AP
            batch = batch.to(device)
            num_graphs = batch.num_graphs
            num_UEs = batch['UE'].x.shape[0]//num_graphs
            num_APs = batch['AP'].x.shape[0]//num_graphs
            
            x_dict, edge_dict, edge_index = model(batch)
            power = x_dict['UE'].reshape(num_graphs, num_UEs, -1)
            power_matrix = power[:,:,-1][:, None, :]
            pilot_matrix = batch['UE'].x.reshape(num_graphs, num_UEs, -1)
            large_scale = batch['AP','down','UE'].edge_attr.reshape(num_graphs, num_APs, num_UEs)
            channel_variance = variance_calculate(large_scale, pilot_matrix, tau, rho_p)
            sum_weighted = torch.sum(power_matrix * channel_variance, dim=1, keepdim=True)   # shape (M,1)
            power_matrix = power_matrix / torch.maximum(sum_weighted, torch.ones_like(sum_weighted)/num_antenna)
            
            per_batch_power.append(power_matrix)
            per_batch_large_scale.append(large_scale)
            # per_batch_phi.append(power_matrix)
        per_batch_phi = pilot_matrix
        per_batch_power = torch.cat(per_batch_power, dim=1)
        per_batch_large_scale = torch.cat(per_batch_large_scale, dim=1)
         
        all_power.append(per_batch_power)
        all_large_scale.append(per_batch_large_scale)
        all_phi.append(per_batch_phi)
    
    total_min_rate = 0.0
    total_samples = 0.0
    for each_power, each_large_scale, each_phi in zip(all_power, all_large_scale, all_phi):
        num_graphs = len(each_power)
        each_channel_variance = variance_calculate(each_large_scale, each_phi, tau=tau, rho_p=rho_p)
        rate = rate_calculation(each_power, each_large_scale, each_channel_variance, each_phi, rho_d=rho_d, num_antenna=num_antenna)
        min_rate, _ = torch.min(rate, dim=1)
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
    def __init__(self, global_model, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}
        self.v = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()}

    def aggregate(self, global_model, local_weights, local_sizes=None):
        """FedAdam aggregation with safe dtype handling."""
        with torch.no_grad():
            # Weighted FedAvg first
            if local_sizes is None:
                avg_weights = copy.deepcopy(local_weights[0])
                for key in avg_weights.keys():
                    # Skip non-floating tensors (e.g., num_batches_tracked)
                    if not torch.is_floating_point(avg_weights[key]):
                        continue
                    for i in range(1, len(local_weights)):
                        avg_weights[key] += local_weights[i][key].to(avg_weights[key].device)
                    avg_weights[key] /= float(len(local_weights))
            else:
                total = sum(local_sizes)
                avg_weights = {}
                for key in local_weights[0].keys():
                    if not torch.is_floating_point(local_weights[0][key]):
                        avg_weights[key] = local_weights[0][key]
                        continue
                    avg_weights[key] = sum(
                        local_weights[i][key] * (local_sizes[i] / total)
                        for i in range(len(local_weights))
                    )

            # Compute "gradient" = difference between global and averaged local weights
            grad = {
                k: global_model.state_dict()[k] - avg_weights[k]
                for k in avg_weights.keys() if torch.is_floating_point(avg_weights[k])
            }

            # Update moment estimates
            for k in grad.keys():
                self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grad[k]
                self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grad[k] ** 2)

            # Update global model parameters
            new_state = {}
            for k, v in global_model.state_dict().items():
                if torch.is_floating_point(v):
                    m_hat = self.m[k] / (1 - self.beta1)
                    v_hat = self.v[k] / (1 - self.beta2)
                    new_state[k] = v - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
                else:
                    new_state[k] = v  # leave non-float tensors unchanged

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
            if not torch.is_floating_point(avg_state[k]):
                continue
            for i in selected_clients[1:]:
                avg_state[k] += local_weights[i][k]
            avg_state[k] = avg_state[k] / float(len(selected_clients))

        return avg_state
    
    

# Knowledge Graph handle

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
            with torch.no_grad():
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