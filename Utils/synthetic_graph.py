import torch

from torch_geometric.loader import DataLoader
from Utils.data_gen import single_syn_het_graph

## 3. Synthetic node

def return_graph(send_to_server):
    # input: send_to_server
    AP_feat = []
    channel_feat = []
    num_graphs = send_to_server[0]['num_graphs']
    for client_data in send_to_server:
        num_APs = client_data['AP'].shape[0]
        AP_data = client_data['AP'].reshape(num_APs, -1)
        channel_data = client_data['edge_attr'].reshape(num_APs, -1)
        
        AP_feat.append(AP_data)
        channel_feat.append(channel_data)
        
    AP_feat = torch.stack(AP_feat).permute(1, 0, 2) 
    channel_feat = torch.stack(channel_feat).permute(1, 0, 2) 

    AP_feat_avg = AP_feat.mean(dim=1, keepdim=True) 
    channel_feat_avg = channel_feat.mean(dim=1, keepdim=True) 
    num_AP = AP_feat.shape[1]

    all_data = []

    for each_AP in range(num_AP):
        data_single_AP = []
        
        AP_feat_without_current = torch.cat([AP_feat[:, :each_AP], AP_feat[:, each_AP + 1:]], dim=1)
        AP_feat_avg = AP_feat_without_current.mean(dim=1)
        
        channel_feat_without_current = torch.cat([channel_feat[:, :each_AP], channel_feat[:, each_AP + 1:]], dim=1)
        channel_feat_avg = channel_feat_without_current.mean(dim=1) 
        
        for each_sample in range(num_graphs):
            data = single_syn_het_graph(AP_feat_avg[0][:,None], channel_feat_avg[each_sample][:,None])
            data_single_AP.append(data)
        all_data.append(data_single_AP)
    complement_graph = [
        DataLoader(all_data[i], batch_size=num_graphs, shuffle=False)
        for i in range(num_AP)
    ]
    return complement_graph

## 4. Combine graph with the synthetic node

def combine_graph(original_graph, complement_graph):
    # original_graph: batches of local data of a single AP
    # Complement_graph: batches of tensor holds synthetic AP nodes 
    original_graph = original_graph.clone()
    complement_graph = complement_graph.clone()
    num_graphs = original_graph.num_graphs
    num_graphs_com = complement_graph.num_graphs
    if num_graphs != num_graphs_com: 
        raise ValueError(f'Number of graphs mismatched!: {num_graphs} and {num_graphs_com}')
    
    AP_node_org = original_graph.x_dict['AP']
    num_AP_org, AP_feat_dim_org = AP_node_org.shape
    AP_node_com = complement_graph.x_dict['AP']
    num_AP_com, AP_feat_dim_com = AP_node_com.shape
    if AP_feat_dim_org != AP_feat_dim_com: 
        raise ValueError(f'AP node feature dimension mismatched!: {AP_feat_dim_org} and {AP_feat_dim_com}')

    # Node concat
    original_graph['AP'].x = torch.cat([AP_node_org, AP_node_com], dim=0)
    original_graph['AP'].batch = torch.cat([original_graph['AP'].batch, complement_graph['AP'].batch], dim=0)

    # Downlink Concat
    AP_idx_org = original_graph.edge_index_dict['AP', 'down', 'UE'][0]
    AP_idx_com = complement_graph.edge_index_dict['AP', 'down', 'UE'][0] + num_AP_org

    edge_index = original_graph.edge_index_dict['AP', 'down', 'UE']
    num_edges = edge_index.shape[1]
    edge_index = torch.cat([edge_index, edge_index], dim=1)
    edge_index[0, num_edges:] = AP_idx_com
    original_graph['AP', 'down', 'UE'].edge_index = edge_index
    
    edge_attr = original_graph.edge_attr_dict['AP', 'down', 'UE']
    edge_attr_com = complement_graph.edge_attr_dict['AP', 'down', 'UE']
    original_graph['AP', 'down', 'UE'].edge_attr = torch.cat([edge_attr, edge_attr_com], dim=0)

    # Uplink Concat
    AP_idx_org = original_graph.edge_index_dict['UE', 'up', 'AP'][1]
    AP_idx_com = complement_graph.edge_index_dict['UE', 'up', 'AP'][1] + num_AP_org

    edge_index = original_graph.edge_index_dict['UE', 'up', 'AP']
    num_edges = edge_index.shape[1]
    edge_index = torch.cat([edge_index, edge_index], dim=1)
    edge_index[1, num_edges:] = AP_idx_com
    original_graph['UE', 'up', 'AP'].edge_index = edge_index
    
    edge_attr = original_graph.edge_attr_dict['UE', 'up', 'AP']
    edge_attr_com = complement_graph.edge_attr_dict['UE', 'up', 'AP']
    original_graph['UE', 'up', 'AP'].edge_attr = torch.cat([edge_attr, edge_attr_com], dim=0)
    
    
    return original_graph
