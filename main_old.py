import time
from datetime import timedelta
import os
import argparse
import copy
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import scipy.io
from Utils.centralized_train import cen_eval, cen_train, cen_loss_function
from Utils.decentralized_train import (
    get_global_info, distribute_global_info, average_weights, fl_train, fl_eval_rate, load_state_dict_skipping,
    FedAdam, FedAvg, FedAvgGradMatch, FedProx ,
    save_models, load_models
)
from Utils.data_gen import Generate_Input, create_graph, build_loader
from Models.GNN import APHetNet
from torch_geometric.loader import DataLoader
from Utils.comm import variance_calculate, rate_calculation, component_calculate, rate_from_component
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Script")
    
    parser.add_argument('--pre_train', type=str, default=None, help="Path to pre trained model (insinde results/models/ folder, without '.pt')")
    parser.add_argument("--eval_same_data", action="store_true", default=True, help="Eval on the same with training data.")
    
    # System Parameters
    parser.add_argument('--num_ap', type=int, default=30, help="Number of access points")
    parser.add_argument('--num_ue', type=int, default=6, help="Number of user equipments")
    parser.add_argument('--tau', type=int, default=20, help="Pilot length")
    parser.add_argument('--power_f', type=float, default=0.2, help="Transmit power threshold")
    parser.add_argument('--D', type=float, default=1, help="Area diameters (km)")
    parser.add_argument('--num_antenna', type=int, default=1, help="Number of antennas per AP")
    
    # Hyperparameters
    parser.add_argument('--norm_scheme', type=str, choices=['z_score', 'no', 'min_max'], default='z_score', help="Data normalization scheme")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of local training epochs")
    parser.add_argument('--num_rounds', type=int, default=150, help="Number of global training rounds")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate")
    parser.add_argument('--step_size', type=int, default=3, help="Step size for scheduler (rounds)")
    parser.add_argument('--gamma', type=float, default=0.1, help="Gamma for scheduler")
    parser.add_argument('--eval_round', type=int, default=None, help="Evaluation round frequency")
    
    parser.add_argument('--num_train', type=int, default=500, help="Number of training samples")
    parser.add_argument('--num_test', type=int, default=200, help="Number of testing samples")
    parser.add_argument('--num_eval', type=int, default=200, help="Number of evaluation samples")
    
    
    # Centralized hyperparameters
    parser.add_argument('--cen_lr', type=float, default=5e-3, help="Centralized learning rate")
    parser.add_argument('--num_epochs_cen', type=int, default=1000, help="Number of Centralized training epochs")
    
    
    # FL Algorithm Parameters
    parser.add_argument('--fl_scheme', type=str, choices=['fedavg', 'fedavg_gm', 'fedprox', 'fedadam'], default='fedavg', help="Federated Learning scheme")
    parser.add_argument('--client_fraction', type=float, default=1.0, help="Fraction of clients to be selected per round")
    parser.add_argument('--mu', type=float, default=0.1, help="Weight for gradient matching (if using FedAvgGradMatch)")
    parser.add_argument('--server_lr',     type=float, default=1e-2,  help="Server LR for FedAdam")
    parser.add_argument('--server_beta1',  type=float, default=0.9,   help="Server beta1 for FedAdam")
    parser.add_argument('--server_beta2',  type=float, default=0.99,  help="Server beta2 for FedAdam")
    parser.add_argument('--server_eps',    type=float, default=1e-8,  help="Server eps for FedAdam")
    # Model and Data Parameters
    parser.add_argument('--is_edge_update', type=bool, default=True, help="Whether to perform edge update")
    parser.add_argument('--hidden_channels', type=int, default=32, help="Number of hidden channels for GNN")
    parser.add_argument('--num_gnn_layers', type=int, default=4, help="Number of GNN layers")
    
    # Seed for reproducibility
    parser.add_argument('--seed', type=int, default=1712, help="Random seed")

    return parser.parse_args()


def train(timestamp):
    save_dir = "results/models"
    
    
    args = parse_args()
    
    
    excl_contents = (
        'convs_per',
        'running_mean', 'running_var', 'num_batches_tracked'
    )
    # excl_contents = ()
    
    rho_p, rho_d = args.power_f, args.power_f
    num_clients = args.num_ap
    
    eval_round = args.eval_round if args.eval_round else args.num_rounds//10
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    file_name = f'dl_data_2000_{args.num_ue}_{args.num_ap}'
    mat_data = scipy.io.loadmat('Data/' + file_name + '.mat')
    beta_all = mat_data['betas']
    phi_all = mat_data['Phii_cf'].transpose(0, 2, 1)
    opt_rates = mat_data['R_cf_opt_min'][0]
    if args.norm_scheme == 'z_score':
        beta_mean = np.mean(beta_all)
        beta_std = np.std(beta_all)
        beta_all = (beta_all - beta_mean) / (beta_std)
    elif args.norm_scheme == 'no':
        beta_mean = 0
        beta_std = 1
        beta_all = (beta_all - beta_mean) / (beta_std)
    elif args.norm_scheme == 'min_max':
        beta_mean = np.min(beta_all)
        beta_std = np.max(beta_all) - np.min(beta_all)
        beta_all = (beta_all - beta_mean) / (beta_std)
    else: 
        raise ValueError(f'{args.norm_scheme} is not supported.')

    # Define data splits
    perm = np.random.RandomState(args.seed).permutation(beta_all.shape[0])
    train_idx = perm[:args.num_train]
    test_idx  = perm[args.num_train: args.num_train + args.num_test]
    Beta_all, Phi_all = beta_all[train_idx], phi_all[train_idx]
    train_data = create_graph(Beta_all, Phi_all, beta_mean, beta_std, 'het')
    train_loader = build_loader(train_data, args.batch_size, seed=args.seed, drop_last=True)
    
    Beta_test, Phi_test = beta_all[test_idx], phi_all[test_idx]
    test_data = create_graph(Beta_test, Phi_test, beta_mean, beta_std, 'het')
    test_loader = build_loader(test_data, args.batch_size, seed=args.seed, drop_last=True)

    print(f'Avg train opt: {np.mean(opt_rates[train_idx]):0.4f}')
    print(f'Avg eval opt: {np.mean(opt_rates[test_idx]):0.4f}')
    
    # Model Meta
    ap_dim = train_data[0][0]['AP'].x.shape[1]
    ue_dim = train_data[0][0]['UE'].x.shape[1]
    edge_dim = train_data[0][0]['down'].edge_attr.shape[1]
    tt_meta = [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
    dim_dict = {
        'UE': ue_dim,
        'AP': ap_dim,
        'edge': edge_dim,
    }


    # Initialize the models, optimizers, and schedulers for clients
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model = APHetNet(
        metadata=tt_meta,
        dim_dict=dim_dict,
        out_channels=args.hidden_channels,
        num_layers=args.num_gnn_layers,
        hid_layers=args.hidden_channels,
        edge_conv=args.is_edge_update
    ).to(device)

    local_models, optimizers, schedulers = [], [], []
    for _ in range(num_clients):
        model = APHetNet(
            metadata=tt_meta,
            dim_dict=dim_dict,
            out_channels=args.hidden_channels,
            num_layers=args.num_gnn_layers,
            hid_layers=args.hidden_channels,
            edge_conv=args.is_edge_update
        ).to(device)
        model.load_state_dict(global_model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        local_models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # Initialize federated algorithm
    if args.fl_scheme == 'fedavg_gm':
        fed = FedAvgGradMatch(client_fraction=args.client_fraction, mu=args.mu, seed=args.seed)
    elif args.fl_scheme == 'fedavg':
        fed = FedAvg(client_fraction=args.client_fraction)
    elif args.fl_scheme == 'fedprox':
        fed = FedProx(client_fraction=args.client_fraction, mu=args.mu, seed=args.seed)
    elif args.fl_scheme == 'fedadam':
        fed = FedAdam(global_model,
                    client_fraction=args.client_fraction,
                    seed=args.seed,
                    lr=args.server_lr,
                    beta1=args.server_beta1,
                    beta2=args.server_beta2,
                    eps=args.server_eps)
    else: 
        raise ValueError(f'{args.fl_scheme.upper()} is not supported.')
    
    
    if args.pre_train:
        model_filename = f'{save_dir}/{args.pre_train}.pt'
        print(f'Loading pre-trained model from {model_filename}...')
        state = torch.load(model_filename, map_location="cpu")
        global_model.load_state_dict(state['global'])
        for i, model in enumerate(local_models):
            model.load_state_dict(global_model.state_dict())
            # load_state_dict_skipping(
            #     m, global_model.state_dict(),
            #     exclude_contains=("convs_per","running_mean","running_var","num_batches_tracked")
            # )
            model.load_state_dict(state["locals"][i], strict=False)
        # if "locals" in state and i in state["locals"]:
        #     m.load_state_dict(state["locals"][i], strict=False)
        # load_models(model_filename, global_model, local_models)
    else:
        # Training loop
        print(
            f'======= Training model with lr:{args.lr}, {args.num_ap} APs and {args.num_ue} UEs... \n'
            f'======= {timestamp} ======='
        )
        
        start_time = time.time()
        for round in range(args.num_rounds):
            total_train_rate = fl_eval_rate(
                train_loader, local_models,
                tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, 
                num_antenna=args.num_antenna, isEdgeUpd=args.is_edge_update
            )
            total_eval_rate = fl_eval_rate(
                test_loader, local_models,
                tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, 
                num_antenna=args.num_antenna, isEdgeUpd=args.is_edge_update    
            )

            # Exchange global information
            send_to_server = get_global_info(
                train_loader, local_models, optimizers, 
                tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, 
                num_antenna=args.num_antenna, isEdgeUpd=args.is_edge_update
            )
            response_all = distribute_global_info(send_to_server)

            # Train local models
            local_weights = []
            local_gradients = []
            total_loss = 0.0
            if args.fl_scheme in ['fedavg_gm', 'fedavg', 'fedprox', 'fedadam']:
                selected_clients = fed.sample_clients(num_clients)
            else: 
                raise ValueError(f'Handling sample in {args.fl_scheme.upper()} is not supported.')
            
            for client_idx, (model, opt, sch, batches, responses_ap) in enumerate(zip(local_models, optimizers, schedulers, train_loader, response_all)):
                if client_idx not in selected_clients:
                    local_weights.append(copy.deepcopy(model.state_dict()))
                    local_gradients.append(None)
                    continue
                for _ in range(args.num_epochs):
                    train_loss, local_gradient = fl_train(
                        batches, responses_ap, model, opt, 
                        tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, 
                        num_antenna=args.num_antenna, isEdgeUpd=args.is_edge_update    
                    )
                    # optimizer.step()
                    # sch.step()
                local_weights.append(copy.deepcopy(model.state_dict()))
                local_gradients.append(local_gradient)
                total_loss += train_loss
                
            
            for sch in schedulers:
                sch.step()
                

            avg_loss = total_loss / len(selected_clients)
            if args.fl_scheme in ['fedavg_gm']:
                global_weights = fed.aggregate(local_weights, selected_clients, local_gradients)
            elif args.fl_scheme in ['fedavg']:
                global_weights = fed.aggregate(local_weights, selected_clients)
            elif args.fl_scheme == 'fedprox':
                global_weights = fed.aggregate(local_weights, selected_clients, global_model)
            elif args.fl_scheme == 'fedadam':
                sel_weights = [local_weights[i] for i in selected_clients]
                global_weights = fed.aggregate(global_model, sel_weights)
            else: 
                raise ValueError(f'Handling global update in {args.fl_scheme.upper()} is not supported.')
            
            global_model.load_state_dict(global_weights)
            # Update global models
            for model in local_models:
                model.load_state_dict(global_weights)
                # load_state_dict_skipping(
                #     model, global_model.state_dict(), 
                #     exclude_contains=excl_contents
                # )
            
            if round % eval_round == 0:
                # lrs = [opt.param_groups[0]['lr'] for opt in optimizers]
                # print(f"[Round {round+1}] client LR: mean={np.mean(lrs):.3e}  min={np.min(lrs):.3e}  max={np.max(lrs):.3e}")
                print(f"Round {round+1:03d}/{args.num_rounds}: "
                    f"Avg Training Loss = {avg_loss:.4f} | "
                    f"Avg Training Rate = {total_train_rate:.4f} | "
                    f"Avg Eval Rate = {total_eval_rate:.4f} | "
                )
        end_time = time.time()
        execution_time = end_time - start_time
        # print(f"Execution Time: {execution_time:.4f} seconds")
        print(f"Execution Time: {timedelta(seconds=execution_time)}")
        
        model_filename = f'{save_dir}/{timestamp}_{args.fl_scheme}_{args.lr}_{args.num_rounds}.pt'
        # Save the model's state_dict
        # torch.save(global_model.state_dict(), model_filename)
        save_models(args, global_model, local_models, model_filename)
        print(f'Save file to {model_filename}.')
            
    ## Centralized GNN training   
    train_data_cen = create_graph(Beta_all, Phi_all, beta_mean, beta_std, 'het', isDecentralized=False)
    train_loader_cen = DataLoader(train_data_cen, batch_size=args.batch_size, shuffle=True)
    test_data_cen = create_graph(Beta_test, Phi_test, beta_mean, beta_std, 'het', isDecentralized=False)
    test_loader_cen = DataLoader(test_data_cen, batch_size=args.batch_size, shuffle=False)

    cen_model = APHetNet(
        metadata=tt_meta,
        dim_dict=dim_dict,
        out_channels=args.hidden_channels,
        num_layers=args.num_gnn_layers,
        hid_layers=args.hidden_channels,
        edge_conv=True
    ).to(device)
    cen_optimizer = torch.optim.Adam(cen_model.parameters(), lr=args.cen_lr)
    cen_scheduler = StepLR(cen_optimizer, step_size=10, gamma=0.9)

    eval_epochs_cen = args.num_epochs_cen//10 if args.num_epochs_cen//10 else 1
    print(f'Training Centralized GNN for benchmark...')
    for epoch in range(args.num_epochs_cen):
        cen_model.eval()
        with torch.no_grad():
            train_eval = cen_eval(
                train_loader_cen, cen_model,
                tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, num_antenna=args.num_antenna
            )
            test_eval = cen_eval(
                test_loader_cen, cen_model,
                tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, num_antenna=args.num_antenna
            )
            
        cen_model.train()
        train_loss, _ = cen_train(
            train_loader_cen, cen_model, cen_optimizer,
            tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, num_antenna=args.num_antenna
        )
        cen_scheduler.step()
        if epoch%eval_epochs_cen==0:
            print(
                f"Epoch {epoch+1:03d}/{args.num_epochs_cen} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Rate: {train_eval:.4f} | "
                f"Test Rate: {test_eval:.4f} "
            )
        

    return local_models, cen_model, perm

def eval(local_models, cen_model, timestamp, perm):
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    rho_p, rho_d = args.power_f, args.power_f
    num_clients = args.num_ap
    
    eval_round = args.eval_round if args.eval_round else args.num_rounds//10
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Load data
    if args.eval_same_data:
        print('Same data eval...')
        eval_idx = perm[:args.num_eval]
        # file_name = f'cf_data_2000_{args.num_ue}_{args.num_ap}'
        file_name = f'dl_data_2000_{args.num_ue}_{args.num_ap}'
        eval_mat = scipy.io.loadmat('Data/' + file_name + '.mat')
        Beta_eval = eval_mat['betas']
        Phi_eval = eval_mat['Phii_cf'].transpose(0,2,1)
        
        Beta_eval, Phi_eval = Beta_eval[eval_idx], Phi_eval[eval_idx]

        opt_rates = eval_mat['R_cf_opt_min'][:, eval_idx]
    else:
        print('Different data eval...')
        file_name = f'eval_data_{args.num_eval}_{args.num_ue}_{args.num_ap}'
        # file_name = 'cf_data_1000_6_30'
    # file_name = f'eval_data_{args.num_eval}_{args.num_ue}_{args.num_ap}'
        eval_mat = scipy.io.loadmat('Data/' + file_name + '.mat')
        Beta_eval = eval_mat['betas'][:args.num_eval]
        Phi_eval = eval_mat['Phii_cf'][:args.num_eval].transpose(0,2,1)

        opt_rates = eval_mat['R_cf_opt_min'][:, :args.num_eval]
    
    
    if args.norm_scheme == 'z_score':
        beta_mean_eval = np.mean(Beta_eval)
        beta_std_eval = np.std(Beta_eval)
        Beta_eval = (Beta_eval - beta_mean_eval) / (beta_std_eval)
    elif args.norm_scheme == 'no':
        beta_mean_eval = 0
        beta_std_eval = 1
        Beta_eval = (Beta_eval - beta_mean_eval) / (beta_std_eval)
    elif args.norm_scheme == 'min_max':
        beta_mean_eval = np.min(Beta_eval)
        beta_std_eval = np.max(Beta_eval) - np.min(Beta_eval)
        Beta_eval = (Beta_eval - beta_mean_eval) / (beta_std_eval)
    else: 
        raise ValueError(f'{args.norm_scheme} is not supported.')

    # Define data splits
    eval_data = create_graph(Beta_eval, Phi_eval, beta_mean_eval, beta_std_eval,  'het')
    eval_loader = build_loader(eval_data, args.num_eval, seed=1712, drop_last=False)
    
    eval_data_cen = create_graph(Beta_eval, Phi_eval, beta_mean_eval, beta_std_eval, 'het', isDecentralized=False)
    eval_loader_cen = DataLoader(eval_data_cen, batch_size=args.num_eval, shuffle=False)  
    
    
    fl_rates = fl_eval_rate(
        eval_loader, local_models,
        tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, 
        num_antenna=args.num_antenna, isEdgeUpd=args.is_edge_update,
        eval_mode=True
    )
    
    
    cen_model.eval()
    for batch in eval_loader_cen:
        batch = batch.to(device)
        num_graph = batch.num_graphs
        x_dict, edge_dict, edge_index = cen_model(batch)
        
        gnn_rates, all_one_rates = cen_loss_function(
            batch, x_dict, edge_dict,
            tau=args.tau, rho_p=args.power_f, rho_d=args.power_f, num_antenna=args.num_antenna, 
            eval_mode=True
        )
        if device.type == 'cuda':
            gnn_rates = gnn_rates.detach().cpu().numpy() 
            all_one_rates = all_one_rates.detach().cpu().numpy() 
            fl_rates = fl_rates.detach().cpu().numpy()
        else:
            # raise ValueError(f'{device} handling is not supported now!')
            gnn_rates = gnn_rates.detach().cpu().numpy() 
            all_one_rates = all_one_rates.detach().cpu().numpy() 
            fl_rates = fl_rates.detach().cpu().numpy()
        
    print(f'Avg rate FL: {np.mean(fl_rates):0.4f}')
    print(f'Avg rate centralized: {np.mean(gnn_rates):0.4f}')
    print(f'Avg rate opt: {np.mean(opt_rates):0.4f}')
    
    
    num_eval = args.num_eval
    min_rate, max_rate = 0, 2
    # y_axis = np.arange(0, 1.0, 1/202)
    y_axis = np.linspace(0, 1, num_eval+2)
    gnn_rates.sort();  opt_rates.sort(); all_one_rates.sort(); fl_rates.sort()
    gnn_rates = np.insert(gnn_rates, 0, min_rate); gnn_rates = np.insert(gnn_rates,num_eval+1,max_rate)
    fl_rates = np.insert(fl_rates, 0, min_rate); fl_rates = np.insert(fl_rates,num_eval+1,max_rate)
    all_one_rates = np.insert(all_one_rates, 0, min_rate); all_one_rates = np.insert(all_one_rates,num_eval+1,max_rate)
    opt_rates = np.insert(opt_rates, 0, min_rate); opt_rates = np.insert(opt_rates,num_eval+1,max_rate)
            
    plt.plot(all_one_rates, y_axis, label = 'Maximum Power')
    plt.plot(fl_rates, y_axis, label = 'GNN-FL')
    plt.plot(gnn_rates, y_axis, label = 'Centralized GNN')
    plt.plot(opt_rates, y_axis, label = 'Optimal')
    plt.xlabel('Minimum rate [bps/Hz]', {'fontsize':16})
    plt.ylabel('Empirical CDF', {'fontsize':16})
    plt.legend(fontsize = 12)
    plt.grid()
    
    figure_name = f'{timestamp}_{args.fl_scheme}_{args.lr}_{args.num_rounds}'
    save_path = f'results/figs/{figure_name}.png' 
    plt.savefig(save_path, dpi=300)  

    
    
if __name__ == '__main__':

    
        
    save_dir = "results/models/"
    os.makedirs(save_dir, exist_ok=True)
    save_dir = "results/figs/"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    
    
    local_models, cen_model, perm = train(timestamp)
    
    eval(local_models, cen_model, timestamp, perm)
    
    
    
    
    