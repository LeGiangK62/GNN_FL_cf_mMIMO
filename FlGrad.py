import time
import os

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import scipy.io
import torch
import numpy as np

import scipy.io
from Utils.data_gen import build_cen_loader, build_decen_loader

from Models.GNN import APHetNet, APHetNetFL
from Utils.args import parse_args
from Utils.centralized_train import cen_eval, cen_train, cen_loss_function
from Utils.decentralized_train import FedAvg, FedAvgM, FedSoftMin

from Utils.fl_train import fl_train, get_global_info, server_return, fl_eval, server_return_GAP


SAVE_DIR = 'results'
MODEL_DIR = "results/models/"
EVAL_DIR = "results/eval/"
FIG_DIR = "results/figs"
TRAIN_DIR = "results/train/"

def init_folder():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
     
def lr_factor(round_idx, warmup_rounds, gamma):
    """Round-wise LR factor in (0, 1]. round_idx is 0-based."""
    if round_idx < warmup_rounds:
        # linear warmup
        return float(round_idx + 1) / warmup_rounds
    else:
        # slow exponential decay
        return gamma ** (round_idx - warmup_rounds)   


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = time.strftime('%y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    init_folder()
    
    args = parse_args()
    seed = args.seed
    
    # trainin param
    cen_pretrain = args.cen_pretrain
    
    
    # Model param
    hidden_channels = args.hidden_channels
    num_gnn_layers = args.num_gnn_layers
    
    # Centralized hyperparams
    num_epochs_cen = args.num_epochs_cen
    cen_lr = args.cen_lr
    
    # FL hyperparams
    num_rounds = args.num_rounds
    num_epochs = args.num_epochs
    eval_round = num_rounds//10
    lr = args.lr
    step_size = num_rounds//4
    gamma = args.gamma
    client_fraction = args.client_fraction
    num_global_ap = args.num_global_ap
    
    
    
    # training parameters
    num_train = args.num_train
    num_test = args.num_test
    num_eval = args.num_eval
    
    batch_size = args.batch_size
    
    ## Communciation params
    tau = args.tau
    num_antenna = args.num_antenna
    rho_p, rho_d = args.power_f, args.power_f
    num_clients = args.num_ap
    eval_round = args.eval_round if args.eval_round else args.num_rounds//10
    num_ap = args.num_ap
    num_ue = args.num_ue
    
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    file_name = f'dl_data_with_power_10000_{num_ue}_{num_ap}'
    mat_data = scipy.io.loadmat('Data/' + file_name + '.mat')
    beta_all = mat_data['betas']
    gamma_all = mat_data['Gammas']
    power_all = mat_data['power']/rho_d
    phi_all = mat_data['Phii_cf'].transpose(0, 2, 1)
    opt_train_rates = mat_data['R_cf_opt_min'][0]
    label_all = None
    
    perm = np.random.RandomState(seed).permutation(beta_all.shape[0])
    train_idx = perm[:num_train]
    test_idx  = perm[num_train: num_train + num_test]
    eval_idx  = perm[-num_eval:]
    
    ## FL Data
    train_data, train_loader = build_decen_loader(
        beta_all[train_idx],
        gamma_all[train_idx], 
        phi_all[train_idx],
        batch_size, seed=seed
    )
    test_data, test_loader = build_decen_loader(
        beta_all[test_idx], 
        gamma_all[test_idx],
        phi_all[test_idx], 
        batch_size, seed=seed
    )
    eval_data, eval_loader = build_decen_loader(
        beta_all[eval_idx], 
        gamma_all[eval_idx],
        phi_all[eval_idx], 
        num_eval, seed=seed
    )

    ## Centralized Data
    train_data_cen, train_loader_cen = build_cen_loader(
        beta_all[train_idx],
        gamma_all[train_idx], 
        phi_all[train_idx],
        batch_size, isShuffle=True
    )
    test_data_cen, test_loader_cen = build_cen_loader(
        beta_all[test_idx], 
        gamma_all[test_idx],
        phi_all[test_idx], 
        batch_size
    )
    eval_data_cen, eval_loader_cen = build_cen_loader(
        beta_all[eval_idx], 
        gamma_all[eval_idx],
        phi_all[eval_idx], 
        num_eval
    )
    
    ## Model Meta
    ap_dim = train_data_cen[0]['AP'].x.shape[1]
    ue_dim = train_data_cen[0]['UE'].x.shape[1]
    edge_dim = train_data_cen[0]['down'].edge_attr.shape[1]
    tt_meta = [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
    dim_dict = {
        'UE': ue_dim,
        'AP': ap_dim,
        'edge': edge_dim,        
    }
    
    
    ## Centralized Model
    cen_model = APHetNet(
        metadata=tt_meta,
        dim_dict=dim_dict,
        out_channels=hidden_channels,
        num_layers=num_gnn_layers,
        hid_layers=hidden_channels//2,
    ).to(device)
    torch.nn.utils.clip_grad_norm_(cen_model.parameters(), 1.0)
    # cen_optimizer = torch.optim.Adam(cen_model.parameters(), lr=cen_lr)
    cen_optimizer = torch.optim.AdamW(
        cen_model.parameters(), lr=cen_lr, weight_decay=1e-4
    )


    cen_scheduler = torch.optim.lr_scheduler.StepLR(
        cen_optimizer, step_size=num_epochs_cen//10, gamma=0.8
    )
    
    
    ## Centralized Training
    if cen_pretrain is not None:
        model_filename = f'{MODEL_DIR}/{cen_pretrain}.pth'
        cen_model.load_state_dict(torch.load(model_filename))
    else:
        start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n ===={start_time}==== Training Centralized GNN ... ")
        start_time = time.time()
        all_rate = []
        all_rate_test = []
        eval_epochs_cen = num_epochs_cen//10 if num_epochs_cen//10 else 1
        print(f'Training Centralized GNN for benchmark...')
        print(f'Optimal rate: train {np.mean(opt_train_rates[train_idx])}, test {np.mean(opt_train_rates[test_idx])}')
        for epoch in range(num_epochs_cen):
            cen_model.train()
            train_loss = cen_train(
                epoch/(2*num_epochs_cen//3),
                train_loader_cen, cen_model, cen_optimizer,
                tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
            )
            
            cen_model.eval()
            with torch.no_grad():
                train_eval = cen_eval(
                    train_loader_cen, cen_model,
                    tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
                )   
                test_eval = cen_eval(
                    test_loader_cen, cen_model,
                    tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna
                )  
            all_rate.append(train_eval)    
            all_rate_test.append(test_eval)    
            cen_scheduler.step()
            if epoch%eval_epochs_cen==0:
                print(
                    f"Epoch {epoch+1:03d}/{num_epochs_cen} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Rate: {train_eval:.4f} | "
                    f"Test Rate: {test_eval:.4f} "
                )
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {timedelta(seconds=execution_time)}")
        
        ## Centralized Training Results

        plt.figure(figsize=(6,4), dpi=180)
        plt.plot(all_rate, label='Training Rate', linewidth=2)
        plt.plot(all_rate_test, label='Testing Rate', linewidth=2)
        plt.axhline(y=np.mean(opt_train_rates[train_idx]), linewidth=2, color='r', linestyle='--', label='Training Optimal')
        plt.axhline(y=np.mean(opt_train_rates[test_idx]), linewidth=2, color='b', linestyle='--', label='Testing Optimal')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title('Centralized GNN Training Rate Curve', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        figure_name = f'{timestamp}_cen'
        save_path = TRAIN_DIR + f'{figure_name}.png' 
        plt.savefig(save_path, dpi=300)  
        
        model_filename = f'{MODEL_DIR}/{timestamp}_cen.pth'
        torch.save(cen_model.state_dict(), model_filename)
        print(f'Save centralized GNN to {model_filename}.')
    
    # FL-GNN

    ## Model
    ap_dim = train_data[0][0]['AP'].x.shape[1]
    ue_dim = train_data[0][0]['UE'].x.shape[1]  # Original UE dim (tau)
    edge_dim = train_data[0][0]['down'].edge_attr.shape[1]

    tt_meta = [('UE', 'up', 'AP'), ('AP', 'down', 'UE')]
    dim_dict = {
        'UE': ue_dim,  # Augmented UE dimension
        'AP': ap_dim,
        'edge': edge_dim,
    }

    # FL model expects augmented UE features: [tau + 3]
    aug_ue_dim = 4

    # Initialize the models, optimizers, and schedulers for clients
    global_model = APHetNetFL(
        metadata=tt_meta,
        dim_dict=dim_dict,
        out_channels=hidden_channels,
        aug_feat_dim=aug_ue_dim,  # DS, PC, UI, rate_without_me + 3?
        num_layers=num_gnn_layers,
        hid_layers=hidden_channels//2,
        isDecentralized=False  # Use same architecture as centralized
    ).to(device)

    global_optimizer = torch.optim.AdamW(global_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    global_scheduler = torch.optim.lr_scheduler.StepLR(
        global_optimizer, step_size=num_rounds//3, gamma=0.5
    )

    # Optional: Warm-start from centralized model (transfer learning)
    # if cen_pretrain is not None:
    #     # Load centralized weights but skip UE input layer due to dimension mismatch
    #     from Utils.decentralized_train import load_state_dict_skipping
    #     load_state_dict_skipping(
    #         global_model,
    #         cen_model.state_dict(),
    #         exclude_contains=('convs.0.msg.UE',)  # Skip first UE message layer
    #     )

    local_models, optimizers, schedulers = [], [], []
    for _ in range(num_clients):
        model = APHetNetFL(
            metadata=tt_meta,
            dim_dict=dim_dict,
            out_channels=hidden_channels,
            aug_feat_dim=aug_ue_dim,  # DS, PC, UI, rate_without_me + 3?
            num_layers=num_gnn_layers,
            hid_layers=hidden_channels//2,
            isDecentralized=False
        ).to(device)
        model.load_state_dict(global_model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
        local_models.append(model)
        optimizers.append(optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_rounds, eta_min=1e-5)
        schedulers.append(scheduler)
    
    from Utils.fl_train import FedAvg, FedAdam, FedGM, sample_clients
    if args.fl_scheme == 'fedavg':
        fed = FedAvg(client_fraction=client_fraction)
    elif args.fl_scheme == 'fedadam':
        fed = FedAdam(client_fraction=client_fraction)
    elif args.fl_scheme == 'fedgm':
        fed = FedGM(client_fraction=client_fraction, server_lr=args.server_lr)
    else:
        raise ValueError(f'Handling global update in {args.fl_scheme.upper()} is not supported!')
    
    ## Training FL-GNN
    
    fl_all_rate = []
    fl_all_rate_test = []

    ### Training loop
    if args.fl_pretrain is None:
        start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n ===={start_time}==== Training FL-GNN using {args.fl_scheme.upper()} ... ")
        start_time = time.time()
        print(f'Optimal rate: train {np.mean(opt_train_rates[train_idx])}, test {np.mean(opt_train_rates[test_idx])}')
        for round in range(num_rounds):
            ## Send the information to the global server
            ## DS, PC, UI - for rate calculation
            ## AP, UE, others, to augmented graph
            send_to_server = get_global_info(
                train_loader, local_models,
                tau=tau, rho_p=rho_p, rho_d=rho_d,
                num_antenna=num_antenna
            )            
            # response_from_server = server_return(train_loader, send_to_server, num_antenna=num_antenna)
            response_from_server = server_return_GAP(train_loader, send_to_server, num_antenna=num_antenna)

            local_gradients = []
            local_rates = []
            # total_rate = 0.0
            # total_loss = 0.0
            selected_clients = sample_clients(client_fraction, num_clients)

            for client_idx, (model, opt, client_data_tuple) in enumerate(zip(local_models, optimizers, zip(*response_from_server))):
                batches = [item['loader'] for item in client_data_tuple]
                batch_rate = [item['rate_pack'] for item in client_data_tuple]

                if client_idx not in selected_clients:
                    continue
                
                # client_loss_sum = 0.0
                # client_rate_sum = 0.0
                for _ in range(num_epochs):
                    train_loss, train_min = fl_train(
                        batches,
                        batch_rate,
                        model, opt,
                        tau=tau, rho_p=rho_p, rho_d=rho_d,
                        num_antenna=num_antenna,
                    )
                #     client_loss_sum += train_loss
                #     client_rate_sum += train_min
                # train_loss = client_loss_sum / num_epochs
                # train_min = client_rate_sum / num_epochs

                # total_loss += train_loss
                # total_rate += train_min

            # avg_loss = total_loss / len(selected_clients)
            # avg_rate = total_rate / len(selected_clients)

            global_weights = fed.aggregate(global_model, local_models, selected_clients)

            global_model.load_state_dict(global_weights)
            for model in local_models:
                model.load_state_dict(global_model.state_dict())


            if round % eval_round == 0:
                ## Evaluate on the client local data (both train and test)
                total_train_rate = fl_eval(
                    train_loader, local_models,
                    tau, rho_p, rho_d, num_antenna
                )
                total_train_rate = torch.mean(total_train_rate).cpu().detach()

                total_eval_rate = fl_eval(
                    test_loader, local_models,
                    tau, rho_p, rho_d, num_antenna
                )
                total_eval_rate = torch.mean(total_eval_rate).cpu().detach()

                fl_all_rate.append(total_train_rate)
                fl_all_rate_test.append(total_eval_rate)

                print(f"Round {round+1:03d}/{num_rounds}: "
                    # f"Avg Loss = {avg_loss:.4f} | "
                    # f"Avg Min Rate = {avg_rate:.4f} | "
                    f"Avg Train Rate = {total_train_rate:.4f} | "
                    f"Avg Eval Rate = {total_eval_rate:.4f} | "
                )
            for client_idx in selected_clients:
                schedulers[client_idx].step()
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution Time: {timedelta(seconds=execution_time)}")

        plt.figure(figsize=(6,4), dpi=180)
        x_axis = [i * eval_round for i in range(len(fl_all_rate))]
        plt.plot(x_axis, fl_all_rate, label='Training Rate', linewidth=2)
        plt.plot(x_axis, fl_all_rate_test, label='Testing Rate', linewidth=2)
        plt.axhline(y=np.mean(opt_train_rates[train_idx]), linewidth=2, color='r', linestyle='--', label='Training Optimal')
        plt.axhline(y=np.mean(opt_train_rates[test_idx]), linewidth=2, color='b', linestyle='--', label='Testing Optimal')
        plt.xlabel('Rounds', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title(f'{args.fl_scheme.upper()}  GNN Training Rate Curve - {args.lr}_{args.num_rounds}', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        figure_name = f'{timestamp}_fl'
        save_path = TRAIN_DIR + f'{figure_name}.png' 
        plt.savefig(save_path, dpi=300) 

        fl_model_filename = f'{MODEL_DIR}/{timestamp}_fl.pth'
        torch.save(global_model.state_dict(), fl_model_filename)
        print(f'Save FL GNN to {fl_model_filename}.')
    else:
        fl_model_filename = f'{MODEL_DIR}/{args.fl_pretrain}.pth'
        global_model.load_state_dict(torch.load(fl_model_filename))


    # Evaluation - CDF
    if args.eval_plot:
        print(f'Evaluation' + '='*20)
        # Eval
        cen_model.eval()
        for batch in eval_loader_cen:
            batch = batch.to(device)
            num_graph = batch.num_graphs
            x_dict, edge_dict, edge_index = cen_model(batch)
            
            gnn_rates, all_one_rates = cen_loss_function(
                batch, x_dict, edge_dict,
                tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna, 
                eval_mode=True
            )

        for local_model in local_models:
            local_model.eval()


        fl_gnn_rates = fl_eval(
            eval_loader, local_models,
            tau, rho_p, rho_d, num_antenna
        )

            
        fl_gnn_rates = fl_gnn_rates.detach().cpu().numpy() 
        gnn_rates = gnn_rates.detach().cpu().numpy() 
        all_one_rates = all_one_rates.detach().cpu().numpy()
        opt_rates = opt_train_rates[eval_idx] 
        max_value = np.ceil(max(np.max(all_one_rates), np.max(fl_gnn_rates), np.max(gnn_rates), np.max(opt_rates))*100)/100
        
        min_rate, max_rate = 0, max_value
        # y_axis = np.arange(0, 1.0, 1/202)
        y_axis = np.linspace(0, 1, num_eval+2)
        gnn_rates.sort();  opt_rates.sort(); all_one_rates.sort(); fl_gnn_rates.sort()
        gnn_rates = np.insert(gnn_rates, 0, min_rate); gnn_rates = np.insert(gnn_rates,num_eval+1,max_rate)
        fl_gnn_rates = np.insert(fl_gnn_rates, 0, min_rate); fl_gnn_rates = np.insert(fl_gnn_rates,num_eval+1,max_rate)
        all_one_rates = np.insert(all_one_rates, 0, min_rate); all_one_rates = np.insert(all_one_rates,num_eval+1,max_rate)
        opt_rates = np.insert(opt_rates, 0, min_rate); opt_rates = np.insert(opt_rates,num_eval+1,max_rate)

        plt.figure(figsize=(6,4), dpi=180)
        plt.plot(all_one_rates, y_axis, label = 'Maximum Power', linewidth=2)
        plt.plot(gnn_rates, y_axis, label = 'Centralized GNN', linewidth=2)
        plt.plot(fl_gnn_rates, y_axis, label = 'FL GNN', linewidth=2)
        plt.plot(opt_rates, y_axis, label = 'Optimal', linewidth=2)
        plt.xlabel('Minimum rate [bps/Hz]', {'fontsize':16})
        plt.ylabel('Empirical CDF', {'fontsize':16})
        plt.legend(fontsize = 14)
        plt.grid()
        
        figure_name = f'{timestamp}_eval'
        eval_path = EVAL_DIR + f'/{figure_name}.png' 
        plt.savefig(eval_path, dpi=300, bbox_inches='tight')
        print(f'Save Evaluation figure to {eval_path}.')


    # print('Current best: bottleneck_indicator, contribution_ratio,  interference_share,  global_sinr')
    # print("Old: GAP -> UE -> GAP")
    # print("New: UE -> GAP -> UE")
    print("Pre UE -> Mix  -> GAP -> Mix")
    print("1 layer fixed")

    # old best: DS,PC, UI in UE, no GAP, GAP-AP enhanced?
    
