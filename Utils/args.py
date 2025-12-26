import argparse


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
    parser.add_argument('--lr', type=float, default=5e-4, help="FL Learning rate")
    parser.add_argument('--step_size', type=int, default=3, help="Step size for scheduler (rounds)")
    parser.add_argument('--gamma', type=float, default=0.1, help="Gamma for FL scheduler")
    parser.add_argument('--eval_round', type=int, default=None, help="Evaluation round frequency")
    
    parser.add_argument('--num_train', type=int, default=500, help="Number of training samples")
    parser.add_argument('--num_test', type=int, default=200, help="Number of testing samples")
    parser.add_argument('--num_eval', type=int, default=200, help="Number of evaluation samples")
    
    
    # Centralized hyperparameters
    parser.add_argument('--cen_lr', type=float, default=5e-3, help="Centralized learning rate")
    parser.add_argument('--num_epochs_cen', type=int, default=1000, help="Number of Centralized training epochs")
    parser.add_argument('--cen_pretrain', type=str, default=None, help="Name of model to load directly without training")
    
    
    # FL Algorithm Parameters
    parser.add_argument('--fl_scheme', type=str, choices=['fedavg', 'fedavg_gm', 'fedprox', 'fedadam'], default='fedavg', help="Federated Learning scheme")
    parser.add_argument('--client_fraction', type=float, default=1.0, help="Fraction of clients to be selected per round")
    parser.add_argument('--num_global_ap', type=int, default=1, help="Number of Global AP for knowledge graph")
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