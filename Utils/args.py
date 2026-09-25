import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Script")
    
    parser.add_argument("--no_save", action="store_true", default=False, help="No saving anything.")
    parser.add_argument("--param_free", action="store_true", default=False, help="Server no uses params for KG learning.")
    parser.add_argument("--ctde", action="store_true", default=False, help="Train on the global coherent sum rate (centralised training, decentralised execution). Off = purely-local loss.")
    parser.add_argument("--no_kg", action="store_true", default=False, help="Disable the server KG entirely: clients run their local AP<->UE graph with no KG injection (ablation baseline).")
    parser.add_argument('--lam', type=float, default=0.2, help="Weight of the local term when --ctde is on: L = -global_rate + lam*local_loss.")
    parser.add_argument('--filetype', type=str, choices=['png', 'pdf'], default='png', help="Result files type (pdf or png - default)")
    parser.add_argument('--pre_train', type=str, default=None, help="Path to pre trained model (insinde .results/models/ folder, without '.pt')")
    parser.add_argument("--eval_same_data", action="store_true", default=False, help="Eval on the same with training data.")
    parser.add_argument("--eval_plot", action="store_true", default=True, help="Eval Visualization (CDF)")
    parser.add_argument("--latex_table", action="store_true", default=False, help="Latex-ready table")
    
    # System Parameters
    parser.add_argument('--comm_rounds', type=int, default=1, help="Number of Comm rounds")
    parser.add_argument('--num_ap', type=int, default=30, help="Number of access points")
    parser.add_argument('--num_ue', type=int, default=6, help="Number of user equipments")
    parser.add_argument('--tau', type=int, default=20, help="Pilot length")
    parser.add_argument('--power_f', type=float, default=0.2, help="Transmit power threshold")
    parser.add_argument('--D', type=float, default=1, help="Area diameters (km)")
    parser.add_argument('--num_antenna', type=int, default=1, help="Number of antennas per AP")

    # ISAC parameters
    parser.add_argument('--num_sr', type=int, default=2, help="Number of sensing receivers")
    parser.add_argument('--num_tar', type=int, default=1, help="Number of sensing targets")
    parser.add_argument('--nu', type=float, default=1, help="Sensing resolution (m2)")
    parser.add_argument('--crlb_lambda', type=float, default=1.0,
                        help="Weight of the block-coordinate global CRLB surrogate in each client loss")
    parser.add_argument('--crlb_gamma_weighted', action='store_true', default=False,
                        help="Use Eq. (7) sensing power sum_k Gamma_mk P_mk in "
                             "the CRLB training surrogate. Off preserves the "
                             "published sum_k P_mk path.")
    parser.add_argument('--crlb_ratio_hinge', action='store_true', default=False,
                        help="Penalize relu(sigma2_xy - nu) instead of the "
                             "cleared-denominator CRLB expression. Off preserves "
                             "the published surrogate.")

    
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
    
    # FL hyperparameters
    parser.add_argument('--fl_pretrain', type=str, default=None, help="Name of FL model to load directly without training")
    parser.add_argument('--noKG_pretrain', type=str, default=None, help="Name of no KQ FL model to load directly without training")
    

    # Centralized hyperparameters
    parser.add_argument('--cen_lr', type=float, default=5e-3, help="Centralized learning rate")
    parser.add_argument('--num_epochs_cen', type=int, default=10, help="Number of Centralized training epochs")
    parser.add_argument('--cen_pretrain', type=str, default=None, help="Name of model to load directly without training")
    parser.add_argument('--cen_hidden_channels', type=int, default=32 , help="Number of hidden channels for centralized GNN")
    parser.add_argument('--cen_num_gnn_layers', type=int, default=1, help="Number of centralized GNN layers")
    
    
    # FL Algorithm Parameters
    parser.add_argument('--fl_scheme', type=str, choices=['fedavg', 'fedadam', 'fedgm', 'fedprox', 'scaffold'], default='fedavg', help="Federated Learning scheme")
    parser.add_argument('--client_fraction', type=float, default=1.0, help="Fraction of clients to be selected per round")
    parser.add_argument('--num_global_ap', type=int, default=1, help="Number of Global AP for knowledge graph")
    parser.add_argument('--mu', type=float, default=0.1, help="Weight for gradient matching (if using FedAvgGradMatch)")
    parser.add_argument('--server_lr',     type=float, default=1e-2,  help="Server LR")
    parser.add_argument('--server_beta1',  type=float, default=0.9,   help="Server beta1 for FedAdam")
    parser.add_argument('--server_beta2',  type=float, default=0.99,  help="Server beta2 for FedAdam")
    parser.add_argument('--server_eps',    type=float, default=1e-8,  help="Server eps for FedAdam")
    parser.add_argument('--alpha',    type=float, default=0.5,  help="Salpha for sumrate loss")
    # Model and Data Parameters
    parser.add_argument('--is_edge_update', type=bool, default=True, help="Whether to perform edge update")
    parser.add_argument('--hidden_channels', type=int, default=32, help="Number of hidden channels for GNN")
    parser.add_argument('--num_gnn_layers', type=int, default=4, help="Number of GNN layers")


    # Quantum Parameters (legacy path: main_sumrate_qml.py / Models/qml.py)
    parser.add_argument('--n_qubits', type=int, default=5, help="Number of quantum bit (qubits)")
    parser.add_argument('--n_layers', type=int, default=2, help="Number of hidden channels for Quantum layers")
    parser.add_argument('--q_dev', type=str, default="default.qubit", help="Number of GNN layers")

    # ---- Quantum star-subgraph client (Quantum/, see CLAUDE.md) ----
    parser.add_argument('--client', type=str,
                        choices=['classical', 'quantum', 'star', 'qgnn', 'qgnn_c',
                                 'sqgnn', 'sqgnn_c'],
                        default='classical',
                        help="Client model. 'classical' = Models.KG_models.ClientGNN "
                             "(reproduces the published results, default). "
                             "'quantum' = Quantum.models.qclient.QClientGNN (PQC star "
                             "message passing). 'star' = baseline B2, same fixed-k star "
                             "topology with a classical aggregator. 'qgnn' = "
                             "Quantum.models.qgnn_client.ClientQGNN: two-directional "
                             "quantum message passing with NO APConvLayer at all. "
                             "'qgnn_c' = its baseline B2', same topology and same angle "
                             "bottleneck with MLP cores. 'sqgnn' partitions all UEs "
                             "into disjoint fixed-k quantum subsets; 'sqgnn_c' is "
                             "its classical-core twin.")
    parser.add_argument('--sq_k', type=int, default=4,
                        help="UEs per disjoint subset; circuit width is 2k+1.")
    parser.add_argument('--sq_reupload', type=int, default=1,
                        help="Data re-upload blocks in each subset circuit, with "
                             "independent weights per upload.")
    parser.add_argument('--sq_agg', type=str, choices=['sum', 'mean'],
                        default='sum',
                        help="Aggregate per-subset AP readouts by sum or mean.")
    parser.add_argument('--sq_pad_flag', dest='sq_pad_flag',
                        action='store_true', default=True,
                        help="Encode +1 for real slots and -1 for padding (default).")
    parser.add_argument('--sq_no_pad_flag', dest='sq_pad_flag',
                        action='store_false',
                        help="Disable the explicit padding-slot marker.")
    parser.add_argument('--sq_ent_layers', type=int, default=2,
                        help="Entangling layers per subset-circuit block.")
    parser.add_argument('--q_k', type=int, default=4,
                        help="Star size: number of sampled UE neighbours. Circuit "
                             "width is 2k+2 wires and is independent of K.")
    parser.add_argument('--q_ent_layers', type=int, default=2,
                        help="Entangling layers L inside U_MSG / U_AGG")
    parser.add_argument('--q_layers', type=int, default=1,
                        help="Number of stacked quantum star layers")
    parser.add_argument('--q_msg_mode', type=str, choices=['pair', 'star'],
                        default='pair',
                        help="'pair' = paper-faithful (centre register untouched during "
                             "U_MSG). 'star' = ablation letting U_MSG see the centre.")
    parser.add_argument('--q_no_entangle', action='store_true', default=False,
                        help="Ablation B3: remove CRX/CNOT, keep the parameter count.")
    parser.add_argument('--q_share_update', action='store_true', default=False,
                        help="Share U_AGG weights across neighbour slots.")
    parser.add_argument('--q_sample', type=str,
                        choices=['pilot', 'importance', 'topk', 'uniform'],
                        default='pilot',
                        help="Star sampling policy (ablation S0-S3). 'uniform'=S0 "
                             "(QGNN_Comm reference), 'topk'=S1, 'importance'=S2 "
                             "(P proportional to beta), 'pilot'=S3 (cover distinct "
                             "pilot groups first, then strongest contaminators). "
                             "Stochastic policies are deterministic at eval.")
    parser.add_argument('--fl_share', type=str,
                        choices=['all', 'quantum', 'quantum_head', 'classical_core'],
                        default='all',
                        help="Which client tensors are federated. 'all' = standard "
                             "FedAvg (published behaviour). 'quantum' = only the PQC "
                             "angles are uploaded; classical encoders/heads stay "
                             "local and personalised, cutting per-round uplink by "
                             "~1300x. 'classical_core' is the size-matched control.")
    parser.add_argument('--uplink_topk', type=float, default=None,
                        help="Baseline B4: top-k magnitude sparsification fraction "
                             "of the uploaded delta (e.g. 0.01).")
    parser.add_argument('--uplink_bits', type=int, default=None,
                        help="Baseline B4: uniform quantisation bit-width of the "
                             "uploaded delta (e.g. 8).")
    parser.add_argument('--q_relaxed_invariance', action='store_true', default=False,
                        help="Restore the classical UE->AP post layers, breaking exact "
                             "UE-count invariance (ablation).")
    parser.add_argument('--q_interleave', action='store_true', default=False,
                        help="Apply one masked star layer before EVERY classical "
                             "AP->UE layer instead of once up front. Off by default: "
                             "diagnostic run H reaches 10.91 bit/s/Hz with a single "
                             "UE->AP layer, so the extra layers cost PQC evaluations "
                             "for no demonstrated benefit.")
    parser.add_argument('--q_no_interleave', action='store_true', default=False,
                        help=argparse.SUPPRESS)   # deprecated; interleaving is now opt-in
    parser.add_argument('--q_no_mag_channel', action='store_true', default=False,
                        help="Drop the raw-edge-attribute path around the circuit. "
                             "PQC expectation values are bounded in [-1,1], so this "
                             "tests whether a purely quantum message is magnitude-blind "
                             "-- the failure mode measured in run I2 (mean-pooling "
                             "collapses, sum trains). Mirrored on --client star.")
    parser.add_argument('--q_ap_qubits', type=int, default=2,
                        help="AP wires in the AP->UE circuit (--client qgnn/qgnn_c). "
                             "The AP is the broadcast SOURCE there, so 1 qubit (2 "
                             "angles) is the tightest bottleneck in the model; 2 "
                             "doubles that channel for ~0.2 h per 150-round run.")
    parser.add_argument('--q_ue_qubits', type=int, default=2,
                        help="Qubits carrying the UE embedding in the AP->UE "
                             "circuit (2 angles each). The measured bottleneck of "
                             "the 1-qubit version was per-UE bandwidth into the "
                             "circuit, which set the whole convergence time.")
    parser.add_argument('--q_edge_qubits', type=int, default=2,
                        help="Qubits carrying the edge attribute in the AP->UE "
                             "circuit (2 angles each).")
    parser.add_argument('--q_reupload', type=int, default=1,
                        help="Data re-uploading blocks in the AP->UE circuit. A "
                             "Pauli encoding used once gives a degree-1 "
                             "trigonometric polynomial in the input; R uploads "
                             "reach degree R. Each block has its own parameters. "
                             "Cost is linear in R, not exponential.")
    parser.add_argument('--q_share_dirs', type=str, default='none',
                        choices=['none', 'msg', 'all'],
                        help="Share circuit parameters between the UE->AP and AP->UE "
                             "directions. 'msg' shares the (edge,neighbour) ladder "
                             "only; 'all' shares the aggregation block too. Shapes "
                             "match in both directions by construction, so every "
                             "mode is legal.")
    parser.add_argument('--zero_init_power', action='store_true', default=False,
                        help="Zero-initialise the final Linear of the power head, so "
                             "every AP starts at sum_k r_mk = 0 and the activation "
                             "sigma(sum_k r_mk) starts at 0.5. Off by default; the "
                             "published path is unaffected. Motivation: the activation "
                             "admits a degenerate zero-power solution whose gradient "
                             "sigma(1-sigma) vanishes, and the fixed-size clients fall "
                             "into it within ~15 rounds at K=15. Applies to every "
                             "client kind, so it must be enabled for all arms or none.")
    parser.add_argument('--q_msg_plain', action='store_true', default=False,
                        help="Control B2a (--client star only): keep APConvLayer's stock "
                             "message MLP inside the masked layer, i.e. ask whether edge "
                             "masking alone trains, before any bottleneck is imposed.")
    parser.add_argument('--q_bottleneck', type=int, default=2,
                        help="Features per neighbour slot entering the aggregator. "
                             "2 = the quantum geometry (RY,RZ on one qubit). Only "
                             "--client star can exceed it; used to test whether the "
                             "bottleneck, not the aggregator, is what limits the model.")
    parser.add_argument('--q_readout', type=int, default=4,
                        help="Aggregator output width. 4 = the quantum readout "
                             "(<Z>,<X> on centre and ancilla).")
    parser.add_argument('--q_shots', type=int, default=None,
                        help="Finite measurement shots at EVALUATION (experiment E5). "
                             "Default None = analytic expectation values.")

    # Seed for reproducibility
    parser.add_argument('--seed', type=int, default=1712, help="Random seed")

    return parser.parse_args()
