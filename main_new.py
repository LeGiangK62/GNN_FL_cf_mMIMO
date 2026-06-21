"""
main_new.py
===========
A NEW federated-learning scheme for ISAC cell-free massive MIMO (sum-rate).

Difference from main_ISAC.py / fl_train_isac.py
------------------------------------------------
Old scheme (GAP-augmented):
    * every client (AP) holds a FULL heterogeneous graph (AP-UE comm edges +
      AP-SR sensing edges + the SR nodes themselves);
    * clients ship DS/PC/UI + UE/AP/SR embeddings to the server, the server
      builds per-client GAP-augmented graphs and ships them back.

New scheme (this file):
    1.  Each AP is a client and holds ONLY a local AP<->UE graph.
        It has NO sensing receivers (SR) and NO target knowledge.
    2.  After a local forward pass the client sends ONLY its AP embedding
        to the server.
    3.  The server owns the SR / target information (sensing geometry
        q_a, q_b, q_c per AP + AP coordinates). It runs a trainable GNN over
        a knowledge graph that contains ONLY AP nodes and AP<->AP edges,
        fusing all AP embeddings with the sensing/target context.
    4.  The server broadcasts the refined AP embedding (the shared KG) back
        to every client.
    5.  Each client injects its KG embedding into its local AP node and runs
        the local AP<->UE graph again to infer the power allocation that
        maximises the global sum rate.

Loss      : pure sum-rate (no CRLB term).  SR / target only enriches the
            server-side AP knowledge graph.
Server     : a trainable AP<->AP GNN; gradients from every client's sum-rate
            loss flow back into it. One global server model (not federated),
            updated each round alongside FedAvg of the client models.

Only this file is edited; helpers (component / rate maths, FedAvg) are reused
from Utils.* and Models.*.
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin, LayerNorm, LeakyReLU
from torch.utils.data import Subset, TensorDataset
from torch.utils.data import DataLoader as TorchLoader
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeoLoader

from Utils.args import parse_args
from Utils.fl_train import FedAvg, FedProx, sample_clients

from Utils.fl_kg_isac import ClientGNN, compute_sensing, build_split, train_round, ServerGNN, evaluate

# Centralized benchmark (same components as main_ISAC.py)
from Models.GNN import IsacHetNet
from Utils.isac_data import build_cen_loader_isac
from Utils.centralized_train import (
    cen_train_isac_sumrate, cen_eval_isac_sumrate, cen_loss_function_isac_sumrate,
)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SAVE_DIR = ".results/new_scheme"
MODEL_DIR = SAVE_DIR + "/models/"
EVAL_DIR = SAVE_DIR + "/eval/"
TRAIN_DIR = SAVE_DIR + "/train/"


def init_folder():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)




# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = time.strftime("%y_%m_%d_%H_%M_%S", time.localtime(time.time()))
    init_folder()

    args = parse_args()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- params ----
    hidden_channels = args.hidden_channels
    num_gnn_layers = args.num_gnn_layers
    num_rounds = args.num_rounds
    lr = args.lr
    client_fraction = args.client_fraction
    num_train, num_test, num_eval = args.num_train, args.num_test, args.num_eval
    batch_size = args.batch_size
    comm_rounds = args.comm_rounds
    tau = args.tau
    num_antenna = args.num_antenna
    rho_p, rho_d = args.power_f, args.power_f
    num_ap, num_ue = args.num_ap, args.num_ue
    eval_round = args.eval_round if args.eval_round else max(1, num_rounds // 10)

    # centralized benchmark hyperparams
    nu = args.nu
    cen_pretrain = args.cen_pretrain
    cen_lr = args.cen_lr
    num_epochs_cen = args.num_epochs_cen

    # sensing scaling (same constants as main_ISAC)
    c = 3e8
    sigma_s = 1e-3
    B_sens = 20 * 1e6
    zeta = (np.pi * B_sens / (sigma_s * c)) ** 2 * 8

    # ---- load data ----
    file_name = f"noQ_dl_isac_sumrate_data_2000_{num_ue}_{num_ap}"
    mat_data = scipy.io.loadmat("Data/" + file_name + ".mat")

    beta_all = mat_data["betas"]
    gamma_all = mat_data["Gammas"]
    phi_all = mat_data["Phii_cf"].transpose(0, 2, 1)        # [N, K, tau]
    ap_cor_all = mat_data["ap_locations"]                   # [N, M, 2]
    sr_cor_all = mat_data["sr_locations"]                   # [N, T, 2]
    rcs_all = mat_data["rcs_values"]                        # [N, M, T]
    rates_equal_solutions = mat_data["R_equal"][0]
    rates_log_solutions = mat_data["R_log"][0]

    perm = np.random.RandomState(seed).permutation(beta_all.shape[0])
    train_idx = perm[:num_train]
    test_idx = perm[num_train: num_train + num_test]
    eval_idx = perm[-num_eval:]

    # ---- normalisation stats from the training split ----
    ap_mean = ap_cor_all[train_idx].reshape(-1, 2).mean(0)
    ap_std = ap_cor_all[train_idx].reshape(-1, 2).std(0) + 1e-9

    # q stats (standardise the Fisher-info features for a stable server input)
    q_train = compute_sensing(ap_cor_all[train_idx], sr_cor_all[train_idx],
                              rcs_all[train_idx], zeta)
    q_mean = q_train.reshape(-1, 3).mean(0)
    q_std = q_train.reshape(-1, 3).std(0) + 1e-9

    def build(idx, bs, order_seed):
        return build_split(
            beta_all[idx], gamma_all[idx], phi_all[idx],
            ap_cor_all[idx], sr_cor_all[idx], rcs_all[idx], zeta,
            ap_mean, ap_std, q_mean, q_std, bs, order_seed)

    train_aps, train_sens, M, K = build(train_idx, batch_size, seed)
    test_aps, test_sens, _, _ = build(test_idx, batch_size, seed)
    eval_aps, eval_sens, _, _ = build(eval_idx, num_eval, seed)

    # ---- centralized benchmark loaders (full ISAC graph: AP+UE+SR) ----
    train_data_cen, train_loader_cen = build_cen_loader_isac(
        beta_all[train_idx], gamma_all[train_idx], phi_all[train_idx], batch_size,
        zeta, nu, rcs_all[train_idx], ap_cor_all[train_idx], sr_cor_all[train_idx],
        isShuffle=True)
    test_data_cen, test_loader_cen = build_cen_loader_isac(
        beta_all[test_idx], gamma_all[test_idx], phi_all[test_idx], batch_size,
        zeta, nu, rcs_all[test_idx], ap_cor_all[test_idx], sr_cor_all[test_idx])
    eval_data_cen, eval_loader_cen = build_cen_loader_isac(
        beta_all[eval_idx], gamma_all[eval_idx], phi_all[eval_idx], num_eval,
        zeta, nu, rcs_all[eval_idx], ap_cor_all[eval_idx], sr_cor_all[eval_idx])

    # ---- models ----
    dim_dict = {"UE": tau, "AP": 2, "comm_edge": 2}
    out_channels = hidden_channels

    global_model = ClientGNN(dim_dict, out_channels,
                             num_layers=num_gnn_layers,
                             hid_layers=hidden_channels // 2).to(device)
    local_models, optimizers, schedulers = [], [], []
    for _ in range(M):
        m = ClientGNN(dim_dict, out_channels,
                      num_layers=num_gnn_layers,
                      hid_layers=hidden_channels // 2).to(device)
        m.load_state_dict(global_model.state_dict())
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        local_models.append(m)
        optimizers.append(opt)
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=num_rounds, eta_min=1e-5))
    
    server_model = ServerGNN(out_channels, sensing_dim=5, sig_dim=tau,
                             hidden=hidden_channels // 2, num_layers=2,
                             kg_edge_mode="inner", param_free=args.param_free).to(device)
    # ServerGNN is parameter-free (hand-crafted KG) -> no optimizer/scheduler.
    server_params = list(server_model.parameters())
    if server_params:
        server_opt = torch.optim.AdamW(server_params, lr=lr, weight_decay=1e-4)
        server_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            server_opt, T_max=num_rounds, eta_min=1e-5)
    else:
        server_opt, server_sched = None, None

    if args.fl_scheme == "fedprox":
        fed = FedProx(client_fraction=client_fraction, mu=args.mu)
        fed.set_global_weights(global_model)
    else:
        fed = FedAvg(client_fraction=client_fraction)

    # ---- training ----
    fl_train_curve, fl_test_curve = [], []
    if args.fl_pretrain is None:
        start = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n ===={start}==== Training new-scheme FL ({args.fl_scheme.upper()}) ...")
        print(f"Equal power: train {np.mean(rates_equal_solutions[train_idx]):.4f}, "
              f"test {np.mean(rates_equal_solutions[test_idx]):.4f}")
        print(f"Log approx : train {np.mean(rates_log_solutions[train_idx]):.4f}, "
              f"test {np.mean(rates_log_solutions[test_idx]):.4f}")
        t0 = time.time()
        for rnd in range(num_rounds):
            selected = sample_clients(client_fraction, M)
            train_round(train_aps, train_sens, M, server_model, server_opt,
                        local_models, optimizers, selected, fed, global_model,
                        tau, rho_d, num_antenna, comm_rounds, device,
                        ctde=args.ctde, lam=args.lam, use_kg=not args.no_kg)

            global_weights = fed.aggregate(global_model, local_models, selected)
            global_model.load_state_dict(global_weights)
            for m in local_models:
                m.load_state_dict(global_model.state_dict())
                
            for ci in selected:
                schedulers[ci].step()
            if server_sched is not None:
                server_sched.step()

            if rnd % eval_round == 0:
                tr = evaluate(train_aps, train_sens, M, server_model, local_models,
                              tau, rho_d, num_antenna, comm_rounds, device,
                              use_kg=not args.no_kg).mean().item()
                te = evaluate(test_aps, test_sens, M, server_model, local_models,
                              tau, rho_d, num_antenna, comm_rounds, device,
                              use_kg=not args.no_kg).mean().item()
                fl_train_curve.append(tr)
                fl_test_curve.append(te)
                print(f"Round {rnd + 1:03d}/{num_rounds}: "
                      f"Train Rate = {tr:.4f} | Eval Rate = {te:.4f}")

        print(f"Execution Time: {timedelta(seconds=time.time() - t0)}")
        if not args.no_save:
            torch.save(global_model.state_dict(), f"{MODEL_DIR}/{timestamp}_fl.pth")
            torch.save(server_model.state_dict(), f"{MODEL_DIR}/{timestamp}_server.pth")
            print(f"Saved client model to {MODEL_DIR}/{timestamp}_fl.pth")
            print(f"Saved server model to {MODEL_DIR}/{timestamp}_server.pth")

        plt.figure(figsize=(6, 4), dpi=180)
        x_axis = [i * eval_round for i in range(len(fl_train_curve))]
        plt.plot(x_axis, fl_train_curve, label="Training Rate", linewidth=2)
        plt.plot(x_axis, fl_test_curve, label="Testing Rate", linewidth=2)
        plt.axhline(np.mean(rates_log_solutions[train_idx]), color="r", ls="--",
                    label="Training Optimal")
        plt.axhline(np.mean(rates_log_solutions[test_idx]), color="b", ls="--",
                    label="Testing Optimal")
        plt.xlabel("Rounds"); plt.ylabel("Sum rate")
        plt.title(f"New-scheme FL ({args.fl_scheme.upper()}) - {lr}_{num_rounds}")
        plt.grid(True, ls="--", alpha=0.6); plt.legend(); plt.tight_layout()
        if not args.no_save:
            plt.savefig(TRAIN_DIR + f"{timestamp}_fl.png", dpi=300)
    else:
        global_model.load_state_dict(torch.load(f"{MODEL_DIR}/{args.fl_pretrain}.pth"))
        for m in local_models:
            m.load_state_dict(global_model.state_dict())
        server_path = f"{MODEL_DIR}/{args.fl_pretrain.replace('_fl', '_server')}.pth"
        if os.path.exists(server_path):
            server_model.load_state_dict(torch.load(server_path))
        print(f"Loaded pretrained FL model {args.fl_pretrain}.")

    # ---- centralized GNN benchmark (similar to main_ISAC.py) ----
    ap_dim_cen = train_data_cen[0]["AP"].x.shape[1]
    ue_dim_cen = train_data_cen[0]["UE"].x.shape[1]
    sr_dim_cen = train_data_cen[0]["SR"].x.shape[1]
    comm_edge_dim = train_data_cen[0][("AP", "comm_down", "UE")].edge_attr.shape[1]
    sens_edge_dim = train_data_cen[0][("AP", "sens_down", "SR")].edge_attr.shape[1]
    cen_dim_dict = {
        "UE": ue_dim_cen, "AP": ap_dim_cen, "SR": sr_dim_cen,
        "comm_edge": comm_edge_dim, "sens_edge": sens_edge_dim,
    }

    cen_model = IsacHetNet(
        dim_dict=cen_dim_dict,
        out_channels=args.cen_hidden_channels,
        num_layers=args.cen_num_gnn_layers // 2,
        hid_layers=args.cen_hidden_channels // 2,
    ).to(device)
    cen_optimizer = torch.optim.AdamW(cen_model.parameters(), lr=cen_lr, weight_decay=1e-4)
    cen_scheduler = torch.optim.lr_scheduler.StepLR(
        cen_optimizer, step_size=max(1, num_epochs_cen // 10), gamma=0.8)

    if cen_pretrain is not None:
        cen_model.load_state_dict(torch.load(f"{MODEL_DIR}/{cen_pretrain}.pth"))
        print(f"Loaded pretrained centralized GNN {cen_pretrain}.")
    else:
        start = datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"\n ===={start}==== Training centralized GNN benchmark ...")
        t0 = time.time()
        eval_epochs_cen = max(1, num_epochs_cen // 10)
        for epoch in range(num_epochs_cen):
            cen_model.train()
            train_loss = cen_train_isac_sumrate(
                epoch / (2 * num_epochs_cen // 3 + 1),
                train_loader_cen, cen_model, cen_optimizer,
                tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna, nu=nu)
            cen_model.eval()
            with torch.no_grad():
                tr = cen_eval_isac_sumrate(train_loader_cen, cen_model,
                                           tau=tau, rho_p=rho_p, rho_d=rho_d,
                                           num_antenna=num_antenna, nu=nu)
                te = cen_eval_isac_sumrate(test_loader_cen, cen_model,
                                           tau=tau, rho_p=rho_p, rho_d=rho_d,
                                           num_antenna=num_antenna, nu=nu)
            cen_scheduler.step()
            if epoch % eval_epochs_cen == 0:
                print(f"Epoch {epoch + 1:03d}/{num_epochs_cen} | "
                      f"Loss {train_loss:.4f} | Train Rate {tr:.4f} | Test Rate {te:.4f}")
        print(f"Centralized Execution Time: {timedelta(seconds=time.time() - t0)}")
        if not args.no_save:
            torch.save(cen_model.state_dict(), f"{MODEL_DIR}/{timestamp}_cen.pth")
            print(f"Saved centralized GNN to {MODEL_DIR}/{timestamp}_cen.pth")

    # ---- evaluation (CDF) ----
    if args.eval_plot:
        print("Evaluation" + "=" * 20)
        fl_rates = evaluate(eval_aps, eval_sens, M, server_model, local_models,
                            tau, rho_d, num_antenna, comm_rounds, device,
                            use_kg=not args.no_kg)
        fl_rates = fl_rates.detach().cpu().numpy()

        # centralized GNN rates on the eval split
        cen_model.eval()
        with torch.no_grad():
            for batch in eval_loader_cen:
                batch = batch.to(device)
                x_dict, edge_dict, _ = cen_model(batch)
                cen_rates, _ = cen_loss_function_isac_sumrate(
                    batch, x_dict, edge_dict,
                    tau=tau, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
                    nu=nu, eval_mode=True)
        cen_rates = cen_rates.detach().cpu().numpy()

        rates_equal = rates_equal_solutions[eval_idx].copy()
        rates_log = rates_log_solutions[eval_idx].copy()

        print(f"Sum rate avg: Centralized {cen_rates.mean():.2f} | "
              f"FL new-scheme {fl_rates.mean():.2f} | "
              f"Log approx {rates_log.mean():.2f}")
        print(f"  FL vs Centralized : {fl_rates.mean() * 100 / cen_rates.mean():.2f}%")
        print(f"  FL vs Log approx  : {fl_rates.mean() * 100 / rates_log.mean():.2f}%")

        n = len(fl_rates)
        max_value = np.ceil(max(fl_rates.max(), cen_rates.max(),
                                rates_equal.max(), rates_log.max()) * 100) / 100
        y_axis = np.linspace(0, 1, n + 2)
        for arr in (fl_rates, cen_rates, rates_equal, rates_log):
            arr.sort()
        def pad(a):
            a = np.insert(a, 0, 0.0)
            return np.insert(a, n + 1, max_value)
        plt.figure(figsize=(6, 4), dpi=180)
        plt.plot(pad(cen_rates), y_axis, label="Centralized GNN", linewidth=2)
        plt.plot(pad(fl_rates), y_axis, label="FL (new scheme)", linewidth=2)
        plt.plot(pad(rates_equal), y_axis, label="Equal Power", linewidth=2)
        plt.plot(pad(rates_log), y_axis, label="Log Approx.", linewidth=2)
        plt.xlabel("Sum rate [bps/Hz]", {"fontsize": 16})
        plt.ylabel("Empirical CDF", {"fontsize": 16})
        plt.legend(fontsize=14); plt.grid()
        eval_path = EVAL_DIR + f"{timestamp}_eval.png"
        if not args.no_save:
            plt.savefig(eval_path, dpi=300, bbox_inches="tight")
            print(f"Saved evaluation figure to {eval_path}.")

    print("eta = 5, p = 1.5")
    print("No marginal at all ")
    print("No net_util in ap embed ")