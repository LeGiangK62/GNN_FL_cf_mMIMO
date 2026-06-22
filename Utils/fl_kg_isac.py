import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeoLoader


from Utils.comm import power_from_raw, component_calculate, rate_from_component
from Models.KG_models import ClientGNN, ServerGNN

# =============================================================================
# Data : local AP<->UE graphs (client side) + sensing tensor (server side)
# =============================================================================



def client_data(
        beta_single_sample, gamma_single_sample, phi_single_sample,
        zeta=None, nu=None,
        ap_cor=None,
        # q_a_single=None, q_b_single=None, q_c_single=None,
        ap_id=None, sample_id=None,
    ):
    """
    Build a HeteroData object for client.

    Communication side mirrors data_gen.full_het_graph:
        Nodes:  AP [num_AP, 1], UE [num_UE, tau]
        Edges:  AP --down--> UE      with attr [beta, gamma]
                UE --up--> AP        with attr [beta, gamma]
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_AP, num_UE = beta_single_sample.shape

    # ---------- AP / UE nodes ----------
    ap_features = np.ones((num_AP, 1), dtype=np.float32)
    # Append target distance to AP feature  -> [num_AP, 2]
    ap_features = np.concatenate(
        [
            ap_features, 
            np.asarray(ap_cor, dtype=np.float32), 
        ], 
        axis=1,
    )
    ue_features = phi_single_sample

    x_ap = torch.tensor(ap_features, dtype=torch.float32).to(device)
    x_ue = torch.tensor(ue_features, dtype=torch.float32).to(device)

    # ---------- AP <-> UE edges ----------
    edge_index_ap_down_ue = []
    edge_index_ue_up_ap = []
    for ap_idx in range(num_AP):
        for ue_idx in range(num_UE):
            edge_index_ap_down_ue.append([ap_idx, ue_idx])
    for ue_idx in range(num_UE):
        for ap_idx in range(num_AP):
            edge_index_ue_up_ap.append([ue_idx, ap_idx])

    edge_index_ap_down_ue = torch.tensor(edge_index_ap_down_ue, dtype=torch.long).t().contiguous().to(device)
    edge_index_ue_up_ap   = torch.tensor(edge_index_ue_up_ap,   dtype=torch.long).t().contiguous().to(device)

    beta_up  = beta_single_sample.reshape(-1, 1)
    gamma_up = gamma_single_sample.reshape(-1, 1)
    edge_attr_ap_to_ue = torch.tensor(
        np.concatenate((beta_up, gamma_up), axis=1), dtype=torch.float32
    ).to(device)

    beta_down  = beta_single_sample.T.reshape(-1, 1)
    gamma_down = gamma_single_sample.T.reshape(-1, 1)
    edge_attr_ue_up_ap = torch.tensor(
        np.concatenate((beta_down, gamma_down), axis=1), dtype=torch.float32
    ).to(device)

    data = HeteroData()
    data['AP'].x = x_ap
    data['UE'].x = x_ue
    data['AP', 'comm_down', 'UE'].edge_index = edge_index_ap_down_ue
    data['AP', 'comm_down', 'UE'].edge_attr  = edge_attr_ap_to_ue
    data['UE', 'comm_up', 'AP'].edge_index = edge_index_ue_up_ap
    data['UE', 'comm_up', 'AP'].edge_attr  = edge_attr_ue_up_ap

    data.ap_id = ap_id
    data.sample_id = sample_id
    return data


def server_data(rcs_single, ap_cor, sr_cor, zeta,
                ap_mean, ap_std, q_mean, q_std, tar_cor=None,
                ap_id=None, sample_id=None):
    """Build the server-side AP<->SR knowledge graph for one sample.

    Thin wrapper kept for naming continuity -- delegates to
    ``make_server_graph``.  Unlike the original draft, the client AP
    embeddings are NOT baked into the graph here: the graph is static (so it
    can live in a DataLoader) and the embeddings are fused inside ``ServerGNN``
    every round.  The target (default origin) enters both the Fisher-info and
    the direction/distance node features.
    """
    data = make_server_graph(ap_cor, sr_cor, rcs_single, zeta,
                             ap_mean, ap_std, q_mean, q_std, tar_cor=tar_cor)
    data.ap_id = ap_id
    data.sample_id = sample_id
    return data


def compute_sensing(ap_cor, sr_cor, rcs, zeta, tar_cor=None):
    """Per-AP sensing geometry (Fisher info diagonal/cross) from SR + target.

    The bistatic Fisher information depends on the geometry *relative to the
    target*.  ``tar_cor`` is the target location (defaults to the origin, which
    reproduces the original behaviour); pass a real target to generalise.

    ap_cor  : [N, M, 2]   AP coordinates (server only)
    sr_cor  : [N, T, 2]   sensing-receiver coordinates (server only)
    rcs     : [N, M, T]   radar cross sections (server only)
    tar_cor : [2] or None  target location (default origin)
    Returns q_all : [N, M, 3]  ->  (q_a, q_b, q_c) per AP.
    """
    N, M, _ = ap_cor.shape
    eps = np.finfo(float).eps
    tar = np.zeros((1, 2)) if tar_cor is None else np.asarray(tar_cor, dtype=float).reshape(1, 2)
    q_all = np.zeros((N, M, 3), dtype=np.float32)
    for s in range(N):
        ap = ap_cor[s] - tar                       # [M, 2]  AP -> target vector
        sr = sr_cor[s] - tar                       # [T, 2]  SR -> target vector
        dist_ap = np.linalg.norm(ap, axis=1, keepdims=True)   # [M, 1]
        dist_sr = np.linalg.norm(sr, axis=1, keepdims=True)   # [T, 1]
        dc_ap = ap / (dist_ap + eps)               # [M, 2]
        dc_sr = sr / (dist_sr + eps)               # [T, 2]

        grad_x = dc_ap[:, 0:1] + dc_sr[:, 0:1].T   # [M, T]
        grad_y = dc_ap[:, 1:2] + dc_sr[:, 1:2].T   # [M, T]
        rcs_s = rcs[s]                              # [M, T]

        q_a = zeta * np.sum(rcs_s * grad_x ** 2, axis=1)        # [M]
        q_b = zeta * np.sum(rcs_s * grad_y ** 2, axis=1)        # [M]
        q_c = zeta * np.sum(rcs_s * grad_x * grad_y, axis=1)    # [M]
        q_all[s] = np.stack([q_a, q_b, q_c], axis=1)
    return q_all




def make_server_graph(ap_cor, sr_cor, rcs, zeta,
                      ap_mean, ap_std, q_mean, q_std, tar_cor=None):
    """One sample's server-side AP<->SR bipartite graph (target-aware).

    Nodes:
        AP     : [M, 8]  (normalised AP coords, standardised q_a/q_b/q_c,
                          unit direction to target, log distance to target)
        SR     : [T, 5]  (normalised SR coords, unit direction to target,
                          log distance to target)
        TARGET : [1, 2]  (normalised target coordinates) -- a single global
                         hub node every AP/SR connects to.
    Edges:
        AP     --sens_down--> SR       attr = rcs
        SR     --sens_up-->   AP       attr = rcs
        AP     --ap2tar-->   TARGET    attr = log distance AP->target
        TARGET --tar2ap-->   AP        attr = log distance AP->target
        SR     --sr2tar-->   TARGET    attr = log distance SR->target
        TARGET --tar2sr-->   SR        attr = log distance SR->target

    The target (default origin) enters both the Fisher-info features and the
    explicit direction/distance features, so the server reasons about *where*
    each AP/SR sits relative to the sensed target -- not only its coordinates.
    The client AP embeddings are fused into the AP nodes inside ``ServerGNN``.
    """
    eps = np.finfo(float).eps
    tar = np.zeros((1, 2), dtype=np.float32) if tar_cor is None \
        else np.asarray(tar_cor, dtype=np.float32).reshape(1, 2)

    M = ap_cor.shape[0]
    T = sr_cor.shape[0]

    # geometry relative to the target
    v_ap = ap_cor - tar                                   # [M, 2]
    d_ap = np.linalg.norm(v_ap, axis=1, keepdims=True)    # [M, 1]
    dir_ap = v_ap / (d_ap + eps)
    v_sr = sr_cor - tar                                   # [T, 2]
    d_sr = np.linalg.norm(v_sr, axis=1, keepdims=True)    # [T, 1]
    dir_sr = v_sr / (d_sr + eps)

    # target-aware Fisher info per AP, standardised with the training stats
    q = compute_sensing(ap_cor[None], sr_cor[None], rcs[None], zeta, tar_cor=tar)[0]  # [M, 3]
    q_norm = (q - q_mean) / q_std

    ap_norm = (ap_cor - ap_mean) / ap_std                 # [M, 2]
    sr_norm = (sr_cor - ap_mean) / ap_std                 # [T, 2]  (share AP coord scale)

    ap_feat = np.concatenate([ap_norm, q_norm, dir_ap, np.log1p(d_ap)], axis=1).astype(np.float32)
    sr_feat = np.concatenate([sr_norm, dir_sr, np.log1p(d_sr)], axis=1).astype(np.float32)
    tar_norm = ((tar - ap_mean) / ap_std).astype(np.float32)   # [1, 2]

    data = HeteroData()
    data["AP"].x = torch.tensor(ap_feat, dtype=torch.float32)       # [M, SERVER_AP_FEAT_DIM]
    # Raw (non-standardised, >=0) per-AP Fisher entries (q_a, q_b, q_c) kept
    # alongside x so the server can build the localization FIM/CRB directly
    # (x carries only the z-scored q_norm, which can go negative).
    data["AP"].q_raw = torch.tensor(q, dtype=torch.float32)         # [M, 3]
    data["SR"].x = torch.tensor(sr_feat, dtype=torch.float32)       # [T, SERVER_SR_FEAT_DIM]
    data["TARGET"].x = torch.tensor(tar_norm, dtype=torch.float32)  # [1, SERVER_TAR_FEAT_DIM]

    # fully connected AP<->SR bipartite edges, weighted by rcs
    ap_sr = [[a, s] for a in range(M) for s in range(T)]       # (a, s) row-major
    sr_ap = [[s, a] for s in range(T) for a in range(M)]
    data["AP", "sens_down", "SR"].edge_index = torch.tensor(ap_sr, dtype=torch.long).t().contiguous()
    data["SR", "sens_up", "AP"].edge_index = torch.tensor(sr_ap, dtype=torch.long).t().contiguous()
    rcs_mat = np.asarray(rcs, dtype=np.float32)               # [M, T]
    data["AP", "sens_down", "SR"].edge_attr = torch.tensor(rcs_mat.reshape(-1, 1), dtype=torch.float32)
    data["SR", "sens_up", "AP"].edge_attr = torch.tensor(rcs_mat.T.reshape(-1, 1), dtype=torch.float32)

    # AP<->TARGET and SR<->TARGET edges (single target node, index 0), the
    # global hub through which all APs/SRs coordinate.  Edge attr = log distance.
    ap_idx = torch.arange(M, dtype=torch.long)
    sr_idx = torch.arange(T, dtype=torch.long)
    zeros_m = torch.zeros(M, dtype=torch.long)
    zeros_t = torch.zeros(T, dtype=torch.long)
    d_ap_log = torch.tensor(np.log1p(d_ap), dtype=torch.float32)   # [M, 1]
    d_sr_log = torch.tensor(np.log1p(d_sr), dtype=torch.float32)   # [T, 1]

    data["AP", "ap2tar", "TARGET"].edge_index = torch.stack([ap_idx, zeros_m], dim=0)
    data["AP", "ap2tar", "TARGET"].edge_attr = d_ap_log
    data["TARGET", "tar2ap", "AP"].edge_index = torch.stack([zeros_m, ap_idx], dim=0)
    data["TARGET", "tar2ap", "AP"].edge_attr = d_ap_log

    data["SR", "sr2tar", "TARGET"].edge_index = torch.stack([sr_idx, zeros_t], dim=0)
    data["SR", "sr2tar", "TARGET"].edge_attr = d_sr_log
    data["TARGET", "tar2sr", "SR"].edge_index = torch.stack([zeros_t, sr_idx], dim=0)
    data["TARGET", "tar2sr", "SR"].edge_attr = d_sr_log
    return data


def make_ap_ue_graph(beta_log_ap, gamma_ap, phi, ap_feat):
    """One client's local graph: a single AP node + its K UE nodes.

    beta_log_ap : [K]   log1p(beta) for this AP to every UE
    gamma_ap    : [K]   channel variance (Gamma) for this AP to every UE
    phi         : [K, tau]  pilot sequences (UE node features)
    ap_feat     : [2]   AP node feature (normalised AP coordinates)
    """
    K = phi.shape[0]
    data = HeteroData()
    data["AP"].x = torch.tensor(ap_feat[None, :], dtype=torch.float32)   # [1, 2]
    data["UE"].x = torch.tensor(phi, dtype=torch.float32)                # [K, tau]

    # ---- power-free pilot-energy signature (privacy-light interference key) ----
    # z[t] = sum_{k: pilot t} beta_{m,k} = phi^T @ beta  -> [tau].  No power, no
    # DS/PC/UI leave the client; <z_m, z_m'> reconstructs the pilot-contamination
    # coupling between APs at the server.  L2-normalised so the inner product is a
    # cosine in [0, 1] (beta >= 0), giving well-scaled edge weights.
    beta = np.expm1(beta_log_ap).astype(np.float32)                      # [K] real gain
    sig = phi.T.astype(np.float32) @ beta                                # [tau]
    sig = sig / (np.linalg.norm(sig) + 1e-9)
    data["AP"].sig = torch.tensor(sig[None, :], dtype=torch.float32)     # [1, tau]

    edge_attr = torch.tensor(
        np.stack([beta_log_ap, gamma_ap], axis=1), dtype=torch.float32   # [K, 2]
    )

    # AP (node 0) -> every UE
    down = torch.tensor([[0] * K, list(range(K))], dtype=torch.long)
    # every UE -> AP (node 0)
    up = torch.tensor([list(range(K)), [0] * K], dtype=torch.long)

    data["AP", "comm_down", "UE"].edge_index = down
    data["AP", "comm_down", "UE"].edge_attr = edge_attr
    data["UE", "comm_up", "AP"].edge_index = up
    data["UE", "comm_up", "AP"].edge_attr = edge_attr
    return data


def build_split(beta, gamma, phi, ap_cor, sr_cor, rcs, zeta,
                ap_mean, ap_std, q_mean, q_std, batch_size, order_seed):
    """Build aligned per-AP graph loaders + the server sensing loader.

    All loaders share one shuffle order so that, when iterated in parallel,
    batch b of every AP loader and of the sensing loader refer to the SAME
    set of samples.  Client graphs see only the (normalised) AP coordinates;
    the sensing tensor (standardised q_a/q_b/q_c + AP coords) is server-only.
    """
    N, M, K = beta.shape
    beta_log = np.log1p(beta)
    ap_norm = (ap_cor - ap_mean) / ap_std                                   # [N, M, 2]

    # ----- per-AP local graphs (client side: AP<->UE only) -----
    per_ap_data = [
        [make_ap_ue_graph(beta_log[s, ap], gamma[s, ap], phi[s], ap_norm[s, ap])
         for s in range(N)]
        for ap in range(M)
    ]

    # ----- server-side AP<->SR knowledge graphs (one per sample) -----
    server_graphs = [
        make_server_graph(ap_cor[s], sr_cor[s], rcs[s], zeta,
                          ap_mean, ap_std, q_mean, q_std)
        for s in range(N)
    ]

    g = torch.Generator().manual_seed(order_seed)
    order = torch.randperm(N, generator=g).tolist()

    ap_loaders = [
        GeoLoader(Subset(per_ap_data[ap], order),
                  batch_size=batch_size, shuffle=False, drop_last=False)
        for ap in range(M)
    ]
    server_loader = GeoLoader(Subset(server_graphs, order),
                              batch_size=batch_size, shuffle=False, drop_last=False)
    return ap_loaders, server_loader, M, K


# =============================================================================
# Models
# =============================================================================




def phase1_components(outs, batches, tau, rho_d, num_antenna):
    """Per-AP Phase-1 DS/PC/UI lists (under the no-KG power) for the server.

    Returns ds, pc, ui : lists of M tensors ([B,1,K], [B,1,K,K], [B,1,K,K]),
    detached features the server turns into the global / leave-one-out rates.
    """
    ds, pc, ui = [], [], []
    for o, b in zip(outs, batches):
        DS, PC, UI = compute_components(b, o[1], tau, rho_d, num_antenna, flag=False)
        ds.append(DS); pc.append(PC); ui.append(UI)
    return ds, pc, ui


def _kg_for_client(gap, attn, ci, param_free):
    """What client ``ci`` receives: own gap row (param-free server) or the full
    KG + its attention row (conv server).  ``(None, None)`` when KG disabled."""
    if gap is None:
        return None, None
    if param_free:
        return gap[:, ci, :], None
    return gap, attn[:, ci, :]





# =============================================================================
# Rate maths
# =============================================================================

def compute_components(batch, edge_attr_dict, tau, rho_d, num_antenna, flag=False):
    """DS / PC / UI for one client's (single-AP) batch."""
    num_graphs = batch.num_graphs
    num_UEs = batch["UE"].x.shape[0] // num_graphs
    num_APs = batch["AP"].x.shape[0] // num_graphs                    # == 1

    pilot = batch["UE"].x[:, :tau].reshape(num_graphs, num_UEs, -1)
    raw_edge = batch["AP", "comm_down", "UE"].edge_attr.reshape(
        num_graphs, num_APs, num_UEs, -1)
    large_scale = torch.expm1(raw_edge[:, :, :, 0])
    channel_var = raw_edge[:, :, :, 1]

    power_raw = edge_attr_dict["AP", "comm_down", "UE"].reshape(
        num_graphs, num_APs, num_UEs, -1)[:, :, :, -1]
    if flag:  power_raw = torch.ones_like(power_raw)
    power = power_from_raw(power_raw, channel_var, num_antenna)
    DS, PC, UI = component_calculate(power, channel_var, large_scale, pilot, rho_d=rho_d)
    return DS, PC, UI                                                 # [B,1,K], [B,1,K,K], [B,1,K,K]


def global_sum_rate(all_DS, all_PC, all_UI, num_antenna):
    """Coherent global rate from per-AP components (lists of [B,1,...])."""
    DS = torch.cat(all_DS, dim=1)        # [B, M, K]
    PC = torch.cat(all_PC, dim=1)        # [B, M, K, K]
    UI = torch.cat(all_UI, dim=1)        # [B, M, K, K]
    rate = rate_from_component(DS, PC, UI, num_antenna)   # [B, K]
    return rate.sum(dim=1)                                # [B]


# =============================================================================
# Training / evaluation
# =============================================================================

def train_round(ap_loaders, sensing_loader, M, server_model, server_opt,
                local_models, optimizers, selected, fed, global_model,
                tau, rho_d, num_antenna, comm_rounds, device,
                ctde=False, lam=0.2, use_kg=True):
    """Compatibility entry point used by main_new.py -- runs the 3-phase round.

    FedAvg aggregation is performed by the caller (main_new.py) AFTER this
    returns, so it is intentionally NOT done here.
    """
    train_round_new(ap_loaders, sensing_loader, M, server_model, server_opt,
                    local_models, optimizers, selected,
                    tau, rho_d, num_antenna, comm_rounds, device,
                    ctde=ctde, lam=lam, use_kg=use_kg)


@torch.no_grad()
def evaluate_old(ap_loaders, sensing_loader, M, server_model, local_models,
             tau, rho_d, num_antenna, comm_rounds, device, use_kg=True):
    for m in local_models:
        m.eval()
    server_model.eval()

    sum_rates = []
    for batch_tuple in zip(*ap_loaders, sensing_loader):
        client_batches = [b.to(device) for b in batch_tuple[:M]]
        server_b = batch_tuple[M].to(device)                       # AP<->SR graph batch

        # ----- Phase 1: AP embedding + DS/PC/UI, sent to the server -----
        outs = [local_models[i](b) for i, b in enumerate(client_batches)]
        ap_emb = torch.stack([o[0]["AP"] for o in outs], dim=1)    # [B, M, F]
        ds_all, pc_all, ui_all = phase1_components(outs, client_batches, tau, rho_d, num_antenna)

        # ----- Phase 2 + 3: server KG (+ rates), then each client power -----
        gap, attn = (server_model(server_b, ap_emb, ds_all, pc_all, ui_all, num_antenna)
                     if use_kg else (None, None))
        DS_all, PC_all, UI_all, ap_list = [], [], [], []
        for i, b in enumerate(client_batches):                 # Phase 3
            kg_i, attn_i = _kg_for_client(gap, attn, i, server_model.param_free)
            x_dict, edge_attr_dict, _ = local_models[i](b, kg_emb=kg_i, kg_attn=attn_i)
            DS_k, PC_k, UI_k = compute_components(b, edge_attr_dict, tau, rho_d, num_antenna)
            DS_all.append(DS_k); PC_all.append(PC_k); UI_all.append(UI_k)
            ap_list.append(x_dict["AP"])
        ap_emb = torch.stack(ap_list, dim=1)                   # refine for next comm round

        # achieved metric IS the true global (coherent) sum rate across all APs
        sum_rates.append(global_sum_rate(DS_all, PC_all, UI_all, num_antenna))
    return torch.cat(sum_rates, dim=0)                            # [num_samples]


def loss_function_old(client_batch, edge_attr_dict, tau, rho_d, num_antenna, alpha=0.1):
    """Purely-LOCAL sum-rate surrogate for one client (AP).

    Maximise this AP's own desired signal and penalise the interference it
    creates -- using ONLY its own DS/PC/UI.  No DS/PC/UI is exchanged between
    clients (unlike the global-rate loss in the old scheme).

        loss = -sum_k DS_k  +  alpha * sum_k (PC_k + UI_k)
    """
    DS_k, PC_k, UI_k = compute_components(
        client_batch, edge_attr_dict, tau, rho_d, num_antenna)      # [B,1,K], [B,1,K,K] x2

    local_interf_per_ue = PC_k.sum(dim=2) + UI_k.sum(dim=2)          # [B, 1, K]

    weight = 1  # prev version weighted by the global rate; here we stay purely local
    loss = -(weight * DS_k).sum(dim=1).mean() \
        + (alpha * weight * local_interf_per_ue).sum(dim=1).mean()
    return loss


def loss_function_new(client_batch, edge_attr_dict, tau, rho_d, num_antenna, kg, alpha=0.1):
    """Purely-LOCAL sum-rate surrogate for one client (AP).

    Maximise this AP's own desired signal and penalise the interference it
    creates -- using ONLY its own DS/PC/UI.  No DS/PC/UI is exchanged between
    clients (unlike the global-rate loss in the old scheme).

        loss = -sum_k DS_k  +  alpha * sum_k (PC_k + UI_k)
    """
    DS_k, PC_k, UI_k = compute_components(
        client_batch, edge_attr_dict, tau, rho_d, num_antenna)      # [B,1,K], [B,1,K,K] x2


    local_PC = PC_k.sum(dim=2) 
    local_UI = UI_k.sum(dim=2)   
    local_interf_per_ue = 1.0 * local_PC + 1 * local_UI          # [B, 1, K]

    ds_weight = kg[:, :, -2].detach().unsqueeze(-1) # [B, 1, 1]
    int_weight = kg[:, :, -1].detach().unsqueeze(-1) # [B, 1, 1]
    # marginal = kg[:, :, -3].detach().unsqueeze(-1) # [B, 1, 1]

    # net_delta = kg[:, :, -4].detach().mean(dim=1, keepdim=True).unsqueeze(-1)
    # alpha_eff = alpha * torch.exp(net_delta).clamp(0.5, 2.0)
   
    # combine DS importance
    mu = 1.0
    q = 1.0
    ds_eff = ds_weight # * marginal
    ds_eff = ds_weight / (ds_weight.mean().detach() + 1e-9)
    ds_gate = torch.exp(mu * ds_weight) * ds_eff ** q

    eta = 5.0
    p = 1.5
    int_eff = int_weight / (int_weight.mean().detach() + 1e-9)
    int_gate = torch.exp(eta * int_weight) * int_eff ** p


    loss = -(ds_gate * DS_k).sum(dim=1).mean() \
        + (alpha * int_gate  * local_interf_per_ue).sum(dim=1).mean()
    # - full_sumrate.mean()

    return loss

def train_round_old(ap_loaders, sensing_loader, M, server_model, server_opt,
                    local_models, optimizers, selected,
                    tau, rho_d, num_antenna, comm_rounds, device,
                    ctde=False, lam=0.2, use_kg=True):
    """One federated round, run as the explicit 3-phase protocol.

    Phase 1 : every client encodes its OWN local AP<->UE graph and produces an
              AP embedding (sent up to the server).  No gradients here.
    Phase 2 : the server runs its model on the AP<->SR knowledge graph, fusing
              all client AP embeddings + sensing/target context -> KG.
    Phase 3 : every (selected) client injects its KG embedding into its local
              graph, predicts the power allocation, and trains on the PURELY
              LOCAL loss_function.  Server gradients accumulate across clients;
              FedAvg of the client models is done by the caller afterwards.
    """
    for m in local_models:
        m.train()
    server_model.train()

    for batch_tuple in zip(*ap_loaders, sensing_loader):
        client_batches = [b.to(device) for b in batch_tuple[:M]]
        server_b = batch_tuple[M].to(device)                       # AP<->SR graph batch

        # ----- Phase 1: AP embeddings + DS/PC/UI (detached, sent to server) -----
        with torch.no_grad():
            outs = [local_models[i](b) for i, b in enumerate(client_batches)]
            ap_emb = torch.stack([o[0]["AP"] for o in outs], dim=1)  # [B, M, F]
            ds_all, pc_all, ui_all = phase1_components(outs, client_batches,
                                                       tau, rho_d, num_antenna)

        # ----- Phase 2 + 3 -----
        if server_opt is not None:
            server_opt.zero_grad()
        for ci in selected:
            optimizers[ci].zero_grad()

        if ctde:
            # CTDE: forward ALL M clients (with grad) so the GLOBAL coherent sum
            # rate can be assembled in-memory (no DS/PC/UI is communicated; this
            # is the centralised-training side -- deploy stays local-only).
            gap, attn = (server_model(server_b, ap_emb, ds_all, pc_all, ui_all, num_antenna)
                         if use_kg else (None, None))              # Phase 2
            DS_all, PC_all, UI_all = [], [], []
            local_loss, alpha = 0.0, 0.1
            for i in range(M):
                kg_arg, attn_i = _kg_for_client(gap, attn, i, server_model.param_free)
                _, edge_attr_dict, _ = local_models[i](client_batches[i], kg_emb=kg_arg, kg_attn=attn_i)
                DS_k, PC_k, UI_k = compute_components(client_batches[i], edge_attr_dict,
                                                      tau, rho_d, num_antenna)
                DS_all.append(DS_k); PC_all.append(PC_k); UI_all.append(UI_k)
                if i in selected:                                  # local regulariser
                    interf = PC_k.sum(dim=2) + UI_k.sum(dim=2)     # [B, 1, K]
                    local_loss = local_loss - DS_k.sum(dim=1).mean() \
                        + alpha * interf.sum(dim=1).mean()
            rate = global_sum_rate(DS_all, PC_all, UI_all, num_antenna).mean()
            loss = -rate + lam * local_loss
            loss.backward()
        else:
            # Decentralised (default): each selected client trains on its OWN
            # purely-local loss; nothing global is ever assembled.
            for ci in selected:
                if use_kg:
                    gap, attn = server_model(server_b, ap_emb, ds_all, pc_all, ui_all, num_antenna)
                    kg_arg, attn_ci = _kg_for_client(gap, attn, ci, server_model.param_free)
                else:
                    kg_arg, attn_ci = None, None                   # KG disabled (ablation)
                _, edge_attr_dict, _ = local_models[ci](
                    client_batches[ci], kg_emb=kg_arg, kg_attn=attn_ci)
                loss = loss_function_old(client_batches[ci], edge_attr_dict,
                                     tau, rho_d, num_antenna)
                # loss = loss_function_new(
                #     ci=ci,
                #     client_batches=client_batches,
                #     edge_attr_dict_ci=edge_attr_dict,
                #     ds_all=ds_all,
                #     pc_all=pc_all,
                #     ui_all=ui_all,
                #     tau=tau,
                #     rho_d=rho_d,
                #     num_antenna=num_antenna,
                #     alpha=0.0,
                # )
                loss.backward()

        if server_opt is not None:
            torch.nn.utils.clip_grad_norm_(server_model.parameters(), 1.0)
            server_opt.step()
        for ci in selected:
            torch.nn.utils.clip_grad_norm_(local_models[ci].parameters(), 1.0)
            optimizers[ci].step()


def train_round_old(
        ap_loaders, sensing_loader, M, server_model, server_opt,
        local_models, optimizers, selected,
        tau, rho_d, num_antenna, comm_rounds, device,
        ctde=False, lam=0.2, use_kg=True
    ):
    for m in local_models:
        m.train()
    server_model.train()

    for batch_tuple in zip(*ap_loaders, sensing_loader):
        client_batches = [b.to(device) for b in batch_tuple[:M]]
        server_b = batch_tuple[M].to(device)                       # AP<->SR graph batch

        # ----- Phase 1: clients build AP embeddings + DS/PC/UI (detached, sent up) -----
        # No gradients here: this is the uplink summary each client ships to the server.
        with torch.no_grad():
            outs = [local_models[i](b) for i, b in enumerate(client_batches)]
            ap_emb = torch.stack([o[0]["AP"] for o in outs], dim=1)  # [B, M, F]
            ds_all, pc_all, ui_all = phase1_components(outs, client_batches,
                                                       tau, rho_d, num_antenna)

        # ----- Phase 2: server fuses everything into the knowledge graph -----
        if server_opt is not None:
            server_opt.zero_grad()
        for ci in selected:
            optimizers[ci].zero_grad()

        # gap : per-AP KG row [B, M, F+2] (embedding + leave-one-out & full rate)
        # attn: pairwise AP->AP weights [B, M, M] (conv server) or None (param-free)
        gap, attn = (server_model(server_b, ap_emb, ds_all, pc_all, ui_all, num_antenna)
                     if use_kg else (None, None))

        # ----- Phase 3: each (selected) client conditions on the KG and trains -----
        if ctde:
            # Centralised training, decentralised execution: forward ALL M clients
            # (with grad) so the GLOBAL coherent sum rate can be assembled in memory.
            # Nothing global is communicated at deploy time -- this is the train-side only.
            DS_all, PC_all, UI_all = [], [], []
            local_loss, alpha = 0.0, 0.1
            for i in range(M):
                kg_arg, attn_i = _kg_for_client(gap, attn, i, server_model.param_free)
                _, edge_attr_dict, _ = local_models[i](
                    client_batches[i], kg_emb=kg_arg, kg_attn=attn_i)
                DS_k, PC_k, UI_k = compute_components(client_batches[i], edge_attr_dict,
                                                      tau, rho_d, num_antenna)
                DS_all.append(DS_k); PC_all.append(PC_k); UI_all.append(UI_k)
                if i in selected:                                  # local regulariser
                    interf = PC_k.sum(dim=2) + UI_k.sum(dim=2)     # [B, 1, K]
                    local_loss = local_loss - DS_k.sum(dim=1).mean() \
                        + alpha * interf.sum(dim=1).mean()
            rate = global_sum_rate(DS_all, PC_all, UI_all, num_antenna).mean()
            loss = -rate + lam * local_loss
            loss.backward()
        else:
            # Decentralised (default): each selected client trains on its OWN
            # purely-local loss; nothing global is ever assembled.
            for ci in selected:
                kg_arg, attn_ci = _kg_for_client(gap, attn, ci, server_model.param_free)
                _, edge_attr_dict, _ = local_models[ci](
                    client_batches[ci], kg_emb=kg_arg, kg_attn=attn_ci)
                loss = loss_function_old(client_batches[ci], edge_attr_dict,
                                         tau, rho_d, num_antenna)
                loss.backward()

        # ----- Optimiser step (server grads accumulate across clients; FedAvg by caller) -----
        if server_opt is not None:
            torch.nn.utils.clip_grad_norm_(server_model.parameters(), 1.0)
            server_opt.step()
        for ci in selected:
            torch.nn.utils.clip_grad_norm_(local_models[ci].parameters(), 1.0)
            optimizers[ci].step()



def train_round_new(
        ap_loaders, sensing_loader, M, server_model, server_opt,
        local_models, optimizers, selected,
        tau, rho_d, num_antenna, comm_rounds, device,
        ctde=False, lam=0.2, use_kg=True
    ):
    for m in local_models:
        m.train()
    server_model.train()

    for batch_tuple in zip(*ap_loaders, sensing_loader):
        client_batches = [b.to(device) for b in batch_tuple[:M]]
        server_b = batch_tuple[M].to(device)        


        # Phase 1: Client create initial AP embedding from local data
        with torch.no_grad():
            outs = [local_models[i](b) for i, b in enumerate(client_batches)]
            ap_emb = torch.stack([o[0]["AP"] for o in outs], dim=1)  # [B, M, F]
            ds_all, pc_all, ui_all = phase1_components(outs, client_batches,
                                                       tau, rho_d, num_antenna)
            
            ds_emb = torch.concatenate(ds_all, dim=1).sum(dim=2).unsqueeze(-1)
            pc_emb = torch.concatenate(pc_all, dim=1).sum(dim=(2,3)).unsqueeze(-1)
            ui_emb = torch.concatenate(ui_all, dim=1).sum(dim=(2,3)).unsqueeze(-1)
            ap_emb = torch.concatenate(
                [ap_emb, ds_emb, pc_emb, ui_emb],
                dim=2
            )

        # Phase 2: Server create the KG 
        if server_opt is not None:
            server_opt.zero_grad()
        for ci in selected:
            optimizers[ci].zero_grad()

        # Phase 3: train each selected client with its own local loss.
        # Recompute the server KG per client so each backward has an independent
        # autograd graph; this avoids both retain_graph=True and a large summed
        # multi-client loss.
        for ci in selected:
            if use_kg:
                gap, attn = server_model(server_b, ap_emb, ds_all, pc_all, ui_all, num_antenna)
                gap_ci = torch.cat(
                    [gap[:, :ci], gap[:, ci+1:]],
                    dim=1,
                )
                if attn is not None:
                    attn_ci = torch.cat(
                        [attn[:, ci, :ci], attn[:, ci, ci+1:]],
                        dim=1,
                    )
                    tau_attn = 2.0
                    attn_ci = torch.softmax(tau_attn * attn_ci, dim=1)
                else:
                    # Param Free path
                    # int_others = gap_ci[:, :, -1]       # [B, M-1]
                    # tau_attn = 5.0                      # thử 2, 5, 10
                    # attn_ci = torch.softmax(tau_attn * int_others.detach(), dim=1)
                    attn_ci = None


                ci_ap = gap[:, ci].unsqueeze(1)

                gap_ci = gap_ci - gap[:, ci:ci+1]
                _, edge_attr_dict, _ = local_models[ci](
                    client_batches[ci], kg_emb=gap_ci, kg_attn=attn_ci)
                loss = loss_function_new(client_batches[ci], edge_attr_dict,
                                         tau, rho_d, num_antenna, kg=ci_ap)
                # loss = loss_function_old(client_batches[ci], edge_attr_dict,
                #                          tau, rho_d, num_antenna)
            else:
                _, edge_attr_dict, _ = local_models[ci](client_batches[ci])
                loss = loss_function_old(client_batches[ci], edge_attr_dict,
                                         tau, rho_d, num_antenna)
            loss.backward()


        # ----- Optimiser step (server grads accumulate across clients; FedAvg by caller) -----
        if server_opt is not None:
            torch.nn.utils.clip_grad_norm_(server_model.parameters(), 1.0)
            server_opt.step()
        for ci in selected:
            torch.nn.utils.clip_grad_norm_(local_models[ci].parameters(), 1.0)
            optimizers[ci].step()


@torch.no_grad()
def evaluate(ap_loaders, sensing_loader, M, server_model, local_models,
             tau, rho_d, num_antenna, comm_rounds, device, use_kg=True):
    for m in local_models:
        m.eval()
    server_model.eval()

    sum_rates = []
    for batch_tuple in zip(*ap_loaders, sensing_loader):
        client_batches = [b.to(device) for b in batch_tuple[:M]]
        server_b = batch_tuple[M].to(device)                       # AP<->SR graph batch


        # Phase 1: Client create initial AP embedding from local data
        with torch.no_grad():
            outs = [local_models[i](b) for i, b in enumerate(client_batches)]
            ap_emb = torch.stack([o[0]["AP"] for o in outs], dim=1)  # [B, M, F]
            ds_all, pc_all, ui_all = phase1_components(outs, client_batches,
                                                       tau, rho_d, num_antenna)
            
            ds_emb = torch.concatenate(ds_all, dim=1).sum(dim=2).unsqueeze(-1)
            pc_emb = torch.concatenate(pc_all, dim=1).sum(dim=(2,3)).unsqueeze(-1)
            ui_emb = torch.concatenate(ui_all, dim=1).sum(dim=(2,3)).unsqueeze(-1)
            ap_emb = torch.concatenate(
                [ap_emb, ds_emb, pc_emb, ui_emb],
                dim=2
            )

        # Phase 2: Server create the KG 
        gap, attn = (server_model(server_b, ap_emb, ds_all, pc_all, ui_all, num_antenna)
                    if use_kg else (None, None))
        # gap = gap.detach()

        # Phase 3: Client models train using the local data and the KG
        DS_all, PC_all, UI_all, ap_list = [], [], [], []
        for ci, b in enumerate(client_batches): 
            if use_kg:
                gap_ci = torch.cat(
                    [gap[:, :ci], gap[:, ci+1:]],
                    dim = 1
                )
                gap_ci = gap_ci - gap[:, ci:ci+1]
                # ci_ap = gap[:, ci].unsqueeze(1)

                x_dict, edge_attr_dict, _ = local_models[ci](b, kg_emb=gap_ci)
            else:
                x_dict, edge_attr_dict, _ = local_models[ci](b)
            DS_k, PC_k, UI_k = compute_components(b, edge_attr_dict, tau, rho_d, num_antenna)
            DS_all.append(DS_k); PC_all.append(PC_k); UI_all.append(UI_k)
            ap_list.append(x_dict["AP"])
        ap_emb = torch.stack(ap_list, dim=1)                   # refine for next comm round
        sum_rates.append(global_sum_rate(DS_all, PC_all, UI_all, num_antenna))
    return torch.cat(sum_rates, dim=0)    # [num_samples]
