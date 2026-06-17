import torch
import torch.nn as nn
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, LayerNorm, LeakyReLU
from torch.utils.data import Subset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GeoLoader

from Models.GNN import APConvLayer, MLP

from Utils.comm import power_from_raw, component_calculate, rate_from_component

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


# feature widths of the server AP<->SR graph (kept in sync between the graph
# builder `make_server_graph` and `ServerGNN`)
SERVER_AP_FEAT_DIM = 8   # ap_xy(2) + q_a,q_b,q_c(3) + dir-to-target(2) + log-dist(1)
SERVER_SR_FEAT_DIM = 5   # sr_xy(2) + dir-to-target(2) + log-dist(1)


def make_server_graph(ap_cor, sr_cor, rcs, zeta,
                      ap_mean, ap_std, q_mean, q_std, tar_cor=None):
    """One sample's server-side AP<->SR bipartite graph (target-aware).

    Nodes:
        AP : [M, 8]  (normalised AP coords, standardised q_a/q_b/q_c,
                      unit direction to target, log distance to target)
        SR : [T, 5]  (normalised SR coords, unit direction to target,
                      log distance to target)
    Edges:
        AP --sens_down--> SR   attr = rcs
        SR --sens_up-->   AP   attr = rcs

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

    data = HeteroData()
    data["AP"].x = torch.tensor(ap_feat, dtype=torch.float32)   # [M, SERVER_AP_FEAT_DIM]
    data["SR"].x = torch.tensor(sr_feat, dtype=torch.float32)   # [T, SERVER_SR_FEAT_DIM]

    # fully connected AP<->SR bipartite edges, weighted by rcs
    ap_sr = [[a, s] for a in range(M) for s in range(T)]       # (a, s) row-major
    sr_ap = [[s, a] for s in range(T) for a in range(M)]
    data["AP", "sens_down", "SR"].edge_index = torch.tensor(ap_sr, dtype=torch.long).t().contiguous()
    data["SR", "sens_up", "AP"].edge_index = torch.tensor(sr_ap, dtype=torch.long).t().contiguous()
    rcs_mat = np.asarray(rcs, dtype=np.float32)               # [M, T]
    data["AP", "sens_down", "SR"].edge_attr = torch.tensor(rcs_mat.reshape(-1, 1), dtype=torch.float32)
    data["SR", "sens_up", "AP"].edge_attr = torch.tensor(rcs_mat.T.reshape(-1, 1), dtype=torch.float32)
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

def smlp(in_dim, out_dim, hid=None):
    """Small MLP ending in a non-linearity (server dense message passing)."""
    hid = hid or out_dim
    return Seq(
        Lin(in_dim, hid), LayerNorm(hid), LeakyReLU(0.1),
        Lin(hid, out_dim), LeakyReLU(0.1),
    )


class ClientGNN(nn.Module):
    """Local AP<->UE GNN held by one client (AP).

    forward(batch, kg_emb=None):
        kg_emb is None  -> raw pass; returns the AP embedding to send up.
        kg_emb given    -> KG-conditioned pass; predicts the power allocation.
    The AP embedding (x_dict['AP']) is returned in both modes.
    """

    def __init__(self, dim_dict, out_channels, num_layers=3, hid_layers=32):
        super().__init__()
        self.ue_dim = dim_dict["UE"]
        self.ap_dim = dim_dict["AP"]
        self.comm_edge_dim = dim_dict["comm_edge"]
        self.out_channels = out_channels

        init_comm = {"UE": self.ue_dim, "AP": self.ap_dim, "edge": self.comm_edge_dim}
        node = {"UE": out_channels, "AP": out_channels}

        # UE -> AP, then AP -> UE (raw edge width grows to out_channels)
        self.convs_pre = nn.ModuleList([
            APConvLayer(node, self.comm_edge_dim, out_channels, init_comm,
                        [("UE", "comm_up", "AP")]),
            APConvLayer(node, self.comm_edge_dim, out_channels, init_comm,
                        [("AP", "comm_down", "UE")]),
        ])
        # joint AP<->UE refinement (edges are out_channels wide now)
        self.convs_post = nn.ModuleList([
            APConvLayer(node, out_channels, out_channels, init_comm,
                        [("UE", "comm_up", "AP"), ("AP", "comm_down", "UE")])
            for _ in range(num_layers)
        ])

        hid = hid_layers
        self.ue_encoder = MLP([self.ue_dim, hid, out_channels - self.ue_dim],
                              batch_norm=True, dropout_prob=0.1)
        self.ap_encoder = MLP([self.ap_dim, hid, out_channels],
                              batch_norm=True, dropout_prob=0.1)
        # projects the broadcast knowledge-graph embedding into the AP node
        self.kg_proj = MLP([out_channels, hid, out_channels],
                           batch_norm=True, dropout_prob=0.1)

        self.power_edge = nn.Sequential(
            MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1),
            Seq(Lin(hid, 1)),
        )

    def forward(self, batch, kg_emb=None):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        x_dict["AP"] = self.ap_encoder(x_dict["AP"])
        aug_ue = self.ue_encoder(x_dict["UE"])
        x_dict["UE"] = torch.cat([x_dict["UE"][:, :self.ue_dim], aug_ue], dim=1)

        ## TODO: inject the shared knowledge-graph context into the local AP node
        # if kg_emb is not None:
        #     x_dict["AP"] = x_dict["AP"] + self.kg_proj(kg_emb)

        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        for conv in self.convs_post:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        edge_power = self.power_edge(edge_attr_dict[("AP", "comm_down", "UE")])
        edge_attr_dict[("AP", "comm_down", "UE")] = torch.cat(
            [edge_attr_dict[("AP", "comm_down", "UE")][:, :-1], edge_power], dim=1
        )
        return x_dict, edge_attr_dict, edge_index_dict


class ServerGNN(nn.Module):
    """Trainable knowledge-graph GNN held by the server.

    Input  : the server-side AP<->SR bipartite graph (``make_server_graph``)
             + the AP embeddings shipped up by the clients.
        * AP nodes  : sensing/target features (server-owned) FUSED with the
                      client AP embedding.
        * SR nodes  : sensing-receiver features (server-owned only).
        * AP--SR edges carry the rcs weights.
    Message passing : AP -> SR -> AP (rcs-weighted) couples every AP through the
             shared sensing receivers / target, then an AP <-> AP step lets the
             APs coordinate directly.  The result is the shared knowledge graph
             over AP nodes that is broadcast back to the clients.
    Output : refined AP embeddings, [B, M, F].

        forward(server_batch, ap_emb)
            server_batch : batched HeteroData (B graphs, M AP + T SR each)
            ap_emb       : [B, M, F]  client AP embeddings
            return       : [B, M, F]  refined AP (knowledge-graph) embeddings

    NOTE: all sub-modules are materialised at construction (no LazyLinear), so
    an optimizer built right after the constructor tracks every parameter.
    """

    def __init__(self, feat_dim, sensing_dim=5, hidden=None, num_layers=2):
        super().__init__()
        F = feat_dim
        hid = hidden or feat_dim
        self.num_layers = num_layers

        # encoders: server node features -> F, and a projection for the client AP embedding
        self.ap_enc = smlp(SERVER_AP_FEAT_DIM, F, hid)
        self.sr_enc = smlp(SERVER_SR_FEAT_DIM, F, hid)
        self.ap_emb_proj = Lin(F, F)

        # per-layer message / update blocks (edge attr = rcs, width 1)
        self.msg_as = nn.ModuleList([smlp(F + 1, F, hid) for _ in range(num_layers)])  # AP -> SR
        self.upd_s = nn.ModuleList([smlp(F + F, F, hid) for _ in range(num_layers)])
        self.msg_sa = nn.ModuleList([smlp(F + 1, F, hid) for _ in range(num_layers)])  # SR -> AP
        self.upd_a = nn.ModuleList([smlp(F + F, F, hid) for _ in range(num_layers)])
        self.msg_aa = nn.ModuleList([smlp(F + F, F, hid) for _ in range(num_layers)])  # AP <-> AP
        self.upd_aa = nn.ModuleList([smlp(F + F, F, hid) for _ in range(num_layers)])
        self.out = smlp(F, F, hid)

    def forward(self, server_batch, ap_emb):
        B, M, F = ap_emb.shape
        eps = 1e-9

        ap_x = server_batch["AP"].x.view(B, M, -1)                       # [B, M, 8]
        T = server_batch["SR"].x.shape[0] // B
        sr_x = server_batch["SR"].x.view(B, T, -1)                       # [B, T, 5]
        rcs = server_batch["AP", "sens_down", "SR"].edge_attr.view(B, M, T)
        rcs_e = rcs.unsqueeze(-1)                                        # [B, M, T, 1]

        # fuse server sensing features with the client AP embeddings
        ha = self.ap_enc(ap_x) + self.ap_emb_proj(ap_emb)               # [B, M, F]
        hs = self.sr_enc(sr_x)                                          # [B, T, F]

        for l in range(self.num_layers):
            # AP -> SR  (rcs-weighted aggregation over APs)
            ha_e = ha.unsqueeze(2).expand(B, M, T, F)
            m_as = self.msg_as[l](torch.cat([ha_e, rcs_e], dim=-1))      # [B, M, T, F]
            agg_s = (m_as * rcs_e).sum(1) / (rcs.sum(1).unsqueeze(-1) + eps)   # [B, T, F]
            hs = hs + self.upd_s[l](torch.cat([hs, agg_s], dim=-1))

            # SR -> AP  (rcs-weighted aggregation over SRs)
            hs_e = hs.unsqueeze(1).expand(B, M, T, F)
            m_sa = self.msg_sa[l](torch.cat([hs_e, rcs_e], dim=-1))      # [B, M, T, F]
            agg_a = (m_sa * rcs_e).sum(2) / (rcs.sum(2).unsqueeze(-1) + eps)   # [B, M, F]
            ha = ha + self.upd_a[l](torch.cat([ha, agg_a], dim=-1))

            # AP <-> AP  (direct coordination -> the shared knowledge graph)
            hi = ha.unsqueeze(2).expand(B, M, M, F)
            hj = ha.unsqueeze(1).expand(B, M, M, F)
            m_aa = self.msg_aa[l](torch.cat([hi, hj], dim=-1))           # [B, M, M, F]
            agg_aa = m_aa.mean(2)                                        # [B, M, F]
            ha = ha + self.upd_aa[l](torch.cat([ha, agg_aa], dim=-1))

        return self.out(ha)                                             # [B, M, F]


# =============================================================================
# Rate maths
# =============================================================================

def compute_components(batch, edge_attr_dict, tau, rho_d, num_antenna):
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
                tau, rho_d, num_antenna, comm_rounds, device):
    """Compatibility entry point used by main_new.py -- runs the 3-phase round.

    FedAvg aggregation is performed by the caller (main_new.py) AFTER this
    returns, so it is intentionally NOT done here.
    """
    train_round_new(ap_loaders, sensing_loader, M, server_model, server_opt,
                    local_models, optimizers, selected,
                    tau, rho_d, num_antenna, comm_rounds, device)


@torch.no_grad()
def evaluate(ap_loaders, sensing_loader, M, server_model, local_models,
             tau, rho_d, num_antenna, comm_rounds, device):
    for m in local_models:
        m.eval()
    server_model.eval()

    sum_rates = []
    for batch_tuple in zip(*ap_loaders, sensing_loader):
        client_batches = [b.to(device) for b in batch_tuple[:M]]
        server_b = batch_tuple[M].to(device)                       # AP<->SR graph batch

        # ----- Phase 1: each client encodes its own local AP<->UE graph -----
        ap_emb = torch.stack(
            [local_models[i](b)[0]["AP"] for i, b in enumerate(client_batches)],
            dim=1,
        )                                                          # [B, M, F]

        # ----- Phase 2 + 3: server KG, then each client predicts its power -----
        kg = server_model(server_b, ap_emb)                    # Phase 2
        DS_all, PC_all, UI_all, ap_list = [], [], [], []
        for i, b in enumerate(client_batches):                 # Phase 3
            x_dict, edge_attr_dict, _ = local_models[i](b, kg_emb=kg[:, i, :])
            DS_k, PC_k, UI_k = compute_components(b, edge_attr_dict, tau, rho_d, num_antenna)
            DS_all.append(DS_k); PC_all.append(PC_k); UI_all.append(UI_k)
            ap_list.append(x_dict["AP"])
        ap_emb = torch.stack(ap_list, dim=1)                   # refine for next comm round

        # achieved metric IS the true global (coherent) sum rate across all APs
        sum_rates.append(global_sum_rate(DS_all, PC_all, UI_all, num_antenna))
    return torch.cat(sum_rates, dim=0)                            # [num_samples]


def loss_function(client_batch, edge_attr_dict, tau, rho_d, num_antenna, alpha=0.1):
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

def train_round_new(ap_loaders, sensing_loader, M, server_model, server_opt,
                    local_models, optimizers, selected,
                    tau, rho_d, num_antenna, comm_rounds, device):
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

        # ----- Phase 1: local AP embeddings (detached, sent to server) -----
        with torch.no_grad():
            ap_emb = torch.stack(
                [local_models[i](b)[0]["AP"] for i, b in enumerate(client_batches)],
                dim=1,
            )                                                      # [B, M, F]
            # optional extra comm rounds: refine embeddings through the server KG
            # for _ in range(max(0, comm_rounds - 1)):
            #     kg = server_model(server_b, ap_emb)
            #     ap_emb = torch.stack(
            #         [local_models[i](b, kg_emb=kg[:, i, :])[0]["AP"]
            #          for i, b in enumerate(client_batches)],
            #         dim=1,
            #     )

        # ----- Phase 2 + 3: server KG, then each client trains locally -----
        server_opt.zero_grad()
        for ci in selected:
            optimizers[ci].zero_grad()

        for ci in selected:
            kg = server_model(server_b, ap_emb)                    # Phase 2 (grad -> server)
            _, edge_attr_dict, _ = local_models[ci](
                client_batches[ci], kg_emb=kg[:, ci, :])           # Phase 3 (grad -> client ci)
            loss = loss_function(client_batches[ci], edge_attr_dict,
                                 tau, rho_d, num_antenna)          # purely local objective
            loss.backward()

        torch.nn.utils.clip_grad_norm_(server_model.parameters(), 1.0)
        server_opt.step()
        for ci in selected:
            torch.nn.utils.clip_grad_norm_(local_models[ci].parameters(), 1.0)
            optimizers[ci].step()