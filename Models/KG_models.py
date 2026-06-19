import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin

from Models.GNN import APConvLayer, IsacConvLayer, MLP
from Utils.comm import rate_from_component


def global_sum_rate(all_DS, all_PC, all_UI, num_antenna):
    """Coherent global rate from per-AP components (lists of [B,1,...])."""
    DS = torch.cat(all_DS, dim=1)        # [B, M, K]
    PC = torch.cat(all_PC, dim=1)        # [B, M, K, K]
    UI = torch.cat(all_UI, dim=1)        # [B, M, K, K]
    rate = rate_from_component(DS, PC, UI, num_antenna)   # [B, K]
    return rate.sum(dim=1)                                # [B]





def server_gap_dim(feat_dim):
    """Width of each AP node in the broadcast KG.

    gap = [ embedding(F) | rate_without_this_AP(1) | full_global_rate(1) ] -> F+2.
    """
    return feat_dim + 2


# feature widths of the server AP<->SR graph (kept in sync between the graph
# builder `make_server_graph` and `ServerGNN`)
SERVER_AP_FEAT_DIM = 8   # ap_xy(2) + q_a,q_b,q_c(3) + dir-to-target(2) + log-dist(1)
SERVER_SR_FEAT_DIM = 5   # sr_xy(2) + dir-to-target(2) + log-dist(1)
SERVER_TAR_FEAT_DIM = 2  # normalised target coordinates



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
        # projects each KG AP-embedding before the server-attention aggregation.
        self.kg_proj = MLP([server_gap_dim(out_channels), hid, out_channels],
                           batch_norm=True, dropout_prob=0.1)

        self.power_edge = nn.Sequential(
            MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1),
            Seq(Lin(hid, 1)),
        )

    def forward(self, batch, kg_emb=None, kg_attn=None):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        x_dict["AP"] = self.ap_encoder(x_dict["AP"])
        aug_ue = self.ue_encoder(x_dict["UE"])
        x_dict["UE"] = torch.cat([x_dict["UE"][:, :self.ue_dim], aug_ue], dim=1)

        # Inject the broadcast KG into this client's AP node.  Two cases:
        #  * full KG [B, M, F+2] + attention -> server-attention aggregation;
        #  * own row [B, F+2]               -> direct injection (param-free server).
        if kg_emb is not None:
            if kg_emb.dim() == 3:                       # [B, M, F+2] + kg_attn [B, M]
                msg = self.kg_proj(kg_emb)              # [B, M, F]
                ctx = (kg_attn.unsqueeze(-1) * msg).sum(dim=1)   # [B, F]
            else:                                       # [B, F+2] this AP's own gap row
                ctx = self.kg_proj(kg_emb)              # [B, F]
            x_dict["AP"] = x_dict["AP"] + ctx

        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        for conv in self.convs_post:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        edge_power = self.power_edge(edge_attr_dict[("AP", "comm_down", "UE")])
        edge_attr_dict[("AP", "comm_down", "UE")] = torch.cat(
            [edge_attr_dict[("AP", "comm_down", "UE")][:, :-1], edge_power], dim=1
        )
        return x_dict, edge_attr_dict, edge_index_dict


SERVER_EDGE_TYPES = [
    ("AP", "sens_down", "SR"), ("SR", "sens_up", "AP"),
    ("AP", "ap2tar", "TARGET"), ("TARGET", "tar2ap", "AP"),
    ("SR", "sr2tar", "TARGET"), ("TARGET", "tar2sr", "SR"),
]


class ServerGNN(nn.Module):
    """Server KG model.  Two modes (``param_free``):

      * ``param_free=False`` (default) -- a heterogeneous GNN (``IsacConvLayer``)
        over the AP / SR / TARGET graph + a learned pairwise AP->AP attention.
      * ``param_free=True`` -- NO conv / attention; the per-AP embedding is the
        client embedding passed through.

    In BOTH modes the server uses the per-AP DS/PC/UI shipped up to compute the
    GLOBAL coherent sum rate (``global_sum_rate``) and, per AP, the leave-one-out
    rate (rate WITHOUT that AP), and appends both to each AP's KG row:
        gap[m] = [ embedding(F) | rate_without_m(1) | full_rate(1) ]   ([B, M, F+2])
    Each client later extracts these to gauge its marginal contribution.

        forward(server_batch, ap_emb, ds, pc, ui, num_antenna) -> (gap, attn)
            attn is [B, M, M] (conv mode) or None (param_free mode).
    """

    def __init__(self, feat_dim, sensing_dim=5, sig_dim=0, hidden=None,
                 num_layers=2, kg_edge_mode="inner", param_free=False):
        super().__init__()
        F = feat_dim
        hid = hidden or feat_dim
        self.feat_dim = F
        self.num_layers = num_layers
        self.param_free = param_free
        if param_free:
            return                                                # no learnable modules

        self.ap_enc = MLP([SERVER_AP_FEAT_DIM, hid, F], batch_norm=True, dropout_prob=0.1)
        self.sr_enc = MLP([SERVER_SR_FEAT_DIM, hid, F], batch_norm=True, dropout_prob=0.1)
        self.tar_enc = MLP([SERVER_TAR_FEAT_DIM, hid, F], batch_norm=True, dropout_prob=0.1)
        self.ap_emb_proj = Lin(F, F)

        node = {"AP": F, "SR": F, "TARGET": F}
        init = {"AP": 0, "SR": 0, "TARGET": 0}
        rels = [r for _, r, _ in SERVER_EDGE_TYPES]
        edge_init = {r: 1 for r in rels}
        edge_dim_raw = {r: 1 for r in rels}
        edge_dim_hid = {r: F for r in rels}
        self.convs_pre = nn.ModuleList([
            IsacConvLayer(node, edge_dim_raw, F, init, edge_init, SERVER_EDGE_TYPES),
        ])
        self.convs_post = nn.ModuleList([
            IsacConvLayer(node, edge_dim_hid, F, init, edge_init, SERVER_EDGE_TYPES)
            for _ in range(num_layers)
        ])
        self.out = MLP([F, hid, F], batch_norm=True, dropout_prob=0.1)
        # pairwise AP->AP attention: sigmoid( Linear([h_src, h_dst]) )
        self.attn_lin = Lin(2 * F, 1)

    @staticmethod
    def _rates(ds, pc, ui, num_antenna, M):
        """Global rate and per-AP leave-one-out rate from per-AP DS/PC/UI lists.

        ds/pc/ui : lists of M tensors.  -> rate_wo [B, M], full [B].
        """
        full = global_sum_rate(ds, pc, ui, num_antenna)                       # [B]
        rate_wo = torch.stack([
            global_sum_rate(ds[:m] + ds[m + 1:], pc[:m] + pc[m + 1:],
                            ui[:m] + ui[m + 1:], num_antenna)                  # rate w/o AP m
            for m in range(M)
        ], dim=1)                                                             # [B, M]
        return rate_wo, full

    def forward(self, server_batch, ap_emb, ds, pc, ui, num_antenna):
        B, M, _ = ap_emb.shape
        F = self.feat_dim

        # rate signals the server hands back (detached features describing state)
        rate_wo, full = self._rates(ds, pc, ui, num_antenna, M)               # [B,M], [B]
        rate_wo_e = rate_wo.unsqueeze(-1)                                     # [B, M, 1]
        full_e = full.view(B, 1, 1).expand(B, M, 1)                          # [B, M, 1]

        if self.param_free:
            emb = ap_emb[..., :F]                                            # pass-through
            gap = torch.cat([emb, rate_wo_e, full_e], dim=-1)               # [B, M, F+2]
            return gap, None

        x_dict = server_batch.x_dict
        edge_index_dict = server_batch.edge_index_dict
        edge_attr_dict = server_batch.edge_attr_dict

        x_dict["AP"] = self.ap_enc(x_dict["AP"]) + self.ap_emb_proj(ap_emb.reshape(B * M, F))
        x_dict["SR"] = self.sr_enc(x_dict["SR"])
        x_dict["TARGET"] = self.tar_enc(x_dict["TARGET"])

        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
        for conv in self.convs_post:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        ha = self.out(x_dict["AP"]).view(B, M, F)                            # conv KG, [B, M, F]

        # pairwise src->dst attention from the KG embeddings
        hi = ha.unsqueeze(2).expand(B, M, M, F)
        hj = ha.unsqueeze(1).expand(B, M, M, F)
        attn = torch.sigmoid(self.attn_lin(torch.cat([hi, hj], dim=-1))).squeeze(-1)  # [B, M, M]

        gap = torch.cat([ha, rate_wo_e, full_e], dim=-1)                     # [B, M, F+2]
        return gap, attn