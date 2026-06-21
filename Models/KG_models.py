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

    gap = [ embedding(F) | rate_wo(1) | full_rate(1)
            | sense_full(1) | sense_marg(1)            # sensing context (CRB)
            | contribution_ratio(1) | interference_share(1) ]  -> F+6.

    The two comm shares are kept LAST so ``loss_function_new`` keeps reading
    them at indices -2/-1; the sensing dims are pure context fed through the
    client's ``kg_proj`` (they are NOT used as loss weights).
    """
    return feat_dim + 6


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

        self.kg_scale = nn.Parameter(torch.tensor(0.1))

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
                if kg_attn is not None:
                    ctx = (kg_attn.unsqueeze(-1) * msg).sum(dim=1)   # [B, F]
                else:
                    ctx = msg.mean(dim=1)   # [B, F] # mean
            else:                                       # [B, F+2] this AP's own gap row
                ctx = self.kg_proj(kg_emb)              # [B, F]

            x_dict["AP"] = x_dict["AP"] + self.kg_scale * ctx
            # x_dict["AP"] = x_dict["AP"] + ctx

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
        self.ap_emb_proj = Lin(F+3, F)

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

    @staticmethod
    def _sensing(q):
        """Target-localization CRB signals from per-AP raw Fisher entries.

        Each AP m contributes a 2x2 Fisher Information Matrix for the target
        position, ``J_m = [[q_a, q_c], [q_c, q_b]]``.  The team FIM is the sum
        over APs; the localization CRB (position-error bound, LOWER = better) is
        ``trace(J^-1) = (A+B) / (A*B - C^2)`` with ``A = sum_m q_a`` etc.

            q : [B, M, 3]  raw (>=0) (q_a, q_b, q_c) per AP.

        Returns ``crb_wo`` [B, M] (leave-one-out CRB: localization WITHOUT AP m)
        and ``crb_full`` [B] (CRB with all APs).  This is the sensing analogue of
        ``_rates``: ``crb_wo - crb_full`` is how much worse the localization gets
        if AP m drops out, i.e. AP m's sensing criticality.
        """
        eps = 1e-9
        qa, qb, qc = q[..., 0], q[..., 1], q[..., 2]                          # [B, M]
        A = qa.sum(dim=1); B = qb.sum(dim=1); C = qc.sum(dim=1)              # [B]
        det = (A * B - C ** 2).clamp_min(eps)
        crb_full = (A + B) / det                                             # [B]
        # leave-one-out: drop AP m's Fisher contribution
        A_wo = A.unsqueeze(1) - qa                                           # [B, M]
        B_wo = B.unsqueeze(1) - qb
        C_wo = C.unsqueeze(1) - qc
        det_wo = (A_wo * B_wo - C_wo ** 2).clamp_min(eps)
        crb_wo = (A_wo + B_wo) / det_wo                                      # [B, M]
        return crb_wo, crb_full

    def forward(self, server_batch, ap_emb, ds, pc, ui, num_antenna):
        B, M, _ = ap_emb.shape
        F = self.feat_dim

        ds_client = ap_emb[:,:,-3:-2]
        pc_client = ap_emb[:,:,-2:-1]
        ui_client = ap_emb[:,:,-1:]

        # rate signals the server hands back (detached features describing state)
        rate_wo, full = self._rates(ds, pc, ui, num_antenna, M)               # [B,M], [B]
        rate_wo_e = rate_wo.unsqueeze(-1)                                     # [B, M, 1]
        full_e = full.view(B, 1, 1).expand(B, M, 1)      
        # marginal = (full_e - rate_wo_e).clamp_min(0.0)           # [B, M, 1]
        # marginal = marginal / (marginal.mean(dim=1, keepdim=True) + 1e-9)
        
        # gap = torch.cat([ha, rate_wo_e, full_e], dim=-1)                     # [B, M, F+2]
        eps = 1e-9
        local_DS = ds_client                              # [B, K, 1]
        total_DS = local_DS.sum(dim=1, keepdim=True)      # [B, 1, 1]

        contribution_ratio = local_DS / (total_DS + eps)  # [B, K, 1]
        

        local_interf = pc_client + ui_client
        total_Interf = local_interf.sum(dim=1, keepdim=True)
        interference_share = (local_interf / (total_Interf + eps))                    # [B, M, 1]
        net_utility = contribution_ratio - interference_share

        # ---- sensing CONTEXT (not a loss weight): localization CRB from FIM ----
        # The raw Fisher entries (q_a,q_b,q_c) ride on the AP nodes; CRB is a
        # function of geometry+RCS only (independent of the power decision), so
        # it is injected as a context feature the client's kg_proj can use, NOT
        # as a multiplicative weight on the comm loss.
        q = server_batch["AP"].q_raw.view(B, M, 3)                          # [B, M, 3]
        crb_wo, crb_full = self._sensing(q)                                 # [B, M], [B]
        dcrb = (crb_wo - crb_full.unsqueeze(1)).clamp_min(0.0)              # [B, M] AP sensing criticality
        sense_full = torch.log1p(crb_full).view(B, 1, 1).expand(B, M, 1)   # [B, M, 1] global, log-compressed
        sense_marg = torch.log1p(dcrb).unsqueeze(-1)                        # [B, M, 1] per-AP, log-compressed

        if self.param_free:
            emb = ap_emb[..., :F]                                            # pass-through
            # NOTE: comm shares kept LAST so loss_function_new reads them at -2/-1.
            gap = torch.cat([emb, sense_full, sense_marg, rate_wo_e, full_e, 
                            #  net_utility, 
                            #  marginal, 
                             contribution_ratio, interference_share], dim=-1)   # [B, M, F+6]
            return gap, None

        x_dict = server_batch.x_dict
        edge_index_dict = server_batch.edge_index_dict
        edge_attr_dict = server_batch.edge_attr_dict

        x_dict["AP"] = self.ap_enc(x_dict["AP"]) + self.ap_emb_proj(ap_emb.reshape(B * M, F + 3))
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

        

        # comm shares kept LAST so loss_function_new reads them at -2/-1.
        gap = torch.cat([ha, sense_full, sense_marg, rate_wo_e, full_e, 
                        #  net_utility, 
                        #  marginal, 
                         contribution_ratio, interference_share], dim=-1)            # [B, M, F+6]
        return gap, attn
