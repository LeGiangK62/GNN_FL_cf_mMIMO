import torch
import numpy as np
import scipy.io
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData


# =============================================================================
# 1. ISAC .mat loader
# =============================================================================

def load_isac_mat(mat_path):
    """
    Load an ISAC sumrate .mat file produced by run_gen_isac_sumrate.m.

    Expected fields (mandatory):
        betas        : [N, M, K]
        Gammas       : [N, M, K]
        Phii_cf      : [N, tau, K]    (transposed to [N, K, tau] on return)
        R_equal      : [1, N]
        R_frac       : [1, N]
        R_log        : [1, N]
        power_eq     : [N, M, K]
        power_frac   : [N, M, K]
        power_log    : [N, M, K]

    Optional sensing fields (loaded if present, else returned as None):
        rcs_values   : [N, M, T]
        sr_locations : [T, 2]
        ap_locations : [M, 2]
        target_location : [N, 2]
        distances_ap_target : [N, M]
        distances_sr_target : [N, T]
        crlb_values  : [N, 2]
    """
    mat = scipy.io.loadmat(mat_path)

    out = {
        'betas':      mat['betas'],
        'gammas':     mat['Gammas'],
        'phi':        mat['Phii_cf'].transpose(0, 2, 1),    # -> [N, K, tau]
        'R_equal':    mat['R_equal'][0],
        'R_frac':     mat['R_frac'][0],
        'R_log':      mat['R_log'][0],
        'power_eq':   mat['power_eq'],
        'power_frac': mat['power_frac'],
        'power_log':  mat['power_log'],
    }

    # Optional sensing payload — currently not saved by run_gen_isac_sumrate.m,
    # but supported here so the loader works once it is added.
    for key in ['rcs_values', 'sr_locations', 'ap_locations',
                'target_location', 'distances_ap_target', 'distances_sr_target',
                'crlb_values']:
        out[key] = mat[key] if key in mat else None

    return out


# =============================================================================
# 2. Heterogeneous graph builder
# =============================================================================

def create_graph_isac(
        Beta_all, Gamma_all, Phi_all,
        RCS_all=None, ap_cor_all=None, sr_cor_all=None,
        isDecentralized=True,
    ):
    """
    Wrap full_het_graph_isac over (sample, AP) indices.

    Beta_all, Gamma_all : [N, M, K]
    Phi_all             : [N, K, tau]
    RCS_all             : [N, M, T] or None
    ap_cor_all          : [N, M, 2]
    sr_cor_all          : [N, T, 2]
    """
    num_sample, num_AP, num_UE = Beta_all.shape
    data_list = []

    if isDecentralized:
        for each_AP in range(num_AP):
            data_single_AP = []
            for each_sample in range(num_sample):
                rcs_s = RCS_all[each_sample, each_AP][np.newaxis, :] if RCS_all is not None else None
                ap_cor_s = ap_cor_all[each_sample, each_AP:each_AP+1] if ap_cor_all is not None else None
                sr_cor_s = sr_cor_all[each_sample] if sr_cor_all is not None else None

                data = full_het_graph_isac(
                    Beta_all[each_sample, each_AP][np.newaxis, :],
                    Gamma_all[each_sample, each_AP][np.newaxis, :],
                    Phi_all[each_sample],
                    rcs_single=rcs_s,
                    ap_cor=ap_cor_s,
                    sr_cor=sr_cor_s,
                    ap_id=each_AP,
                    sample_id=each_sample,
                )
                data_single_AP.append(data)
            data_list.append(data_single_AP)
    else:
        for each_sample in range(num_sample):
            rcs_s = RCS_all[each_sample] if RCS_all is not None else None
            ap_cor_s = ap_cor_all[each_sample] if ap_cor_all is not None else None
            sr_cor_s = sr_cor_all[each_sample] if sr_cor_all is not None else None

            data = full_het_graph_isac(
                Beta_all[each_sample],
                Gamma_all[each_sample],
                Phi_all[each_sample],
                rcs_single=rcs_s,
                ap_cor=ap_cor_s,
                sr_cor=sr_cor_s,
            )
            data_list.append(data)
    return data_list


def full_het_graph_isac(
        beta_single_sample, gamma_single_sample, phi_single_sample,
        rcs_single=None, ap_cor=None, sr_cor=None,
        ap_id=None, sample_id=None,
    ):
    """
    Build a HeteroData object for ISAC cell-free mMIMO.

    Communication side mirrors data_gen.full_het_graph:
        Nodes:  AP [num_AP, 1], UE [num_UE, tau]
        Edges:  AP --down--> UE      with attr [beta, gamma]
                UE --up--> AP        with attr [beta, gamma]

    Sensing side (only attached if rcs_single is provided):
        Nodes:  SR [num_SR, 1]
                Target [1, 2]   (target location features placeholder)
        Edges:  AP --senses--> SR       with attr [rcs]
                SR --sensed_by--> AP    with attr [rcs]
        Plus AP/SR distance-to-target as node features if provided.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_AP, num_UE = beta_single_sample.shape

    # ---------- AP / UE nodes ----------
    ap_features = np.ones((num_AP, 1), dtype=np.float32)
    # Append target distance to AP feature  -> [num_AP, 2]
    ap_features = np.concatenate(
        [ap_features, np.asarray(ap_cor, dtype=np.float32)],
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
    data['AP', 'down', 'UE'].edge_index = edge_index_ap_down_ue
    data['AP', 'down', 'UE'].edge_attr  = edge_attr_ap_to_ue
    data['UE', 'up', 'AP'].edge_index = edge_index_ue_up_ap
    data['UE', 'up', 'AP'].edge_attr  = edge_attr_ue_up_ap

    # ---------- Sensing receivers (optional) ----------
    rcs_mat = np.asarray(rcs_single, dtype=np.float32)   # [num_AP, num_SR]
    assert rcs_mat.shape[0] == num_AP, \
        f"RCS first dim {rcs_mat.shape[0]} must match num_AP {num_AP}"
    num_SR = rcs_mat.shape[1]

    sr_features = np.ones((num_SR, 1), dtype=np.float32)
    if sr_cor is not None:
        sr_features = np.concatenate(
            [sr_features, np.asarray(sr_cor, dtype=np.float32)],
            axis=1,
        )
    x_sr = torch.tensor(sr_features, dtype=torch.float32).to(device)

    # AP -> SR (sense) and SR -> AP (sensed_by)
    edge_index_ap_sr = []
    edge_index_sr_ap = []
    for ap_idx in range(num_AP):
        for sr_idx in range(num_SR):
            edge_index_ap_sr.append([ap_idx, sr_idx])
    for sr_idx in range(num_SR):
        for ap_idx in range(num_AP):
            edge_index_sr_ap.append([sr_idx, ap_idx])

    edge_index_ap_sr = torch.tensor(edge_index_ap_sr, dtype=torch.long).t().contiguous().to(device)
    edge_index_sr_ap = torch.tensor(edge_index_sr_ap, dtype=torch.long).t().contiguous().to(device)

    rcs_ap_sr = rcs_mat.reshape(-1, 1)            # [num_AP*num_SR, 1]
    rcs_sr_ap = rcs_mat.T.reshape(-1, 1)          # [num_SR*num_AP, 1]
    edge_attr_ap_sr = torch.tensor(rcs_ap_sr, dtype=torch.float32).to(device)
    edge_attr_sr_ap = torch.tensor(rcs_sr_ap, dtype=torch.float32).to(device)

    data['SR'].x = x_sr
    data['AP', 'senses',    'SR'].edge_index = edge_index_ap_sr
    data['AP', 'senses',    'SR'].edge_attr  = edge_attr_ap_sr
    data['SR', 'sensed_by', 'AP'].edge_index = edge_index_sr_ap
    data['SR', 'sensed_by', 'AP'].edge_attr  = edge_attr_sr_ap

    data.ap_id = ap_id
    data.sample_id = sample_id
    return data


# =============================================================================
# 3. DataLoader builders
# =============================================================================

def build_loader(per_ap_datasets, batch_size, seed, drop_last=True, num_workers=0):
    n = len(per_ap_datasets[0])
    assert all(len(ds) == n for ds in per_ap_datasets), "All AP datasets must have same length."
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()

    loaders = []
    for ds in per_ap_datasets:
        subset = Subset(ds, order)
        loaders.append(DataLoader(
            subset, batch_size=batch_size, shuffle=False,
            drop_last=drop_last, num_workers=num_workers,
        ))
    return loaders


def build_cen_loader_isac(
        betaMatrix, gammaMatrix, phiMatrix, batchSize,
        rcsMatrix=None, ap_coordination=None, sr_coordination=None,
        isShuffle=False,
    ):
    log_large_scale = np.log1p(betaMatrix)
    data_cen = create_graph_isac(
        log_large_scale, gammaMatrix, phiMatrix,
        RCS_all=rcsMatrix,
        ap_cor_all=ap_coordination,
        sr_cor_all=sr_coordination,
        isDecentralized=False,
    )
    loader_cen = DataLoader(data_cen, batch_size=batchSize, shuffle=isShuffle)
    return data_cen, loader_cen


def build_decen_loader_isac(
        betaMatrix, gammaMatrix, phiMatrix, batchSize,
        rcsMatrix=None, ap_coordination=None, sr_coordination=None,
        seed=1712,
    ):
    log_large_scale = np.log1p(betaMatrix)
    data_decen = create_graph_isac(
        log_large_scale, gammaMatrix, phiMatrix,
        RCS_all=rcsMatrix,
        ap_cor_all=ap_coordination,
        sr_cor_all=sr_coordination,
        isDecentralized=True,
    )
    loader_decen = build_loader(data_decen, batchSize, seed=seed, drop_last=False)
    return data_decen, loader_decen
