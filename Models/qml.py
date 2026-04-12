import pennylane as qml
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn.inits import reset
from torch_geometric.nn.conv import MessagePassing

from .GNN import  MLP

# ─── PQC ──────────────────────────────────────────────────────────────────────

def make_pqc(n_qubits: int, n_layers: int, q_dev):
    """
    Build a PQC using StronglyEntanglingLayers.

    Args:
        n_qubits:  number of qubits
        n_layers:  number of StronglyEntanglingLayers repetitions
        device:    PennyLane device string

    Returns:
        qnode:     a pennylane QNode (callable)
        weight_shape: dict passed to qml.qnn.TorchLayer
    """
    dev = q_dev

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    out_dim = n_qubits
    in_dim = 2**n_qubits

    weight_shape = {"weights": qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)}
    return circuit, weight_shape, in_dim, out_dim


def make_circuit_6(n_qubits: int, n_layers: int, q_dev):
    """
    Build Circuit 6: RX+RZ encoding → [RX all + ring CNOT] x n_layers → RX+RZ all.

    Weight shape: (n_layers + 2, n_qubits)
        weights[0 : n_layers]   — variational RX params per layer
        weights[n_layers]       — final RX params
        weights[n_layers + 1]   — final RZ params

    Input shape:  (n_qubits,)  — one angle per qubit (AngleEmbedding)
    Output shape: (n_qubits,)  — PauliZ expectation values
    """
    dev = q_dev

    @qml.qnode(dev, interface="torch")
    def circuit_6(inputs, weights):
        # Encoding: RX + RZ with input angles (supports batched inputs)
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")

        # Variational layers: RX rotations + ring CNOT entanglement
        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[l, i], wires=i)
            # Ring: 0→1→2→...→(n-1)→0
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])

        # Final layer: trainable RX + RZ
        for i in range(n_qubits):
            qml.RX(weights[n_layers, i], wires=i)
            qml.RZ(weights[n_layers + 1, i], wires=i)

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    in_dim = n_qubits
    out_dim = n_qubits
    weight_shape = {"weights": (n_layers + 2, n_qubits)}
    return circuit_6, weight_shape, in_dim, out_dim





# # ─── Hybrid FC + QLayer ───────────────────────────────────────────────────────

# class HybridNet(nn.Module):
#     """
#     Fully-connected network whose last layer is a quantum layer (TorchLayer).

#     Architecture:
#         input_dim  →  hidden_dim  →  n_qubits  →  [qlayer]  →  n_qubits outputs

#     Args:
#         input_dim:  size of the classical input feature vector
#         hidden_dim: width of the classical hidden layer
#         n_qubits:   number of qubits (also = output dimension)
#         n_layers:   depth of StronglyEntanglingLayers
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         n_qubits: int = 4,
#         n_layers: int = 2,
#     ):
#         super().__init__()

#         # Classical front-end: maps input to n_qubits values in [-π, π]
#         self.classical = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, n_qubits),
#             nn.Tanh(),                          # keeps values ∈ (-1,1); scale below
#         )
#         self.input_scale = nn.Parameter(
#             torch.tensor(torch.pi), requires_grad=False
#         )

#         # Quantum back-end
#         circuit, weight_shape = make_pqc(n_qubits, n_layers)
#         self.qlayer = qml.qnn.TorchLayer(circuit, weight_shape)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch, input_dim)
#         z = self.classical(x) * self.input_scale   # scale to (-π, π)
#         out = self.qlayer(z)                        # (batch, n_qubits) ∈ (-1, 1)
#         return out


# --- GNN + QML

class APConvLayer_qml(MessagePassing):
    def __init__(
            self,
            src_dim_dict,
            edge_dim,
            out_channel,
            init_channel,
            metadata,
            q_dev, n_qubits, n_layers,
            drop_p=0,
            **kwargs
    ):
        super().__init__(aggr='add', **kwargs)
        self.metadata = metadata
        self.src_init_dict = init_channel
        self.edge_init = init_channel['edge']
        self.out_channel = out_channel
        self.src_dim_dict = src_dim_dict
        self.drop_prob = drop_p

        self.msg = nn.ModuleDict() 
        self.upd = nn.ModuleDict() 
        self.edge_upd = nn.ModuleDict() 
        
        self.gamma = nn.ParameterDict()
        self.gamma_edge = nn.ParameterDict()
        
        hidden = out_channel//2
        for edge_type in metadata:
            src_type, short_edge_type, dst_type = edge_type
            src_dim = src_dim_dict[src_type]
            dst_dim = src_dim_dict[dst_type]
            src_init = init_channel[src_type]
            dst_init = init_channel[dst_type]
            # self.msg[src_type] = MLP(
            #     [src_dim + edge_dim, hidden], 
            #     batch_norm=False, dropout_prob=0.1
            # )


            # Change the type of circuit here 
            # TODO: Try the Circuit 6 here
            circuit, weight_shape, in_dim, out_dim = make_circuit_6(n_qubits, n_layers, q_dev)
            
            self.msg[src_type] = Seq(
                MLP(
                    [src_dim + edge_dim, in_dim],
                    batch_norm=False, dropout_prob=0.1
                ),
                qml.qnn.TorchLayer(circuit, weight_shape)
            )


            self.upd[dst_type] = MLP(
                [out_dim + dst_dim, out_channel - dst_init], 
                batch_norm=False, dropout_prob=0.1
            )
            
            self.edge_upd[short_edge_type] = MLP(
                [sum(src_dim_dict.values()) + edge_dim, out_channel - self.edge_init], 
                batch_norm=False, dropout_prob=0.1
            )

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.msg)
        reset(self.upd)
        reset(self.edge_upd)

    def forward(
            self,
            x_dict,
            edge_index_dict,
            edge_attr_dict
    ):
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type not in self.metadata: continue
            src_type, _, dst_type = edge_type

            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]
            
            edge_attr = edge_attr_dict[edge_type]
            

            # Node update                
            msg = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, edge_type=edge_type)
            tmp = torch.cat([x_dst, msg], dim=1)
            tmp = self.upd[dst_type](tmp)
            src_init_dim = self.src_init_dict[dst_type]
            if self.src_dim_dict[dst_type] == self.out_channel:
                tmp = tmp +  x_dst[:,src_init_dim:] # * self.gamma[dst_type]
            x_dict[dst_type] = torch.cat([x_dst[:,:src_init_dim], tmp], dim=1)
            # Edge update
            edge_attr_dict[edge_type] = self.edge_updater(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, edge_type=edge_type)

        return x_dict, edge_attr_dict

    def message(self, x_j, x_i, edge_attr, edge_type):
        # x_j: source node
        # x_i: destination node
        src_type, _, dst_type = edge_type
        out = torch.cat([x_j, edge_attr], dim=1)
        out = self.msg[src_type](out)
        return out

    def edge_update(self, x_j, x_i, edge_attr, edge_type):
        _, short_edge_type, _ = edge_type
        tmp = torch.cat([x_j, edge_attr, x_i], dim=1)
        out = self.edge_upd[short_edge_type](tmp)
        
        if self.out_channel == self.edge_init:
            out = out + edge_attr # * self.gamma_edge
        out = torch.cat([edge_attr[:,:self.edge_init], out], dim=1)
        return out
    

class APHetNetFL_qml(nn.Module):
    def __init__(self, metadata, dim_dict, out_channels, 
                 q_dev, n_qubits, n_layers,
                 aug_feat_dim=3, num_layers=0, hid_layers=4, 
                 isDecentralized=False,):
        super(APHetNetFL_qml, self).__init__()

        GAP_init_dim = out_channels + 0# + 3
        GAP_edge_init_dim = out_channels * 2 # * 3
        GAP_UE_edge = out_channels
        src_dim_dict = dim_dict.copy()
        

        self.ue_dim = src_dim_dict['UE']
        self.ue_dim_aug = src_dim_dict['UE'] + out_channels  + aug_feat_dim
        self.ap_dim = src_dim_dict['AP']
        self.edge_dim = src_dim_dict['edge']

        self.out_channels = out_channels

        ##
        src_dim_dict_gap = dim_dict.copy()
        src_dim_dict_gap['GAP'] = 0
        src_dim_dict_gap['AP'] = src_dim_dict['AP']
        src_dim_dict_gap['edge'] = 0

        ### UE <-> AP
        self.convs_pre = torch.nn.ModuleList()
            
        self.convs_pre.append(APConvLayer_qml(
            {'UE': out_channels, 'AP': out_channels}, 
            self.edge_dim,
            out_channels, src_dim_dict,
            [('UE', 'up', 'AP')],
            q_dev=q_dev, n_qubits=n_qubits, n_layers=n_layers,
        ))

        self.convs_pre.append(APConvLayer_qml(
            {'UE': out_channels, 'AP': out_channels}, 
            self.edge_dim,
            out_channels, src_dim_dict,
            [('AP','down','UE')],
            q_dev=q_dev, n_qubits=n_qubits, n_layers=n_layers,
        ))

        #####

        ### GAP <-> AP
        self.convs_gap = torch.nn.ModuleList()

        self.convs_gap.append(APConvLayer_qml(
            {'GAP': GAP_init_dim, 'AP': out_channels}, 
            GAP_edge_init_dim,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP')],
            q_dev=q_dev, n_qubits=n_qubits, n_layers=n_layers,
        ))

        self.convs_gap.append(APConvLayer_qml(
            {'GAP': GAP_init_dim, 'AP': out_channels }, 
            GAP_edge_init_dim,
            out_channels, src_dim_dict_gap,
            [('AP', 'cross-back', 'GAP')],
            q_dev=q_dev, n_qubits=n_qubits, n_layers=n_layers,
        ))
        
        # for _ in range(num_layers): # too many layers break the model
        self.convs_gap.append(APConvLayer_qml(
            {'GAP': out_channels, 'AP': out_channels }, 
            out_channels,
            out_channels, src_dim_dict_gap,
            [('GAP', 'cross', 'AP'), ('AP', 'cross-back', 'GAP')],
            q_dev=q_dev, n_qubits=n_qubits, n_layers=n_layers,
        ))

        ### AP -> UE, then AP <-> UE
        self.convs_post = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.convs_post.append(APConvLayer_qml(
                {'UE': out_channels, 'AP': out_channels}, 
                out_channels, out_channels, src_dim_dict, 
                [('UE', 'up', 'AP'), ('AP', 'down', 'UE')],
                q_dev=q_dev, n_qubits=n_qubits, n_layers=n_layers,
            ))

        
        hid = hid_layers # too much is not good - 8 is bad, 4 is currently good
        
        # self.ue_encoder_raw = MLP([self.ue_dim, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
        self.ue_encoder_raw = MLP([self.ue_dim, hid, self.ue_dim_aug], batch_norm=True, dropout_prob=0.1) 
        self.ue_encoder_aug = MLP([self.ue_dim_aug, hid, out_channels - self.ue_dim], batch_norm=True, dropout_prob=0.1) 
        self.ap_encoder_raw = MLP([self.ap_dim, hid, out_channels], batch_norm=True, dropout_prob=0.1) 
        self.power_edge = MLP([out_channels, hid], batch_norm=True, dropout_prob=0.1) #  many layer => shit
        self.power_edge = nn.Sequential(
            *[
                self.power_edge, Seq(Lin(hid, 1)), 
            ]
        )

    
        
    def forward(self, batch, isRawData=False):
        x_dict, edge_index_dict, edge_attr_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict


        aug_ap = self.ap_encoder_raw(x_dict['AP'] )
        x_dict['AP'] = aug_ap
        tmp = x_dict['UE'] 
        if isRawData:
            tmp = self.ue_encoder_raw(x_dict['UE'] )
        aug_ue = self.ue_encoder_aug(tmp)

        x_dict['UE'] = torch.cat(
            [x_dict['UE'] [:,:self.ue_dim], aug_ue], 
            dim=1
        )

        # UE -> AP
        for conv in self.convs_pre:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        if not isRawData:
            for conv in self.convs_gap:
                x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        # AP <-> UE
        for conv in self.convs_post:
            x_dict, edge_attr_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

        edge_power = self.power_edge(edge_attr_dict[('AP', 'down', 'UE')])
        edge_attr_dict[('AP', 'down', 'UE')] = torch.cat(
            [edge_attr_dict[('AP', 'down', 'UE')][:,:-1], edge_power], 
            dim=1
        )

        return x_dict, edge_attr_dict, edge_index_dict
    


# ─── Quick smoke-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch, input_dim, hidden_dim, n_qubits, n_layers = 8, 16, 32, 4, 2

    model = HybridNet(input_dim, hidden_dim, n_qubits, n_layers)
    print(model)

    x = torch.randn(batch, input_dim)
    y = model(x)
    print("output shape:", y.shape)   # (8, 4)
    print("output sample:\n", y)
