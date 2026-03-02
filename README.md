# GNN_FL_cf_mMIMO

A Graph Neural Network Federated Learning Apporach for Cell-Free Massive MIMO Communication 

---

## Table of Contents

- [Requirement](#requirements)
- [Installation](#installation)
- [Citation](#citation)
- [Contact](#contact)

---
## Requirements
- CUDA 11.8
- python=3.10
- pytorch=2.0.1
- torch-geometric=2.4.0

```bash
conda create -n env_name python=3.10 cudatoolkit=11.8 -y
```

---
## Installation
### Clone repo

```bash
git clone https://github.com/LeGiangK62/GNN_FL_cf_mMIMO.git
cd GNN_FL_cf_mMIMO
```
### Install dependencies
```bash
pip install -r requirements.txt
```
---

## System scheme

┌─────────────────────────────────────────────────────────────┐
│  1. ALL APs run forward pass (same time) → get_global_info  │
│     - Each AP gets: DS, PC, UI, UE embeddings               │
│                                                             │
│  2. Server aggregates → server_return                       │
│     - Augments UE features with global context              │
│     - Returns rate_pack (other APs' DS/PC/UI)               │
│                                                             │
│  3. ALL APs train on augmented data (same time)             │
│     - Each AP only modifies ITS OWN power                   │
│     - Uses rate_pack (FROZEN) for global rate calculation   │
│                                                             │
│  4. FedAvg aggregates weights                               │
└─────────────────────────────────────────────────────────────┘


## Running command
'''bash
python FlGrad.py --num_train 2000 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --cen_pretrain 01_14_19_18_18_cen --hidden_channels 64 --num_gnn_layers 3 --num_epochs 1 --num_rounds 550 --batch_size 32 --lr 1e-3  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg

python FlGrad.py --num_train 2000 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 64 --num_gnn_layers 3 --num_epochs 1 --num_rounds 550 --batch_size 32 --lr 1e-3  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg

'''
---
## Citation
Please cite my paper (To be update...)

---
## Contact

Mr. Le Tung GIANG - tung.giangle99@gmail.com or giang.lt2399144@pusan.ac.kr


## Note

lr 5e-3 is currently the best => try 1e-2

num_gnn_layer should be 2 or 3; 4 is bad; 3 is current the best
remove the sigmoid in the power MLP en 3 layers x 64

1e-2 > 5e-1 => plateue

bat_norm using is better

4 GNN layers are too much

num_gnn_layers 3 is current optimal 

python FlGrad.py --num_train 1000 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --cen_pretrain 01_14_19_18_18_cen --hidden_channels 128 --num_gnn_layers 5 --num_epochs 1 --num_rounds 50 --
batch_size 32 --lr 1e-4  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg

###
FL GNN



system model: cf-mMIMO, K APs, serving M UEs at the same times

objective: maximize the min-rate over UEs

task: power allocation each AP to each UE, constraint of sum power budget in each AP



approach: using GNN in FL, where each AP is a client, with local data



1. raw forward, each client send a pack of data to server

2. server calculate the augmented local data for each AP, and return with DS, PC, UI (pack rate-which is depend on each client power allocation, but then can be used to calculate the real rate at each AP)

3. each client is trained on the corresponding local augmented data

=====
Each Round:
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Clients → Server (get_global_info)                                  │
│ ────────────────────────────────────────────────────────────────────────────│
│ Each client runs: model(batch, isRawData=True)                              │
│                                                                             │
│ Sends to server:                                                            │
│   • DS, PC, UI      - signal components (computed from power_raw)           │
│   • AP embedding    - node features after GNN layers                        │
│   • UE embedding    - node features after GNN layers                        │
│   • edge_down       - edge features after GNN layers                        │
│   • power_raw       - raw power output (now computed even in isRawData=True)│
│   • largeScale, channelVariance, phiMatrix - channel info                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Server Processing (server_return_GAP)                               │
│ ────────────────────────────────────────────────────────────────────────────│
│ Server computes:                                                            │
│   • global_rate = rate_from_component(all_DS, all_PC, all_UI)               │
│   • bottleneck_indicator = softmax(-global_rate / 0.001)                    │
│   • contribution_ratio = local_DS / total_DS                                │
│   • interference_share = local_interf / total_interf                        │
│   • global_sinr = 2^global_rate - 1                                         │
│                                                                             │
│ For each client, server creates augmented batch:                            │
│   • GAP nodes: other APs' embeddings                                        │
│   • GAP edges: mean/max of other APs' edge features                         │
│   • UE augmented features: [tau, global_context, bottleneck, contrib, ...]  │
│   • rate_pack: frozen DS/PC/UI from other clients                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Local Training (fl_train)                                           │
│ ────────────────────────────────────────────────────────────────────────────│
│ For each selected client:                                                   │
│   for epoch in range(num_epochs):  # default 3                              │
│     • Forward: model(augmented_batch, isRawData=False)                      │
│       - Uses GAP layers (convs_gap)                                         │
│       - Computes edge_power via power_edge MLP                              │
│     • Loss: compute rate using:                                             │
│       - Fresh DS/PC/UI from this client's new power                         │
│       - Frozen DS/PC/UI from other clients (rate_pack)                      │
│     • loss = -soft_min(rate, temperature=2)                                 │
│     • Backward + optimizer.step()                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Aggregation (FedAvg/FedAdam/FedGM)                                  │
│ ────────────────────────────────────────────────────────────────────────────│
│ • global_weights = aggregate(global_model, local_models, selected_clients)  │
│ • global_model.load_state_dict(global_weights)                              │
│ • Broadcast: all local_models ← global_model                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Evaluation (fl_eval_new)                                            │
│ ────────────────────────────────────────────────────────────────────────────│
│ • Compute batch_rates for next round's prev_rate                            │
│ • last_round_rate = batch_rates  # List[Tensor], each [B, num_UE]           │
│                                                                             │
│ If round % eval_round == 0:                                                 │
│   • total_train_rate = fl_eval(train_loader).mean().item()                  │
│   • total_eval_rate = fl_eval(test_loader).mean().item()                    │
│   • Log results                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                              Next Round