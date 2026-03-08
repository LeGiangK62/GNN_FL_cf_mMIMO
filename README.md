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

### Current best for max min
```bash
python FlGrad.py --num_train 200 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --cen_pretrain 01_14_19_18_18_cen --hidden_channels 128 --num_gnn_layers 5 --num_epochs 1 --num_rounds 250 --batch_size 32 --lr 1e-4  --client_fraction 0.6 --server_lr 0.05 --eval_plot --fl_scheme scaffold  --comm
_rounds 2
```

---
## Citation
Please cite my paper (To be update...)

---
## Contact

Mr. Le Tung GIANG - tung.giangle99@gmail.com or giang.lt2399144@pusan.ac.kr
