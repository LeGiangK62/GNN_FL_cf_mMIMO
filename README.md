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
## Data generation

for Downlink sumrate: running run_gen_sumrate.m
requirement: 
---dl_approx_sumrate.m
---dl_fractional_pa.m
---dl_rate_calculate.m
---dl_sinr_calculate.m
---dl_sinr_component_calculate.m
---downlink_sumrate_data.m
---log_approximation.m


## Running command
'''bash
python FlGrad.py --num_train 2000 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --cen_pretrain 01_14_19_18_18_cen --hidden_channels 64 --num_gnn_layers 3 --num_epochs 1 --num_rounds 550 --batch_size 32 --lr 1e-3  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg

python FlGrad.py --num_train 2000 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 64 --num_gnn_layers 3 --num_epochs 1 --num_rounds 550 --batch_size 32 --lr 1e-3  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg

'''

### Current best for max min
```bash
python FlGrad.py --num_train 200 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --cen_pretrain 01_14_19_18_18_cen --hidden_channels 128 --num_gnn_layers 5 --num_epochs 1 --num_rounds 250 --batch_size 32 --lr 1e-4  --client_fraction 0.6 --server_lr 0.05 --eval_plot --fl_scheme scaffold  --comm
_rounds 2

For qml of the strongly ent
python main_sumrate_qml.py --num_train 100 --num_test 500 --num_eval 1000 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 128 --num_gnn_layers 5   --num_epochs 1 --num_rounds 150 --batch_size 32 --lr 5e-4  --client_fraction 0.6 --server_lr 0.05 --eval_plot --fl_scheme fedavg  --comm_rounds 2 --alpha 0.2 --n_qubits 5 --n_layers 3
```

---
## Citation
Please cite my paper (To be update...)

---
## Contact

Mr. Le Tung GIANG - tung.giangle99@gmail.com or giang.lt2399144@pusan.ac.kr


## Todos
### Sumrate: 
### How to not using the DS/PC/UI in calculating the loss
### Currently, only local rate -> not goods


## Current best for sumrate
Save FL GNN to .results/sumrate/models//26_03_17_11_27_15_fl.pth.
Evaluation====================
Sum rate avg: GNN 11.92 - FL GNN 11.68 - 97.98%
```bash
python main_sumrate.py --num_train 500 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 128 --num_gnn_layers 5  --num_epochs 1 --num_rounds 150 --batch_size 32 --lr 1e-4  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg  --comm_rounds 2 --alpha 0.2 --cen_pretrain 26_05_20_15_00_28_cen --fl_pretrain 26_05_20_15_00_28_fl

python main_sumrate.py --num_train 500 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 64 --num_gnn_layers 3  --num_epochs 1 --num_rounds 150 --batch_size 32 --lr 1e-4  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg  --comm_rounds 2 --alpha 0.2 --cen_pre_train 26_05_20_16_21_01_cen --fl_pretrain 26_05_20_16_21_01_fl

.results/sumrate/models//26_05_20_21_08_24_fl.pth also 98%
```


# ISAC

```bash
python main_ISAC.py --num_train 500 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 64 --num_gnn_layers 3  --num_epochs 1 --num_rounds 150 --batch_size 32 --lr 1e-4  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg  --comm_rounds 2 --alpha 0.2 --fl_pretrain 26_05_21_14_50_23_fl


# No other sensing part in loss function - only local observation
python main_ISAC.py --num_train 500 --num_test 500 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 64 --num_gnn_layers 3  --num_epochs 1 --num_rounds 150 --batch_size 32 --lr 1e-4  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg  --comm_rounds 2 --alpha 0.2 --fl_pretrain 26_05_22_11_27_05_fl

python main_ISAC.py --num_train 500 --num_test 100 --num_eval 500 --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 64 --num_gnn_layers 3  --num_epochs 1 --num_rounds 150 --batch_size 32 --lr 1e-4  --client_fraction 1.0 --server_lr 0.05 --eval_plot --fl_scheme fedavg  --comm_rounds 2 --alpha 0.2  --fl_pretrain 26_05_23_15_17_12_fl --cen_pretrain 26_05_23_15_17_12_cen 

```

python main_new.py  --num_train 500 --num_test 100 --num_eval 500  --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1  --hidden_channels 64 --num_gnn_layers 3  --num_rounds 100 --num_epochs 1 --batch_size 32 --lr 1e-4 --client_fraction 0.6  --fl_scheme fedavg  --param_free --fl_pretrain 26_06_29_14_21_57_fl 

## Training
python main_new.py  --num_train 500 --num_test 50 --num_eval 450   --tau 20 --power_f 0.2 --num_antenna 1  --hidden_channels 64 --num_gnn_layers 3  --num_rounds 100 --num_epochs 1 --batch_size 32 --lr 1e-4 --client_fraction 0.6  --fl_scheme fedavg  --param_free --no_kg --num_ap 100 --num_ue 15

## --num_ap 30 --num_ue 6
--fl_pretrain  26_06_29_14_21_57_fl
--cen_pretrain 26_07_07_15_42_24_cen
--noKG_pretrain 26_07_07_11_55_17_fl

## --num_ap 30 --num_ue 10
--fl_pretrain  26_07_07_15_27_50_fl
--cen_pretrain 26_07_07_15_27_50_cen
--noKG_pretrain 26_07_07_16_27_19_fl

## --num_ap 50 --num_ue 6
--fl_pretrain  26_07_07_15_36_33_fl
--cen_pretrain 26_07_07_15_36_33_cen
--noKG_pretrain 26_07_07_16_43_36_fl

--fl_pretrain  26_07_09_11_09_27_fl
--cen_pretrain 26_07_09_11_09_27_cen
--noKG_pretrain 26_07_09_11_09_35_fl

## --num_ap 50 --num_ue 10
--fl_pretrain  26_07_07_18_01_25_fl
--cen_pretrain 26_07_07_17_03_58_cen
--noKG_pretrain 26_07_07_17_03_58_fl

## --num_ap 30 --num_ue 15
--fl_pretrain  26_07_08_10_02_11_fl
--cen_pretrain 26_07_08_10_02_11_cen
--noKG_pretrain 26_07_08_10_02_08_fl

## --num_ap 50 --num_ue 15 !!!!!
--fl_pretrain  26_07_07_17_14_15_fl
--cen_pretrain 26_07_07_16_50_10_cen
--noKG_pretrain 26_07_07_16_50_10_cen
--num_ap 50 --num_ue 15  --fl_pretrain 26_07_09_10_39_22_fl --cen_pretrain 26_07_09_10_39_22_cen --noKG_pretrain 26_07_09_10_39_05_fl 
## --num_ap 100 --num_ue 6
--fl_pretrain  26_07_08_16_32_01_fl
--cen_pretrain 26_07_08_16_32_01_cen
--noKG_pretrain 26_07_08_20_45_31_fl

## --num_ap 100 --num_ue 10
--fl_pretrain  26_07_08_14_08_36_fl
--cen_pretrain 26_07_08_13_03_55_cen
--noKG_pretrain 26_07_08_13_03_55_fl

## --num_ap 100 --num_ue 15
--fl_pretrain  26_07_08_10_36_15_fl
--cen_pretrain 26_07_08_10_36_15_cen
--noKG_pretrain 26_07_08_10_36_09_fl

## Evaluating


python main_new.py --num_train 10 --num_test 40 --num_eval 950 --tau 20 --power_f 0.2 --num_antenna 1 --hidden_channels 64 --num_gnn_layers 3 --batch_size 32 --param_free --filetype pdf --latex_table \
--fl_pretrain   26_07_09_13_12_24_fl --cen_pretrain  26_07_09_13_12_24_cen --noKG_pretrain 26_07_09_13_17_34_fl --num_ap 30 --num_ue 6  



python main_new.py  --num_train 1000 --num_test 100 --num_eval 500  --num_ap 30 --num_ue 6 --tau 20 --power_f 0.2 --num_antenna 1  --hidden_channels 64 --num_gnn_layers 3  --num_rounds 100 --num_epochs 1 --batch_size 32 --lr 1e-4 --client_fraction 0.6  --fl_scheme fedavg  --param_free --no_kg

--fl_pretrain   26_07_09_13_12_24_fl
--cen_pretrain  26_07_09_13_12_24_cen
--noKG_pretrain 26_07_09_13_17_34_fl


--fl_pretrain   26_07_09_14_11_12_fl
--cen_pretrain  26_07_09_14_11_12_cen
--noKG_pretrain 26_07_09_14_11_14_fl