# Performance Gap Analysis: FL-GNN (1.3) vs Centralized (1.7+)

## Executive Summary
The FL-GNN is achieving **~76% of optimal performance (1.3 vs 1.7)**, indicating fundamental issues in the federated learning approach. Here are the **critical problems identified**:

---

## 1. **CRITICAL: Information Bottleneck in Server Aggregation** ⚠️

### Problem
In `server_return_GAP()` at [Utils/fl_train.py:318-380], the server computes a **global bottleneck indicator** based on **aggregated rates across all APs**:

```python
all_DS_stack = torch.stack(all_DS, dim=1)  # [B, K, M]
all_PC_stack = torch.stack(all_PC, dim=1)
all_UI_stack = torch.stack(all_UI, dim=1)
global_rate = rate_from_component(all_DS_stack, all_PC_stack, all_UI_stack, numAntenna=num_antenna)  # [B, M]
bottleneck_indicator = F.softmax(-global_rate / temperature, dim=1)  # [B, M]
```

**Why this is wrong:**
- This bottleneck indicator indicates which **AP is the bottleneck**, NOT which **UE is bottlenecked**
- Each AP then receives this indicator, but it's AP-centric information, not UE-centric
- **Each local client (AP) can only optimize its own power**, but it's receiving global bottleneck info that might not align with LOCAL optimization objectives
- The indicator conflates "which AP is contributing least to the global rate" with "which UE should this AP focus on"

### Impact
- ✗ Local clients optimize for global bottlenecks they cannot directly control
- ✗ Information misalignment between what clients optimize and what affects the system rate

---

## 2. **CRITICAL: Incomplete Information about Other APs** ⚠️

### Problem
In `get_global_info()` at [Utils/fl_train.py:166-222]:
- Each AP computes **only its own DS, PC, UI components**
- These are sent to the server for augmentation
- But in `server_return_GAP()`, other APs' information is **aggregate statistics only** (mean, max, std):

```python
edge_reshaped = other_edge.reshape(num_batch, num_ue_per_graph, num_GAP, edge_feat_dim)
edge_mean = edge_reshaped.mean(dim=1)    # Average over UEs
edge_max = edge_reshaped.max(dim=1)[0]
edge_std = edge_reshaped.std(dim=1)

edge_attr_inteference = torch.cat([edge_mean, edge_max], dim=-1).reshape(-1, edge_feat_dim*2)
```

**Why this is wrong:**
- **Summary statistics lose critical per-UE information**: The exact interference each UE experiences from each other AP
- When an AP optimizes power allocation, it needs to know: "For UE k, if I increase power to 1 Watt, how will this affect its SINR considering other APs' power choices?"
- **You're losing the ability to reason about individual user-level conflicts**

### Impact
- ✗ APs cannot fully model multi-user interference impact
- ✗ Power allocation becomes based on aggregate behavior instead of individual user bottlenecks
- ✗ This explains why local GNN training doesn't converge to global optimal

---

## 3. **CRITICAL: Loss Function Mismatch - Local vs Global Objective** ⚠️

### Problem
In `loss_function()` at [Utils/fl_train.py:21-87]:

**FL Loss (what each AP trains on):**
```python
all_DS = [DS_k] + [r['DS'] for r in clientResponse]  # My DS + Other APs' DS
all_PC = [PC_k] + [r['PC'] for r in clientResponse]  
all_UI = [UI_k] + [r['UI'] for r in clientResponse]

# Calculate rate as if ALL components are summed
rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)  # [B, K]
min_rate, _ = torch.min(rate, dim=1)
temperature = 2
min_rate = -torch.logsumexp(-rate / temperature, dim=1) * temperature  # Soft-min
loss = -min_rate.mean()
```

**Centralized Loss (what optimal training uses):**
```python
# All APs' power is optimized jointly
power_matrix = power_from_raw(power_matrix_raw, channel_var, num_antenna)
all_DS, all_PC, all_UI = component_calculate(power_matrix, ...)
rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)
min_rate, _ = torch.min(rate, dim=1)
loss = torch.mean(-min_rate)  # Hard min, not soft min!
```

**Critical Difference:**
1. **Centralized uses HARD-MIN** (line 71 in centralized_train.py)
2. **FL uses SOFT-MIN with temperature=2** (line 75 in fl_train.py)
3. Hard-min is more aggressive about maximizing the worst user
4. Soft-min is a relaxation that helps with gradients but gives up on fairness

**The deeper issue:**
- Each AP is computing `rate_from_component()` using **partial information**:
  - It knows its own DS, PC, UI
  - It receives OTHER APs' aggregated DS, PC, UI (from server)
  - **But these are not the real rates!** They're hypothetical rates as if THIS AP's power is optimized in isolation

### Impact
- ✗ Each AP's local loss doesn't align with global min-rate objective
- ✗ Soft-min relax makes optimization easier but sacrifices worst-case fairness
- ✗ **Each AP thinks it's optimizing fairly, but it's only optimizing the rate that OTHER APs report**

---

## 4. **MAJOR: Decentralized Data Split vs Centralized Training**

### Problem
In [Utils/data_gen.py:316-328]:

```python
def build_decen_loader(betaMatrix, gammaMatrix, phiMatrix, batchSize, seed=1712):
    log_large_scale = np.log1p(betaMatrix)
    data_decen = create_graph(log_large_scale, gammaMatrix, phiMatrix, 'het')  # isDecentralized=True
    loader_decen = build_loader(data_decen, batchSize, seed=seed, drop_last=False)
```

In [Utils/data_gen.py:102-105]:
```python
if isDecentralized:
    for each_AP in range(num_AP):
        data_single_AP = []  # Each AP gets samples from a SUBSET of the data
```

**What's happening:**
- **Centralized training** sees ALL APs for every channel realization
- **Decentralized training** - Each AP sees the same set of channel realizations, but only its own local channel information
- **This is NOT federated learning with non-IID data** - it's federated learning with IID data but artificial information hiding

### Why it matters:
- Both models train on the same 1000 samples
- But FL-GNN has an artificial communication constraint
- The constraint isn't helping it learn better, it's just making it worse

---

## 5. **MAJOR: Augmented UE Features - Circular Information** ⚠️

### Problem
In `server_return_GAP()` at [Utils/fl_train.py:352-371]:

```python
# Context A: Global Embedding (sum of all APs' UE embeddings except mine)
new_ue_features = ((global_ue_context - all_client_embeddings[client_id]) / (num_client - 1)).to(device)

# Context B: Contribution ratio (my signal / total signal)
local_DS = all_DS[client_id]
contribution_ratio = (local_DS / (total_DS + 1e-9)).reshape(-1, 1)

# Context C: Interference share
local_interf = (all_PC[client_id] + all_UI[client_id]).sum(dim=2)
interference_share = (local_interf / (total_Interf + 1e-9)).reshape(-1, 1)

# Context D: Global quality
global_sinr = (2 ** global_rate - 1).reshape(-1, 1)

aug_batch['UE'].x = torch.cat([
    aug_batch['UE'].x,          # Original features (pilots)
    new_ue_features,             # Other APs' embeddings
    bottleneck_indicator,        # Which AP is limiting (NOT which UE!)
    contribution_ratio,          # My contribution
    interference_share,          # My interference
    global_sinr                  # Global SINR
], dim=-1)
```

**Problems:**
1. **Context D (global_sinr)** is computed from `global_rate`, which was computed in the SAME forward pass
   - This creates a circular dependency: the model's output affects its input for next iteration
   
2. **These augmented features are NOT differentiable w.r.t. power allocation**
   - `global_sinr = 2^global_rate - 1` is computed from OTHER APs' components
   - The gradient flowing back through `global_sinr` won't help THIS AP optimize its power
   - **Only the gradient through its own DS, PC, UI will matter**

3. **Contribution ratio and interference share**: These are useful for context, but:
   - They're computed from the CURRENT power allocation
   - They don't tell the model: "should I increase or decrease power?"
   - They're reactive, not predictive

---

## 6. **MAJOR: Gradient Flow Asymmetry in Rate Calculation**

### Problem
In `rate_from_component()` at [Utils/comm.py:99-122]:

```python
def rate_from_component(desiredSignal, pilotContamination, userInterference, numAntenna, rho_d=0.1):
    sum_DS = desiredSignal.sum(dim=1)  # Sum over all APs
    num = (numAntenna**2) * (sum_DS ** 2)
    
    sum_PC = pilotContamination.sum(dim=1)
    sum_UI = userInterference.sum(dim=1)
    
    sum_PC = sum_PC ** 2
    term1 = sum_PC * (1 - torch.eye(...))  # Cross-user contamination
    term1 = (numAntenna**2) * term1.sum(dim=1)
    
    term2 = numAntenna * sum_UI.sum(dim=1)
    denom = term1 + term2 + 1
    
    rate_all = torch.log2(1 + num/denom)
    return rate_all  # [B, M]
```

**The issue:**
- When an AP computes its local rate using OTHER APs' components, it's using:
  - Its own DS (which is: its power × its channel variance)
  - Other APs' PC and UI components (which depend on OTHER APs' power choices)
  
- **Gradient flow:**
  - `∂loss/∂(my_power)` flows through `∂DS/∂(my_power)` ✓
  - `∂loss/∂(my_power)` does NOT flow through `∂PC/∂(other_AP_power)` (because PC depends on other APs' power, not mine) ✗
  
- **Result:** Each AP only optimizes its own power to maximize its own DS contribution, **without understanding how changing its power affects interference for other users or other APs' optimization landscape**

---

## 7. **MODERATE: Model Architecture Mismatch**

### Problem
- **Centralized model** [Models/GNN.py:178-305]:
  - Straightforward multi-layer GNN
  - All APs' information is available in the graph
  - Trained with full channel knowledge

- **FL model** [Models/GNN.py:308-514]:
  - Has GAP (Global AP) nodes for other APs
  - Has separate UE encoders for raw vs augmented data
  - Much more complex architecture to handle information hiding
  
**Why this matters:**
- The FL model has more capacity (more layers, more components) to compensate for information loss
- But more capacity ≠ better generalization without better information
- You're trying to solve an **inference problem** (how do other APs behave?) with a GNN that wasn't designed for that

---

## 8. **MODERATE: Training Dynamics Mismatch**

### Centralized Training:
```python
for epoch in range(num_epochs_cen):
    loss.backward()
    optimizer.step()  # Updates ALL APs' power jointly
```

### FL Training:
```python
for round in range(num_rounds):
    send_to_server()  # Each AP sends its current state
    server_return_GAP()  # Server augments with other APs' aggregate info
    
    for epoch in range(num_epochs):
        loss.backward()  # Each AP optimizes IN ISOLATION
        optimizer.step()
    
    global_weights = fed.aggregate(global_model, local_models, selected_clients)
```

**Problems:**
1. **Information staleness**: When AP i trains, it uses data from all APs at the START of the round
2. **No immediate feedback**: AP i changes its power, but AP j doesn't see this change until next round
3. **Convergence**: With soft-min loss and stale information, there's no guarantee convergence to any equilibrium

---

## Summary of Root Causes

| Issue | Severity | Impact |
|-------|----------|---------|
| Information bottleneck (global bottleneck vs local UE bottleneck) | **CRITICAL** | Each AP optimizes for wrong objective |
| Incomplete interference information (aggregate statistics) | **CRITICAL** | APs can't model multi-user conflicts |
| Loss function mismatch (soft-min vs hard-min) | **CRITICAL** | Different convergence behavior |
| Gradient flow asymmetry (only own power affects own DS) | **CRITICAL** | APs don't optimize for global fairness |
| Decentralized vs centralized data split | MAJOR | Different information availability |
| Augmented feature circular dependencies | MAJOR | Inputs depend on outputs |
| Model architecture mismatch | MODERATE | More parameters don't help without information |
| Training dynamics asynchrony | MODERATE | Information staleness causes suboptimality |

---

## Why 1.3 instead of 1.7

The performance gap comes from **compounding effects**:

1. **Each AP optimizes locally** → Less efficient power allocation
2. **Without full interference info** → Can't anticipate conflicts
3. **With soft-min loss** → Doesn't push hard on worst user
4. **With delayed feedback** → Convergence is slow
5. **All together** → 24% performance loss

---

## Recommendations for Improvement

### Quick Wins (would help immediately):
1. **Use hard-min loss instead of soft-min** in FL training
2. **Send full DS/PC/UI per UE** instead of aggregate statistics
3. **Synchronize updates more frequently** (every batch instead of every round)
4. **Initialize from centralized model** as warm-start

### Structural Changes (needed for 1.7+):
1. **Rethink information sharing**: Instead of rate-based augmentation, share explicit interference predictions
2. **Use coordinated optimization**: Design mechanism where APs understand global impact of their local changes
3. **Better loss alignment**: Ensure local loss better approximates global min-rate objective
4. **Non-convex game theory approach**: Frame as Stackelberg game or coalition formation problem

---

