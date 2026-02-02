# Visual Reference: Code Locations of Each Issue

## Issue 1: Soft-Min vs Hard-Min Loss

**Location:** `Utils/fl_train.py` lines 71-76

```
CENTRALIZED (optimal):                    FL-GNN (suboptimal):
─────────────────────────────────────────────────────────────
Utils/centralized_train.py:30-32          Utils/fl_train.py:71-76
                                          
min_rate, _ = torch.min(rate, dim=1)      temperature = 2
loss = torch.mean(-min_rate)              min_rate = -torch.logsumexp(
                                              -rate / temperature, dim=1
✓ Hard-min: aggressively fixes                ) * temperature
  worst case                              loss = -min_rate.mean()

                                          ✗ Soft-min: forgives worst case
                                          ✗ Temperature=2 is a relaxation
```

**File:** [Utils/fl_train.py](Utils/fl_train.py#L71-L76)

---

## Issue 2: Bottleneck Indicator (AP vs UE)

**Location:** `Utils/fl_train.py` lines 318-319

```
WRONG (Current):                          RIGHT (Should be):
─────────────────────────────────────────────────────────────
all_DS_stack = [B, K_AP, K_UE]           all_DS_stack = [B, K_AP, K_UE]
all_PC_stack = [B, K_AP, K_UE, K_UE]     all_PC_stack = [B, K_AP, K_UE, K_UE]
all_UI_stack = [B, K_AP, K_UE, K_UE]     all_UI_stack = [B, K_AP, K_UE, K_UE]

global_rate = rate_from_component(       sum_DS = all_DS_stack.sum(dim=1)
    all_DS_stack, ...)                   # [B, K_UE]
# [B, K_AP] ← WRONG!                     
                                         global_rate_per_ue = \
bottleneck_indicator =                     rate_from_component(...)
  F.softmax(-global_rate/temp, dim=1)    # [B, K_UE] ← RIGHT!
# [B, K_AP] - which AP is bad             
                                         bottleneck_indicator = \
✗ Indicates which AP is limiting           F.softmax(-global_rate_per_ue...)
✗ Not which UE is suffering               # [B, K_UE] - which UE is bad
                                          
                                          ✓ Indicates which UE is limiting
                                          ✓ Each AP can focus on helping
                                          ✓ the right UEs
```

**File:** [Utils/fl_train.py](Utils/fl_train.py#L318-L319)

---

## Issue 3: Aggregated Interference Information

**Location:** `Utils/fl_train.py` lines 349-357

```
CURRENT (Aggregate Statistics):           BETTER (Full Information):
─────────────────────────────────────────────────────────────────────
other_edge shape:                         other_edge shape:
[B*K_UE, K_AP-1, edge_feat_dim]          [B*K_UE, K_AP-1, edge_feat_dim]

edge_reshaped.mean(dim=1)                 # Don't aggregate - keep as:
edge_reshaped.max(dim=1)[0]               # [B*K_UE, K_AP-1, edge_feat_dim]
edge_reshaped.std(dim=1)                  
                                          # Or create UE-GAP edges:
→ [num_batch * num_GAP,                   # edge_index_ue_gap = [ue_id, gap_id]
    edge_feat_dim*2]                      # edge_attr_ue_gap = 
                                          #     [channel_to_that_gap]

✗ Loses per-UE structure                  ✓ Preserves which UE sees
✗ UE k doesn't know it's in               ✓ how much interference from
  "mean interference regime"               ✓ which other AP
```

**File:** [Utils/fl_train.py](Utils/fl_train.py#L349-L357)

---

## Issue 4: Gradient Flow Asymmetry

**Location:** `Utils/comm.py` lines 99-122 & `Utils/fl_train.py` lines 21-87

```
GRADIENT FLOW DIAGRAM:

loss ← min_rate ← rate_from_component(DS, PC, UI)
                   │
       ┌───────────┼───────────┐
       │           │           │
       ↓           ↓           ↓
    [DS_all]    [PC_all]    [UI_all]
       │           │           │
       ├─ DS_k ────┤           │  (From own AP's power)
       │   (mine)  │           │
       │           │           │
       └─ DS_j ────┤           │  (From other APs' power)
           (others)│           │
                   │
                   PC_j, UI_j (from other APs' power)
                   │
                   ✗ These have ∂PC/∂(my_power) = 0
                   ✗ Only my DS affects gradient
                   
∴ Gradient only flows through my DS term
∴ I don't learn how my power affects others' interference
∴ I don't optimize for global fairness
```

**File:** 
- [Utils/comm.py](Utils/comm.py#L99-L122) - `rate_from_component()`
- [Utils/fl_train.py](Utils/fl_train.py#L21-L87) - `loss_function()`

---

## Issue 5: Information Staleness

**Location:** `Utils/fl_train.py` lines 104-128 & `FlGrad.py` lines 280-320

```
TIMELINE PER ROUND:

┌─ Round t:
│
├─ T1: send_to_server()
│      all_DS[i], all_PC[i], all_UI[i]
│      ↑ Computed with model weights w_t
│
├─ T2: server_return_GAP()
│      Augment batch with T1 info
│      ↑ Uses OLD information
│
├─ T3: for epoch in range(num_epochs):
│      x_dict, attr = model(batch)
│      loss = loss_function(x_dict, attr, all_DS[i], ...)
│                                         ↑
│                                         Still using T1 info!
│      loss.backward()
│      optimizer.step()
│      ↑ Model weights change: w_t → w_t+ε
│      ↑ But loss still computed with w_t's DS/PC/UI
│
└─ T4: global_weights = aggregate()
       
       
PROBLEM: After first local epoch, model weights changed,
but we're still training with T1's cached components.

By epoch 3: weights = w_t + 3ε, but we're using loss from w_t

SOLUTION: Recompute DS/PC/UI locally at each epoch
```

**File:**
- [Utils/fl_train.py](Utils/fl_train.py#L104-L128) - `fl_train()` function
- [FlGrad.py](FlGrad.py#L280-L320) - training loop

---

## Issue 6: Circular Augmented Features

**Location:** `Utils/fl_train.py` lines 318-371

```
CIRCULAR DEPENDENCY:

model.forward(batch)
    ↓
    power_matrix_raw (output: what I allocate)
    
send_to_server()
    ↓ (using my power_matrix_raw)
    all_DS[0], all_PC[0], all_UI[0]
    
server_return_GAP()
    ↓ (using all power allocations)
    global_rate = rate_from_component(...)
    global_sinr = 2^global_rate - 1
    bottleneck_indicator = F.softmax(-global_rate, ...)
    
aug_batch['UE'].x = cat([
    original_ue_features,
    bottleneck_indicator,    ← from global_rate (THIS sample!)
    global_sinr             ← from global_rate (THIS sample!)
])

model.forward(aug_batch)  ← Input now depends on previous output!
    ↓
    power_matrix_raw (new output)
    
✗ The features fed back are from THIS SAME sample
✗ They don't inform how to change power for NEXT sample
✗ Adding them to input doesn't improve generalization
```

**File:** [Utils/fl_train.py](Utils/fl_train.py#L318-L371)

---

## Issue 7: Loss Function Inconsistency

**Location:** `Utils/fl_train.py` lines 60-87

```
WHAT GETS COMPUTED:
                
┌─────────────────────────────────────────┐
│ rate = rate_from_component(...) [B, K]  │
└─────────────────────────────────────────┘
        │           │
        ├──────┬────┘
        ↓      ↓
    HARD-MIN  SOFT-MIN
    (monitored) (used for loss)
    
    min_rate_detach, _ = torch.min(...)
    ↑ Line 70: computed but not used
    
    temperature = 2
    min_rate = -torch.logsumexp(...)
    ↑ Line 75: THIS is the actual loss
    
    loss = -min_rate.mean()
    ↑ Line 76: soft-min loss


PROBLEM: 
├─ Training optimizes soft-min
├─ Evaluation measures hard-min
├─ They're not the same objective!
├─ min_rate_detach is computed but wasted
└─ No alignment between train and eval


COMPARE TO CENTRALIZED [centralized_train.py:30-32]:

min_rate, _ = torch.min(rate, dim=1)
loss = torch.mean(-min_rate)

✓ Trains on hard-min
✓ Evaluates on hard-min
✓ Consistent objectives
```

**File:**
- [Utils/fl_train.py](Utils/fl_train.py#L60-L87) - FL loss
- [Utils/centralized_train.py](Utils/centralized_train.py#L30-L32) - Centralized loss

---

## Summary Table: Where to Look

| Issue | File | Lines | What to Change |
|-------|------|-------|-----------------|
| 1. Soft-min vs hard-min | `Utils/fl_train.py` | 71-76 | Replace soft-min with hard-min |
| 2. AP vs UE bottleneck | `Utils/fl_train.py` | 318-319 | Compute per-UE bottleneck |
| 3. Aggregated interference | `Utils/fl_train.py` | 349-357 | Keep per-UE structure |
| 4. Gradient asymmetry | `Utils/comm.py` | 99-122 | Redesign rate calculation |
| 5. Stale information | `Utils/fl_train.py` | 104-128 | Recompute per epoch |
| 6. Circular features | `Utils/fl_train.py` | 371 | Remove or fix circular dependencies |
| 7. Model architecture | `Models/GNN.py` | 308-514 | Simplify for decentralized case |

---

## How These Issues Compound

```
┌─────────────────────────────────────────┐
│ Issue 1: Soft-min Loss                  │
│ (Each AP thinks fairness is less imp.)  │
└──────────────┬──────────────────────────┘
               │
               ├──→ Lower min-rate optimization
               │
┌──────────────┴──────────────────────────┐
│ Issue 2: Wrong Bottleneck Dimension     │
│ (Each AP focuses on wrong metric)       │
└──────────────┬──────────────────────────┘
               │
               ├──→ Power allocation doesn't fix actual UE bottlenecks
               │
┌──────────────┴──────────────────────────┐
│ Issue 3: Aggregate Info Loss            │
│ (Each AP doesn't know who needs help)   │
└──────────────┬──────────────────────────┘
               │
               ├──→ Can't target power allocation efficiently
               │
┌──────────────┴──────────────────────────┐
│ Issue 4: Gradient Flow Asymmetry        │
│ (Each AP doesn't learn interference)    │
└──────────────┬──────────────────────────┘
               │
               ├──→ Power allocation doesn't account for interference
               │
┌──────────────┴──────────────────────────┐
│ Issue 5: Stale Information              │
│ (AP trains on outdated assumptions)     │
└──────────────┬──────────────────────────┘
               │
               ├──→ Local optimal ≠ global optimal
               │
               ↓
           1.3 (76% of optimal)
```

---

