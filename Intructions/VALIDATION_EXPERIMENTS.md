# Validation Checklist: How to Confirm These Issues

## Quick Validation Experiments

### Experiment 1: Verify Hard-Min vs Soft-Min Impact

**Hypothesis:** The soft-min loss is the main culprit - centralized uses hard-min, FL uses soft-min.

**Test:**
1. In `Utils/fl_train.py`, line 73-76, change:
```python
# CURRENT (soft-min)
temperature = 2
min_rate = -torch.logsumexp(-rate / temperature, dim=1) * temperature
loss = -min_rate.mean()

# TEST (hard-min)
min_rate, _ = torch.min(rate, dim=1)
loss = -min_rate.mean()
```

2. Run: `python FlGrad.py --num_rounds 50 --fl_scheme fedavg --eval_plot`

3. **Expected result:** If hard-min alone fixes it, FL-GNN should jump from 1.3 → ~1.5+

**Why:** Hard-min is more aggressive about worst-case fairness. If we see improvement, soft-min was indeed masking the issue.

---

### Experiment 2: Verify Information Bottleneck Impact

**Hypothesis:** Each AP is optimizing for the wrong bottleneck (AP-level instead of UE-level).

**Test:**
In `server_return_GAP()`, line 318-319, change:

```python
# CURRENT: What AP is bottleneck?
global_rate = rate_from_component(all_DS_stack, all_PC_stack, all_UI_stack, numAntenna=num_antenna)  # [B, K_AP]
bottleneck_indicator = F.softmax(-global_rate / 0.001, dim=1)

# TEST: What UEs are bottleneck?
# Compute per-UE rates by summing over APs first
sum_DS = all_DS_stack.sum(dim=1)  # [B, K_UE]
sum_PC = all_PC_stack.sum(dim=1)  
sum_UI = all_UI_stack.sum(dim=1)
global_rate_per_ue = rate_from_component(
    sum_DS.unsqueeze(1),  # Reshape to [B, 1, K_UE] for broadcasting
    sum_PC.unsqueeze(1),
    sum_UI.unsqueeze(1),
    numAntenna=num_antenna
)  # [B, K_UE]
bottleneck_indicator = F.softmax(-global_rate_per_ue / 0.001, dim=1)  # Per-UE focus
```

2. Run same command as Experiment 1

3. **Expected result:** If this is the issue, FL-GNN should improve (smaller improvement expected than hard-min fix)

---

### Experiment 3: Verify Information Staleness

**Hypothesis:** FL-GNN performance suffers because DS/PC/UI are recomputed only once per round, not per epoch.

**Test:**
In `fl_train()` function at [fl_train.py:104-128], recompute components per epoch:

```python
def fl_train(
        dataLoader, responseInfo, model, optimizer,
        tau, rho_p, rho_d, num_antenna, round_ratio=0
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    total_loss = 0.0
    total_min_rate = 0.0
    total_graphs = 0

    for batch, response in zip(dataLoader, responseInfo):
        batch = batch.to(device)
        num_graph = batch.num_graphs
        
        # Extract raw channel data for recomputation
        num_UEs = batch['UE'].x.shape[0] // num_graph
        num_APs = batch['AP'].x.shape[0] // num_graph
        tau_val = batch['UE'].x.shape[1]  # Pilot length
        
        for epoch in range(num_epochs):  # Or add this as outer loop
            optimizer.zero_grad()
            x_dict, attr_dict, _ = model(batch, isRawData=False)
            
            # RECOMPUTE DS/PC/UI with current power allocation
            power_matrix_raw = attr_dict[('AP','down','UE')].reshape(
                num_graph, num_APs, num_UEs, -1
            )[:,:,:,-1]
            
            channel_variance = batch['AP', 'down', 'UE'].edge_attr.reshape(
                num_graph, num_APs, num_UEs, -1
            )[:,:,:,1]
            
            large_scale = batch['AP', 'down', 'UE'].edge_attr.reshape(
                num_graph, num_APs, num_UEs, -1
            )[:,:,:,0]
            large_scale = torch.expm1(large_scale)
            
            power_matrix = power_from_raw(power_matrix_raw, channel_variance, num_antenna=1)
            DS_k, PC_k, UI_k = component_calculate(
                power_matrix, channel_variance, large_scale, 
                batch['UE'].x[:,:tau_val].reshape(num_graph, num_UEs, -1),
                rho_d=rho_d
            )
            
            # Use recomputed local + server's remote info
            loss, min_rate = loss_function(
                batch, x_dict, attr_dict, response,
                tau=tau_val, rho_p=rho_p, rho_d=rho_d, num_antenna=num_antenna,
                round_ratio=round_ratio
            )
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * num_graph
            total_min_rate += min_rate.mean().item() * num_graph
            total_graphs += num_graph

    return total_loss/total_graphs, total_min_rate/total_graphs
```

2. Run with `--num_epochs 3` (more local epochs)

3. **Expected result:** If staleness is an issue, using more local epochs WITHOUT recomputation should hurt performance. WITH recomputation, it should improve.

---

### Experiment 4: Test Loss Function Alignment

**Hypothesis:** Training loss (soft-min) doesn't align with eval metric (hard-min).

**Add logging** in `loss_function()` at [fl_train.py:60-76]:

```python
# After computing rate
rate = rate_from_component(all_DS, all_PC, all_UI, num_antenna)

# Log both metrics
min_rate_hard, _ = torch.min(rate.detach(), dim=1)
temperature = 2
min_rate_soft = -torch.logsumexp(-rate.detach() / temperature, dim=1) * temperature

# Print diagnostics
if batch_idx % 10 == 0:
    print(f"Hard-min rate: {min_rate_hard.mean():.4f}, std: {min_rate_hard.std():.4f}")
    print(f"Soft-min rate: {min_rate_soft.mean():.4f}, std: {min_rate_soft.std():.4f}")
    print(f"Gap: {(min_rate_hard - min_rate_soft).mean():.4f}")
```

**Expected output:**
- If hard-min and soft-min diverge significantly, the model is being trained toward soft-min optima, evaluated on hard-min metric → mismatch

---

### Experiment 5: Verify Communication Overhead

**Hypothesis:** Maybe the issue is that each AP is working with **partial state** of the system.

**Test:** Simulate "perfect communication" scenario

Create a variant `server_return_perfect()` that sends:
- Full per-UE DS/PC/UI from all other APs (not aggregate statistics)
- Per-UE interference matrix instead of mean/max/std

```python
def server_return_perfect(dataLoader, globalInformation, num_antenna=1):
    """Send complete information instead of aggregates"""
    num_client = len(globalInformation)
    response_all = []

    for batch_idx, (all_loader, all_response) in enumerate(zip(zip(*dataLoader), zip(*globalInformation))):
        aug_batch_list = []
        
        for client_id, (_, batch) in enumerate(zip(all_response, all_loader)):
            aug_batch = batch.clone()
            device = aug_batch['UE'].x.device

            # Send FULL information from other APs
            other_pack = []
            keys_needed = ['DS', 'PC', 'UI']
            for j in range(num_client):
                if j != client_id:
                    full_data = all_response[j]
                    filtered_data = {k: full_data[k] for k in keys_needed}
                    other_pack.append(filtered_data)

            # No augmentation - just pass through batch as-is
            client_data = {
                'loader': aug_batch,
                'rate_pack': other_pack  # Full, not aggregated
            }
            aug_batch_list.append(client_data)
        response_all.append(aug_batch_list)
    return response_all
```

Then in FlGrad.py, use this instead of `server_return_GAP()`:
```python
response_from_server = server_return_perfect(train_loader, send_to_server, num_antenna=num_antenna)
```

**Expected result:** FL-GNN should approach centralized performance if perfect information is given. If it doesn't, the issue is deeper (in the loss function or gradient flow, not information aggregation).

---

### Experiment 6: Diagnose Bottleneck Type

**Test:** Add detailed logging of what bottleneck the model sees

In `server_return_GAP()`, add:

```python
# After computing global_rate
print(f"Global rate shape: {global_rate.shape}")
print(f"Global rate per AP: {global_rate.mean(dim=0)}")  # [K_AP]

# After bottleneck_indicator
print(f"Bottleneck AP indicators: {bottleneck_indicator.mean(dim=0)}")
print(f"AP with highest focus: {bottleneck_indicator.mean(dim=0).argmax()}")

# The key question: Is this AP indicator well-aligned with actual per-UE bottlenecks?
# To check, we need to compute per-UE rates:
sum_DS_per_ue = all_DS_stack.sum(dim=1)  # [B, K_UE]
sum_PC_per_ue = all_PC_stack.sum(dim=1)
sum_UI_per_ue = all_UI_stack.sum(dim=1)
global_rate_per_ue = rate_from_component(...)  # [B, K_UE]
print(f"Per-UE rates: {global_rate_per_ue.mean(dim=0)}")  # [K_UE]
print(f"UE with lowest rate: {global_rate_per_ue.mean(dim=0).argmin()}")
```

**What to look for:**
- Is the AP with highest bottleneck indicator the one serving the lowest-rate UE?
- Or are they misaligned?

---

## Diagnostic Outputs to Examine

### Add this to check model behavior:

```python
# In fl_train.py, after computing loss_function(), add:

if not isTrain:
    print(f"\n=== Diagnostic Info (Sample) ===")
    print(f"Rate shape: {rate.shape}")
    print(f"Rate per UE (min/mean/max): {rate.min():.4f} / {rate.mean():.4f} / {rate.max():.4f}")
    print(f"Min rate (bottleneck): {torch.min(rate, dim=1)[0].mean():.4f}")
    print(f"Per-AP contribution (DS): {all_DS[0].mean():.6f}")
    print(f"Per-AP PC: {all_PC[0].mean():.6f}")
    print(f"Per-AP UI: {all_UI[0].mean():.6f}")
    
    # Check if own power is dominant
    own_ds_ratio = all_DS[0].sum() / sum([d.sum() for d in all_DS])
    print(f"Own AP DS ratio of total: {own_ds_ratio:.2%}")
```

**What to look for:**
- If own AP contributes < 50% of total DS, other APs are dominating → information sharing working
- If own AP contributes > 70% of total DS, your AP is essentially independent → FL isn't helping

---

## Key Metrics to Track

### Add to evaluation:

```python
def compute_fairness_metrics(rates):
    """rates: [B, K] tensor of rates"""
    min_rate = torch.min(rates, dim=1)[0]  # [B]
    mean_rate = torch.mean(rates, dim=1)  # [B]
    max_rate = torch.max(rates, dim=1)[0]  # [B]
    
    fairness = min_rate / (mean_rate + 1e-9)  # 0-1, higher is fairer
    efficiency = mean_rate
    worst_case = min_rate
    
    return {
        'fairness': fairness.mean().item(),  # Ratio of min to mean
        'efficiency': efficiency.mean().item(),
        'worst_case': worst_case.mean().item(),
        'rate_std': rates.std(dim=1).mean().item(),  # Lower = fairer
    }

# In evaluation loop:
metrics_fl = compute_fairness_metrics(rate_fl)
metrics_cen = compute_fairness_metrics(rate_cen)

print(f"FL  - Worst case: {metrics_fl['worst_case']:.4f}, Fairness: {metrics_fl['fairness']:.2%}")
print(f"CEN - Worst case: {metrics_cen['worst_case']:.4f}, Fairness: {metrics_cen['fairness']:.2%}")
```

---

## Expected Improvements from Each Fix

| Experiment | If True Impact | Expected Improvement |
|-----------|---|---|
| 1: Hard-min fix | Yes | 1.3 → 1.5-1.55 (+10-15%) |
| 2: UE-level bottleneck | Yes | 1.3 → 1.4 (+7%) |
| 3: Recompute per epoch | Moderate | 1.3 → 1.35-1.4 (+3-7%) |
| 4: Loss alignment | Yes | 1.3 → 1.45 (+10%) |
| 5: Perfect communication | Critical | 1.3 → 1.6+ (+20-25%) |
| 6: Expected - diagnosis | N/A | Identifies the real bottleneck |

**Note:** These improvements stack multiplicatively, not additively. If you implement fixes 1-5, you could reach:
```
1.3 × (1.10 × 1.07 × 1.05 × 1.10 × 1.15) ≈ 1.3 × 1.55 ≈ 2.0
```

But this assumes the issues are independent, which they're not.

---

